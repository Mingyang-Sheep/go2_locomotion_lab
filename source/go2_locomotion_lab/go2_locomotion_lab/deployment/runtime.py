"""Shared NumPy runtime used by MuJoCo and Unitree SDK deployment."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class DeploymentConfig:
    route: str
    control_dt: float
    history_steps: int
    joint_names: tuple[str, ...]
    motor_sdk_indices: tuple[int, ...]
    default_joint_pos: np.ndarray
    stiffness: np.ndarray
    damping: np.ndarray
    action_scale: np.ndarray
    command_ranges: np.ndarray
    joint_pos_limits: np.ndarray
    effort_limits: np.ndarray
    velocity_limits: np.ndarray

    @classmethod
    def load(cls, path: str | Path) -> DeploymentConfig:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("schema_version") != 1:
            raise ValueError(f"Unsupported deployment schema: {payload.get('schema_version')}")
        observation = payload["observation"]
        control = payload["control"]
        robot = payload["robot"]
        config = cls(
            route=payload["route"],
            control_dt=float(control["dt"]),
            history_steps=int(observation["history_steps"]),
            joint_names=tuple(robot["joint_names"]),
            motor_sdk_indices=tuple(robot["motor_sdk_indices"]),
            default_joint_pos=np.asarray(robot["default_joint_pos"], dtype=np.float32),
            stiffness=np.asarray(robot["stiffness"], dtype=np.float32),
            damping=np.asarray(robot["damping"], dtype=np.float32),
            action_scale=np.asarray(control["action_scale"], dtype=np.float32),
            command_ranges=np.asarray(control["command_ranges"], dtype=np.float32),
            joint_pos_limits=np.asarray(robot["joint_pos_limits"], dtype=np.float32),
            effort_limits=np.asarray(robot["effort_limits"], dtype=np.float32),
            velocity_limits=np.asarray(robot["velocity_limits"], dtype=np.float32),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.route != "him":
            raise ValueError(f"Expected a HIM deployment bundle, got route={self.route!r}")
        if len(self.joint_names) != 12 or len(self.motor_sdk_indices) != 12:
            raise ValueError("Go2 deployment requires exactly 12 ordered joints")
        if self.history_steps < 1 or self.control_dt <= 0.0:
            raise ValueError("Invalid history length or control period")
        for name, value, shape in (
            ("default_joint_pos", self.default_joint_pos, (12,)),
            ("stiffness", self.stiffness, (12,)),
            ("damping", self.damping, (12,)),
            ("action_scale", self.action_scale, (12,)),
            ("command_ranges", self.command_ranges, (3, 2)),
            ("joint_pos_limits", self.joint_pos_limits, (12, 2)),
            ("effort_limits", self.effort_limits, (12,)),
            ("velocity_limits", self.velocity_limits, (12,)),
        ):
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f"Invalid {name}: expected finite shape {shape}, got {value.shape}")

    def validate_command(self, command: np.ndarray) -> np.ndarray:
        command = np.asarray(command, dtype=np.float32).reshape(3)
        if np.any(command < self.command_ranges[:, 0]) or np.any(command > self.command_ranges[:, 1]):
            raise ValueError(f"Command {command.tolist()} is outside exported training ranges")
        return command

    def action_to_joint_target(self, action: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=np.float32).reshape(12)
        target = self.default_joint_pos + self.action_scale * action
        return np.clip(target, self.joint_pos_limits[:, 0], self.joint_pos_limits[:, 1])


class ObservationHistory:
    """Oldest-to-newest history matching Isaac Lab's CircularBuffer layout."""

    def __init__(self, steps: int, frame_dim: int = 45) -> None:
        self.steps = steps
        self.frame_dim = frame_dim
        self._frames = np.zeros((steps, frame_dim), dtype=np.float32)
        self._initialized = False

    def reset(self, frame: np.ndarray) -> np.ndarray:
        frame = np.asarray(frame, dtype=np.float32).reshape(self.frame_dim)
        self._frames[:] = frame
        self._initialized = True
        return self.value

    def append(self, frame: np.ndarray) -> np.ndarray:
        frame = np.asarray(frame, dtype=np.float32).reshape(self.frame_dim)
        if not self._initialized:
            return self.reset(frame)
        self._frames[:-1] = self._frames[1:]
        self._frames[-1] = frame
        return self.value

    @property
    def value(self) -> np.ndarray:
        return self._frames.reshape(1, -1)


class OnnxPolicy:
    def __init__(self, model_path: str | Path) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("Install the optional 'onnxruntime' dependency to run deployment") from exc
        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        if len(inputs) != 1 or len(outputs) != 1:
            raise ValueError("HIM deployment policy must have one history input and one action output")
        self.input_name = inputs[0].name
        self.output_name = outputs[0].name
        self.input_shape = inputs[0].shape

    def __call__(self, history: np.ndarray) -> np.ndarray:
        history = np.asarray(history, dtype=np.float32)
        output = self.session.run([self.output_name], {self.input_name: history})[0]
        output = np.asarray(output, dtype=np.float32)
        if output.shape != (history.shape[0], 12) or not np.isfinite(output).all():
            raise RuntimeError(f"Policy produced invalid action shape/value: {output.shape}")
        return output


def projected_gravity(quaternion_wxyz: np.ndarray) -> np.ndarray:
    """Rotate the world gravity unit vector into the body frame."""

    q = np.asarray(quaternion_wxyz, dtype=np.float64).reshape(4)
    norm = np.linalg.norm(q)
    if norm < 1.0e-8:
        raise ValueError("Invalid zero-norm IMU quaternion")
    w, x, y, z = q / norm
    rotation = np.array(
        (
            (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
        )
    )
    return (rotation.T @ np.array((0.0, 0.0, -1.0))).astype(np.float32)


def build_proprio(
    angular_velocity: np.ndarray,
    quaternion_wxyz: np.ndarray,
    command: np.ndarray,
    joint_position: np.ndarray,
    joint_velocity: np.ndarray,
    previous_action: np.ndarray,
    default_joint_position: np.ndarray,
) -> np.ndarray:
    """Build the exact 45-D actor frame used during training."""

    frame = np.concatenate(
        (
            np.asarray(angular_velocity, dtype=np.float32).reshape(3),
            projected_gravity(quaternion_wxyz),
            np.asarray(command, dtype=np.float32).reshape(3),
            np.asarray(joint_position, dtype=np.float32).reshape(12)
            - np.asarray(default_joint_position, dtype=np.float32).reshape(12),
            np.asarray(joint_velocity, dtype=np.float32).reshape(12),
            np.asarray(previous_action, dtype=np.float32).reshape(12),
        )
    ).astype(np.float32)
    if frame.shape != (45,) or not np.isfinite(frame).all():
        raise ValueError("Non-finite or malformed proprioception frame")
    return frame
