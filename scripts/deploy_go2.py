#!/usr/bin/env python3
"""Safety-gated Unitree SDK2 runtime for an exported HIM Go2 policy."""

from __future__ import annotations

import argparse
import threading
import time
from pathlib import Path

import numpy as np
from go2_locomotion_lab.deployment import DeploymentConfig, ObservationHistory, OnnxPolicy, build_proprio


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-dir", type=Path, required=True)
    parser.add_argument("--network", default=None)
    parser.add_argument("--vx", type=float, default=0.0)
    parser.add_argument("--vy", type=float, default=0.0)
    parser.add_argument("--wz", type=float, default=0.0)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--dry-run", action="store_true", help="Run inference without importing or commanding SDK2")
    parser.add_argument("--arm", action="store_true", help="Required to publish low-level motor commands")
    parser.add_argument("--max-start-error", type=float, default=0.5)
    parser.add_argument("--max-tilt-deg", type=float, default=45.0)
    return parser.parse_args()


def dry_run(config: DeploymentConfig, policy: OnnxPolicy, args: argparse.Namespace) -> None:
    history = ObservationHistory(config.history_steps)
    previous_action = np.zeros(12, dtype=np.float32)
    iterations = max(1, round(args.duration / config.control_dt))
    for _ in range(iterations):
        frame = build_proprio(
            np.zeros(3),
            np.array((1.0, 0.0, 0.0, 0.0)),
            np.array((args.vx, args.vy, args.wz)),
            config.default_joint_pos,
            np.zeros(12),
            previous_action,
            config.default_joint_pos,
        )
        previous_action = policy(history.append(frame))[0]
    print(f"DRY_RUN_OK iterations={iterations} action_finite={np.isfinite(previous_action).all()}")


def run_robot(config: DeploymentConfig, policy: OnnxPolicy, args: argparse.Namespace) -> None:
    if not args.arm:
        raise RuntimeError(
            "Real-robot output is locked. Re-run with --arm only after completing the deployment checklist."
        )
    if not args.network:
        raise ValueError("--network is required for real-robot deployment")
    try:
        from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
        from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_, LowState_
        from unitree_sdk2py.utils.crc import CRC
    except ImportError as exc:
        raise RuntimeError("Install Unitree's unitree_sdk2_python on the deployment computer") from exc

    ChannelFactoryInitialize(0, args.network)
    state_lock = threading.Lock()
    latest = {"message": None, "time": 0.0}

    def receive(message) -> None:
        with state_lock:
            latest["message"] = message
            latest["time"] = time.monotonic()

    subscriber = ChannelSubscriber("rt/lowstate", LowState_)
    subscriber.Init(receive, 10)
    publisher = ChannelPublisher("rt/lowcmd", LowCmd_)
    publisher.Init()
    deadline = time.monotonic() + 5.0
    while latest["message"] is None and time.monotonic() < deadline:
        time.sleep(0.01)
    if latest["message"] is None:
        raise RuntimeError("No Go2 low-state message received within 5 seconds")

    with state_lock:
        initial_state = latest["message"]
    initial_position = np.array([initial_state.motor_state[i].q for i in config.motor_sdk_indices])
    if np.max(np.abs(initial_position - config.default_joint_pos)) > args.max_start_error:
        raise RuntimeError("Robot is not close enough to the exported default pose; refusing to arm")

    switcher = MotionSwitcherClient()
    switcher.SetTimeout(5.0)
    switcher.Init()
    _, mode = switcher.CheckMode()
    if mode.get("name"):
        raise RuntimeError(
            f"Go2 motion service is still active ({mode['name']!r}); disable it explicitly before low-level control"
        )

    command_message = unitree_go_msg_dds__LowCmd_()
    command_message.head[0] = 0xFE
    command_message.head[1] = 0xEF
    command_message.level_flag = 0xFF
    crc = CRC()
    history = ObservationHistory(config.history_steps)
    previous_action = np.zeros(12, dtype=np.float32)
    command = config.validate_command((args.vx, args.vy, args.wz))
    stop_time = time.monotonic() + args.duration
    max_projected_xy = np.sin(np.deg2rad(args.max_tilt_deg))

    while time.monotonic() < stop_time:
        loop_start = time.monotonic()
        with state_lock:
            state = latest["message"]
            state_time = latest["time"]
        if state is None or loop_start - state_time > 0.1:
            raise RuntimeError("Go2 low-state stream timed out")
        position = np.array([state.motor_state[i].q for i in config.motor_sdk_indices], dtype=np.float32)
        velocity = np.array([state.motor_state[i].dq for i in config.motor_sdk_indices], dtype=np.float32)
        quaternion = np.array(state.imu_state.quaternion, dtype=np.float32)
        gyro = np.array(state.imu_state.gyroscope, dtype=np.float32)
        frame = build_proprio(gyro, quaternion, command, position, velocity, previous_action, config.default_joint_pos)
        if np.linalg.norm(frame[3:5]) > max_projected_xy:
            raise RuntimeError("Tilt safety threshold exceeded")
        action = policy(history.append(frame))[0]
        target = config.action_to_joint_target(action)
        max_step = config.velocity_limits * config.control_dt
        target = np.clip(target, position - max_step, position + max_step)
        for policy_index, motor_index in enumerate(config.motor_sdk_indices):
            motor = command_message.motor_cmd[motor_index]
            motor.mode = 0x01
            motor.q = float(target[policy_index])
            motor.dq = 0.0
            motor.kp = float(config.stiffness[policy_index])
            motor.kd = float(config.damping[policy_index])
            motor.tau = 0.0
        command_message.crc = crc.Crc(command_message)
        publisher.Write(command_message)
        previous_action = action
        remaining = config.control_dt - (time.monotonic() - loop_start)
        if remaining > 0.0:
            time.sleep(remaining)


def main() -> None:
    args = parse_args()
    config = DeploymentConfig.load(args.policy_dir / "deploy.json")
    policy = OnnxPolicy(args.policy_dir / "policy.onnx")
    if args.dry_run:
        dry_run(config, policy, args)
    else:
        run_robot(config, policy, args)


if __name__ == "__main__":
    main()
