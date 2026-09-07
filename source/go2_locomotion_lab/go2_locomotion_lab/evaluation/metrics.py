"""Physics-based locomotion metrics shared by all algorithm branches."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from isaaclab.managers import SceneEntityCfg

from ..tasks.locomotion import mdp


@dataclass
class EvaluationAccumulator:
    num_envs: int
    device: str
    sample_count: int = 0
    episode_count: int = 0
    fall_count: int = 0
    velocity_error_sum: float = 0.0
    feet_slide_sum: float = 0.0
    feet_stumble_sum: float = 0.0
    action_rate_sum: float = 0.0
    torque_sum: float = 0.0
    torque_count: int = 0
    torque_peak: float = 0.0
    episode_lengths: list[float] = field(default_factory=list)

    def __post_init__(self):
        self._current_episode_steps = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

    def update(
        self,
        velocity_error: torch.Tensor,
        feet_slide: torch.Tensor,
        feet_stumble: torch.Tensor,
        action_rate: torch.Tensor,
        torques: torch.Tensor,
        dones: torch.Tensor,
        time_outs: torch.Tensor,
        step_dt: float,
    ) -> None:
        self.sample_count += velocity_error.numel()
        self.velocity_error_sum += velocity_error.sum().item()
        self.feet_slide_sum += feet_slide.sum().item()
        self.feet_stumble_sum += feet_stumble.sum().item()
        self.action_rate_sum += action_rate.sum().item()
        self.torque_sum += torques.abs().sum().item()
        self.torque_count += torques.numel()
        self.torque_peak = max(self.torque_peak, torques.abs().max().item())
        self._current_episode_steps += 1

        done_mask = dones.bool()
        if done_mask.any():
            lengths = self._current_episode_steps[done_mask].float() * step_dt
            self.episode_lengths.extend(lengths.cpu().tolist())
            self.episode_count += int(done_mask.sum().item())
            self.fall_count += int((done_mask & ~time_outs.bool()).sum().item())
            self._current_episode_steps[done_mask] = 0

    def compute(self) -> dict[str, float | int | None]:
        samples = max(self.sample_count, 1)
        return {
            "velocity_tracking_error": self.velocity_error_sum / samples,
            "fall_rate": self.fall_count / self.episode_count if self.episode_count else None,
            "episode_length": sum(self.episode_lengths) / len(self.episode_lengths) if self.episode_lengths else None,
            "feet_slide": self.feet_slide_sum / samples,
            "feet_stumble": self.feet_stumble_sum / samples,
            "action_rate": self.action_rate_sum / samples,
            "torque_mean": self.torque_sum / max(self.torque_count, 1),
            "torque_peak": self.torque_peak,
            "episodes": self.episode_count,
            "samples": self.sample_count,
            # Reserved until Phase B defines an auditable stair protocol.
            "stair_success_rate": None,
            "max_step_height": None,
            "front_foot_clearance": None,
            "rear_foot_clearance": None,
        }


def resolve_metric_entities(env) -> tuple[SceneEntityCfg, SceneEntityCfg]:
    sensor_cfg = SceneEntityCfg("contact_forces", body_names=".*_foot")
    asset_cfg = SceneEntityCfg("robot", body_names=".*_foot")
    sensor_cfg.resolve(env.scene)
    asset_cfg.resolve(env.scene)
    return sensor_cfg, asset_cfg


def collect_step_metrics(env, actions: torch.Tensor, previous_actions: torch.Tensor, sensor_cfg, asset_cfg):
    robot = env.scene["robot"]
    command = env.command_manager.get_command("base_velocity")
    lin_error = torch.linalg.vector_norm(command[:, :2] - robot.data.root_lin_vel_b[:, :2], dim=1)
    yaw_error = (command[:, 2] - robot.data.root_ang_vel_b[:, 2]).abs()
    velocity_error = torch.sqrt(lin_error.square() + yaw_error.square())
    slide = mdp.feet_slide(env, sensor_cfg=sensor_cfg, asset_cfg=asset_cfg)
    stumble = mdp.feet_stumble(env, sensor_cfg=sensor_cfg)
    action_delta = torch.linalg.vector_norm(actions - previous_actions, dim=1)
    return velocity_error, slide, stumble, action_delta, robot.data.applied_torque
