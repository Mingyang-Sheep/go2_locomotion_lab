"""Training-time physics metrics injected into RSL-RL's normal log stream."""

from __future__ import annotations

import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCaster
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

from ..tasks.locomotion import mdp


class TrainingMetricsRslRlVecEnvWrapper(RslRlVecEnvWrapper):
    """Add algorithm-neutral locomotion metrics without changing environment behavior."""

    def __init__(self, env, clip_actions: float | None = None):
        super().__init__(env, clip_actions=clip_actions)
        self._previous_actions = torch.zeros((self.num_envs, self.num_actions), device=self.device)
        self._completed_episodes = torch.zeros((), dtype=torch.long, device=self.device)
        self._fall_episodes = torch.zeros((), dtype=torch.long, device=self.device)
        self._timeout_episodes = torch.zeros((), dtype=torch.long, device=self.device)
        self._foot_sensor_cfg = SceneEntityCfg("contact_forces", body_names=".*_foot")
        self._foot_asset_cfg = SceneEntityCfg("robot", body_names=".*_foot")
        self._foot_sensor_cfg.resolve(self.unwrapped.scene)
        self._foot_asset_cfg.resolve(self.unwrapped.scene)

    def step(self, actions: torch.Tensor):
        monitored_actions = actions
        if self.clip_actions is not None:
            monitored_actions = torch.clamp(actions, -self.clip_actions, self.clip_actions)
        metrics = self._collect_metrics(monitored_actions)

        observations, rewards, dones, extras = super().step(actions)
        terminated = self.unwrapped.reset_terminated
        time_outs = self.unwrapped.reset_time_outs
        completed = terminated | time_outs
        self._completed_episodes += completed.sum()
        self._fall_episodes += (terminated & ~time_outs).sum()
        self._timeout_episodes += time_outs.sum()
        denominator = self._completed_episodes.clamp_min(1).float()
        metrics["Metrics/termination/fall_rate"] = self._fall_episodes.float() / denominator
        metrics["Metrics/termination/timeout_rate"] = self._timeout_episodes.float() / denominator

        # Use a fresh dictionary because RSL-RL buffers references until the end of the rollout.
        log_values = dict(extras.get("log", {}))
        log_values.update(metrics)
        extras["log"] = log_values

        self._previous_actions.copy_(monitored_actions)
        self._previous_actions[dones.bool()] = 0.0
        return observations, rewards, dones, extras

    @torch.no_grad()
    def _collect_metrics(self, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        env = self.unwrapped
        robot = env.scene["robot"]
        command = env.command_manager.get_command("base_velocity")
        torques = robot.data.applied_torque.abs()

        lin_xy_error = torch.linalg.vector_norm(command[:, :2] - robot.data.root_lin_vel_b[:, :2], dim=1)
        yaw_error = (command[:, 2] - robot.data.root_ang_vel_b[:, 2]).abs()
        action_rate = torch.linalg.vector_norm(actions - self._previous_actions, dim=1)

        height_scanner: RayCaster = env.scene.sensors["height_scanner"]
        hit_height = height_scanner.data.ray_hits_w[..., 2]
        finite_hits = torch.isfinite(hit_height)
        local_ground_height = torch.where(finite_hits, hit_height, 0.0).sum(dim=1)
        local_ground_height /= finite_hits.sum(dim=1).clamp_min(1)
        base_height = robot.data.root_pos_w[:, 2] - local_ground_height

        return {
            "Metrics/velocity/lin_xy_error": lin_xy_error,
            "Metrics/velocity/yaw_error": yaw_error,
            "Metrics/stability/base_height": base_height,
            "Metrics/stability/projected_gravity_xy": torch.linalg.vector_norm(
                robot.data.projected_gravity_b[:, :2], dim=1
            ),
            "Metrics/stability/roll_pitch_ang_vel": torch.linalg.vector_norm(
                robot.data.root_ang_vel_b[:, :2], dim=1
            ),
            "Metrics/stability/vertical_velocity": robot.data.root_lin_vel_b[:, 2].abs(),
            "Metrics/control/action_rate": action_rate,
            "Metrics/control/torque_mean": torques.mean(dim=1),
            "Metrics/control/torque_peak": torques.amax(dim=1),
            "Metrics/contact/feet_slide": mdp.feet_slide(
                env, sensor_cfg=self._foot_sensor_cfg, asset_cfg=self._foot_asset_cfg
            ),
            "Metrics/contact/feet_stumble": mdp.feet_stumble(env, sensor_cfg=self._foot_sensor_cfg),
        }
