"""Reward implementations, separate from their configured weights."""

from __future__ import annotations

import torch

from isaaclab.envs import mdp as isaac_mdp
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab_tasks.manager_based.locomotion.velocity.mdp import rewards as velocity_rewards


def track_lin_vel_xy(env, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    return isaac_mdp.track_lin_vel_xy_exp(env, std, command_name, asset_cfg)


def track_ang_vel_z(env, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    return isaac_mdp.track_ang_vel_z_exp(env, std, command_name, asset_cfg)


def lin_vel_z(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    return isaac_mdp.lin_vel_z_l2(env, asset_cfg)


def ang_vel_xy(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    return isaac_mdp.ang_vel_xy_l2(env, asset_cfg)


def flat_orientation(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    return isaac_mdp.flat_orientation_l2(env, asset_cfg)


def joint_torques(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    return isaac_mdp.joint_torques_l2(env, asset_cfg)


def joint_acc(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    return isaac_mdp.joint_acc_l2(env, asset_cfg)


def action_rate(env):
    return isaac_mdp.action_rate_l2(env)


def dof_pos_limits(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    return isaac_mdp.joint_pos_limits(env, asset_cfg)


def feet_air_time(env, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float):
    return velocity_rewards.feet_air_time(env, command_name, sensor_cfg, threshold)


def feet_slide(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    return velocity_rewards.feet_slide(env, sensor_cfg, asset_cfg)


def feet_stumble(env, sensor_cfg: SceneEntityCfg, ratio: float = 4.0) -> torch.Tensor:
    """Count feet whose horizontal impact force dominates vertical support force."""

    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    horizontal = torch.linalg.vector_norm(forces[..., :2], dim=-1).amax(dim=1)
    vertical = forces[..., 2].abs().amax(dim=1)
    return (horizontal > ratio * vertical.clamp_min(1.0e-6)).float().sum(dim=1)


def undesired_contacts(env, threshold: float, sensor_cfg: SceneEntityCfg):
    return isaac_mdp.undesired_contacts(env, threshold, sensor_cfg)


def termination(env):
    return isaac_mdp.is_terminated(env)

