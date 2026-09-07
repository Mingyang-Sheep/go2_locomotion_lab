"""Observation implementations for the stable locomotion contract."""

from __future__ import annotations

from isaaclab.envs import mdp as isaac_mdp
from isaaclab.managers import SceneEntityCfg


def base_ang_vel(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    return isaac_mdp.base_ang_vel(env, asset_cfg)


def projected_gravity(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    return isaac_mdp.projected_gravity(env, asset_cfg)


def velocity_command(env, command_name: str = "base_velocity"):
    return isaac_mdp.generated_commands(env, command_name)


def joint_pos(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    return isaac_mdp.joint_pos_rel(env, asset_cfg)


def joint_vel(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    return isaac_mdp.joint_vel_rel(env, asset_cfg)


def previous_action(env, action_name: str | None = None):
    return isaac_mdp.last_action(env, action_name)


def height_scan(env, sensor_cfg: SceneEntityCfg, offset: float = 0.5):
    """Return the native height scanner vector without reshaping it to 256."""

    return isaac_mdp.height_scan(env, sensor_cfg, offset)

