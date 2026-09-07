"""Termination implementations."""

from isaaclab.envs import mdp as isaac_mdp
from isaaclab.managers import SceneEntityCfg


def time_out(env):
    return isaac_mdp.time_out(env)


def illegal_contact(env, sensor_cfg: SceneEntityCfg, threshold: float):
    return isaac_mdp.illegal_contact(env, threshold, sensor_cfg)

