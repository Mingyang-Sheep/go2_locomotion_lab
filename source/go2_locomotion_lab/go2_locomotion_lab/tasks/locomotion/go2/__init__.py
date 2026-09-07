"""Gymnasium registrations for the parallel Competition and HIM routes."""

import gymnasium as gym

gym.register(
    id="Go2-Locomotion-Competition-Baseline-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.competition_env_cfg:CompetitionBaselineEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.competition_ppo_cfg:CompetitionPpoBaselineRunnerCfg",
    },
)

gym.register(
    id="Go2-Locomotion-HIM-Baseline-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.him_env_cfg:HimBaselineEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.him_ppo_cfg:HimPpoBaselineRunnerCfg",
    },
)

gym.register(
    id="Go2-Locomotion-HIM-Baseline-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.him_env_cfg:HimBaselineEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{__name__}.him_ppo_cfg:HimPpoBaselineRunnerCfg",
    },
)

gym.register(
    id="Go2-Locomotion-Competition-Baseline-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.competition_env_cfg:CompetitionBaselineEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{__name__}.competition_ppo_cfg:CompetitionPpoBaselineRunnerCfg",
    },
)

from .base_env_cfg import ACTION_DIM, CONTROL_DT, PROPRIO_OBSERVATION_DIM, set_fixed_command  # noqa: E402
from .competition_env_cfg import CompetitionBaselineEnvCfg, CompetitionBaselineEnvCfg_PLAY  # noqa: E402
from .him_env_cfg import (  # noqa: E402
    HIM_HISTORY_STEPS,
    HIM_POLICY_OBSERVATION_DIM,
    HimBaselineEnvCfg,
    HimBaselineEnvCfg_PLAY,
)

__all__ = [
    "ACTION_DIM",
    "CONTROL_DT",
    "PROPRIO_OBSERVATION_DIM",
    "CompetitionBaselineEnvCfg",
    "CompetitionBaselineEnvCfg_PLAY",
    "HIM_HISTORY_STEPS",
    "HIM_POLICY_OBSERVATION_DIM",
    "HimBaselineEnvCfg",
    "HimBaselineEnvCfg_PLAY",
    "set_fixed_command",
]
