#!/usr/bin/env python3
"""Isaac Sim smoke test for reset, contracts, finite rewards, and stepping."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--num_steps", type=int, default=8)
parser.add_argument("--task", default="Go2-Locomotion-Competition-Baseline-v0")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

launcher = AppLauncher(args)
simulation_app = launcher.app

import gymnasium as gym
import torch

from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

import go2_locomotion_lab  # noqa: F401, E402
from go2_locomotion_lab.tasks.locomotion.go2 import ACTION_DIM, PROPRIO_OBSERVATION_DIM, set_fixed_command


def main() -> None:
    env_cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = 42
    env_cfg.sim.device = args.device or env_cfg.sim.device
    set_fixed_command(env_cfg, 0.4, 0.0, 0.0)
    env = gym.make(args.task, cfg=env_cfg)
    obs, _ = env.reset()
    assert obs["policy"].shape == (args.num_envs, PROPRIO_OBSERVATION_DIM), obs["policy"].shape
    assert env.unwrapped.action_manager.total_action_dim == ACTION_DIM
    actions = torch.zeros((args.num_envs, ACTION_DIM), device=env.unwrapped.device)
    for _ in range(args.num_steps):
        obs, reward, terminated, truncated, _ = env.step(actions)
        assert obs["policy"].shape == (args.num_envs, PROPRIO_OBSERVATION_DIM)
        assert torch.isfinite(obs["policy"]).all()
        assert torch.isfinite(reward).all()
        assert terminated.shape == (args.num_envs,)
        assert truncated.shape == (args.num_envs,)
    print(
        f"SMOKE_OK num_envs={args.num_envs} obs={tuple(obs['policy'].shape)} "
        f"actions={ACTION_DIM} reward_finite=True"
    )
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
