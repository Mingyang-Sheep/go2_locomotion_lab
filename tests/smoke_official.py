#!/usr/bin/env python3
"""Verify that the unmodified official Go2 velocity task can reset and step."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_steps", type=int, default=2)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

launcher = AppLauncher(args)
simulation_app = launcher.app

import gymnasium as gym
import isaaclab_tasks  # noqa: F401, E402
import torch
from isaaclab_tasks.utils import parse_env_cfg


def main() -> None:
    task = "Isaac-Velocity-Rough-Unitree-Go2-Play-v0"
    cfg = parse_env_cfg(task, device=args.device, num_envs=1)
    env = gym.make(task, cfg=cfg)
    obs, _ = env.reset()
    for _ in range(args.num_steps):
        actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
        obs, rewards, _, _, _ = env.step(actions)
        assert torch.isfinite(rewards).all()
    print(f"OFFICIAL_GO2_SMOKE_OK policy_shape={tuple(obs['policy'].shape)}")
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
