#!/usr/bin/env python3
"""Export a Competition PPO checkpoint to TorchScript and ONNX."""

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

from common import DEFAULT_PLAY_TASK, add_task_args

parser = argparse.ArgumentParser(description=__doc__)
add_task_args(parser, DEFAULT_PLAY_TASK)
parser.set_defaults(num_envs=1)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--output_dir", type=Path, default=None)
parser.add_argument("--disable_fabric", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

import go2_locomotion_lab  # noqa: F401, E402
from common import apply_overrides, prepare_agent_cfg, resolve_checkpoint


def main() -> None:
    env_cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
    agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    agent_cfg = prepare_agent_cfg(agent_cfg)
    apply_overrides(env_cfg, args)
    agent_cfg.device = args.device or env_cfg.sim.device
    checkpoint = resolve_checkpoint(args.checkpoint, agent_cfg.experiment_name, None, None)
    output_dir = args.output_dir or checkpoint.parent / "exported"
    env = RslRlVecEnvWrapper(gym.make(args.task, cfg=env_cfg), clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(str(checkpoint))
    runner.export_policy_to_jit(str(output_dir), filename="policy.pt")
    runner.export_policy_to_onnx(str(output_dir), filename="policy.onnx")
    print(f"[INFO] Exported policy to {output_dir}")
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
