#!/usr/bin/env python3
"""Play a Competition PPO checkpoint."""

import argparse
import time

from common import DEFAULT_PLAY_TASK, add_task_args
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
add_task_args(parser, DEFAULT_PLAY_TASK)
parser.set_defaults(vx=0.4, vy=0.0, wz=0.0)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--num_steps", type=int, default=None)
parser.add_argument("--real_time", action="store_true")
parser.add_argument("--disable_fabric", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import go2_locomotion_lab  # noqa: F401, E402
import gymnasium as gym
import torch
from common import apply_overrides, prepare_agent_cfg, resolve_checkpoint
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
from rsl_rl.runners import OnPolicyRunner


def main() -> None:
    env_cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
    agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    agent_cfg = prepare_agent_cfg(agent_cfg)
    apply_overrides(env_cfg, args)
    agent_cfg.device = args.device or env_cfg.sim.device
    checkpoint = resolve_checkpoint(args.checkpoint, agent_cfg.experiment_name, None, None)
    env = RslRlVecEnvWrapper(gym.make(args.task, cfg=env_cfg), clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(str(checkpoint))
    policy = runner.get_inference_policy(device=env.device)
    obs = env.get_observations()
    step = 0
    while simulation_app.is_running() and (args.num_steps is None or step < args.num_steps):
        start = time.time()
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            policy.reset(dones)
        step += 1
        if args.real_time:
            time.sleep(max(0.0, env.unwrapped.step_dt - (time.time() - start)))
    print(f"[INFO] Loaded {checkpoint}; completed {step} play steps")
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
