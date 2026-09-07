#!/usr/bin/env python3
"""Train Competition-PPO-Baseline with the official RSL-RL runner."""

import argparse
from datetime import datetime

from common import PROJECT_ROOT, add_task_args
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
add_task_args(parser)
parser.add_argument("--max_iterations", type=int, default=None)
parser.add_argument("--resume", action="store_true")
parser.add_argument("--checkpoint", default=None)
parser.add_argument("--load_run", default=None)
parser.add_argument("--load_checkpoint", default=None)
parser.add_argument("--run_name", default=None)
parser.add_argument("--disable_fabric", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import go2_locomotion_lab  # noqa: F401, E402
import gymnasium as gym
from common import apply_overrides, prepare_agent_cfg, resolve_checkpoint
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
from rsl_rl.runners import OnPolicyRunner


def main() -> None:
    env_cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
    agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    agent_cfg = prepare_agent_cfg(agent_cfg)
    apply_overrides(env_cfg, args)
    agent_cfg.seed = args.seed
    agent_cfg.device = args.device or env_cfg.sim.device
    if args.max_iterations is not None:
        agent_cfg.max_iterations = args.max_iterations
    if args.run_name:
        agent_cfg.run_name = args.run_name
    agent_cfg.resume = args.resume
    if args.load_run is not None:
        agent_cfg.load_run = args.load_run
    if args.load_checkpoint is not None:
        agent_cfg.load_checkpoint = args.load_checkpoint

    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        run_name += f"_{agent_cfg.run_name}"
    log_dir = PROJECT_ROOT / "logs" / "rsl_rl" / agent_cfg.experiment_name / run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    env_cfg.log_dir = str(log_dir)
    print(f"[INFO] Logs and checkpoints: {log_dir}", flush=True)
    print("[INFO] Writing reproducibility configs", flush=True)
    dump_yaml(str(log_dir / "params" / "env.yaml"), env_cfg)
    dump_yaml(str(log_dir / "params" / "agent.yaml"), agent_cfg)

    print("[INFO] Creating Isaac Lab environment", flush=True)
    env = gym.make(args.task, cfg=env_cfg)
    print("[INFO] Creating RSL-RL wrapper", flush=True)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    print("[INFO] Creating RSL-RL runner", flush=True)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=str(log_dir), device=agent_cfg.device)
    runner.add_git_repo_to_log(__file__)
    if args.resume:
        resume_path = resolve_checkpoint(
            args.checkpoint, agent_cfg.experiment_name, args.load_run, args.load_checkpoint
        )
        print(f"[INFO] Resuming from {resume_path}")
        runner.load(str(resume_path))
    print("[INFO] Starting PPO learn", flush=True)
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
