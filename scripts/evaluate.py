#!/usr/bin/env python3
"""Evaluate one checkpoint and write JSON plus CSV metrics."""

import argparse
from datetime import datetime, timezone
from pathlib import Path

from common import DEFAULT_PLAY_TASK, PROJECT_ROOT, add_task_args
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
add_task_args(parser, DEFAULT_PLAY_TASK)
parser.set_defaults(vx=0.4, vy=0.0, wz=0.0, num_envs=128)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--num_steps", type=int, default=1000)
parser.add_argument("--output_dir", type=Path, default=PROJECT_ROOT / "outputs" / "evaluation")
parser.add_argument("--disable_fabric", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import go2_locomotion_lab  # noqa: F401, E402
import gymnasium as gym
import torch
from common import apply_overrides, prepare_agent_cfg, resolve_checkpoint
from go2_locomotion_lab.evaluation import EvaluationAccumulator, collect_step_metrics, write_report
from go2_locomotion_lab.evaluation.metrics import resolve_metric_entities
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
    previous_actions = torch.zeros((env.num_envs, env.num_actions), device=env.device)
    accumulator = EvaluationAccumulator(env.num_envs, env.device)
    sensor_cfg, asset_cfg = resolve_metric_entities(env.unwrapped)
    for _ in range(args.num_steps):
        with torch.inference_mode():
            actions = policy(obs)
            values = collect_step_metrics(env.unwrapped, actions, previous_actions, sensor_cfg, asset_cfg)
            obs, _, dones, extras = env.step(actions)
            time_outs = extras.get("time_outs", torch.zeros_like(dones, dtype=torch.bool))
            accumulator.update(*values, dones, time_outs, env.unwrapped.step_dt)
            policy.reset(dones)
            previous_actions.copy_(actions)
            previous_actions[dones.bool()] = 0.0

    metrics = accumulator.compute()
    metrics.update(
        {
            "task": args.task,
            "checkpoint": str(checkpoint),
            "seed": args.seed,
            "num_envs": env.num_envs,
            "num_steps": args.num_steps,
            "vx": args.vx,
            "vy": args.vy,
            "wz": args.wz,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    json_path, csv_path = write_report(metrics, args.output_dir)
    print(f"[INFO] JSON: {json_path}")
    print(f"[INFO] CSV:  {csv_path}")
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
