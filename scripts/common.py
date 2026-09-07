"""Shared command-line and checkpoint helpers."""

from __future__ import annotations

import argparse
import importlib.metadata
from pathlib import Path

DEFAULT_TASK = "Go2-Locomotion-Competition-Baseline-v0"
DEFAULT_PLAY_TASK = "Go2-Locomotion-Competition-Baseline-Play-v0"
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def add_task_args(parser: argparse.ArgumentParser, default_task: str = DEFAULT_TASK) -> None:
    parser.add_argument("--task", default=default_task)
    parser.add_argument("--num_envs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--vx", type=float, default=None, help="Fixed x velocity command in m/s.")
    parser.add_argument("--vy", type=float, default=None, help="Fixed y velocity command in m/s.")
    parser.add_argument("--wz", type=float, default=None, help="Fixed yaw velocity command in rad/s.")


def apply_overrides(env_cfg, args) -> None:
    if args.num_envs is not None:
        env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed
    if getattr(args, "device", None) is not None:
        env_cfg.sim.device = args.device
    if any(value is not None for value in (args.vx, args.vy, args.wz)):
        from go2_locomotion_lab.tasks.locomotion.go2 import set_fixed_command

        set_fixed_command(env_cfg, args.vx or 0.0, args.vy or 0.0, args.wz or 0.0)


def resolve_checkpoint(checkpoint: str | None, experiment_name: str, load_run: str | None, load_checkpoint: str | None):
    if checkpoint:
        path = Path(checkpoint).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return path
    from isaaclab_tasks.utils import get_checkpoint_path

    log_root = PROJECT_ROOT / "logs" / "rsl_rl" / experiment_name
    return Path(get_checkpoint_path(str(log_root), load_run, load_checkpoint))


def prepare_agent_cfg(agent_cfg):
    """Apply Isaac Lab's official RSL-RL version compatibility conversion."""

    from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg

    version = importlib.metadata.version("rsl-rl-lib")
    return handle_deprecated_rsl_rl_cfg(agent_cfg, version)
