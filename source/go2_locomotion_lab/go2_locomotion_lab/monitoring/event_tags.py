"""TensorBoard scalar tag contract and event-file validation."""

from __future__ import annotations

from pathlib import Path

PPO_TENSORBOARD_TAGS = (
    "Loss/value",
    "Loss/surrogate",
    "Loss/entropy",
    "Loss/learning_rate",
    "Policy/mean_std",
    "Train/mean_reward",
    "Train/mean_episode_length",
    "Perf/total_fps",
)

LOCOMOTION_TENSORBOARD_TAGS = (
    "Metrics/velocity/lin_xy_error",
    "Metrics/velocity/yaw_error",
    "Metrics/stability/base_height",
    "Metrics/stability/projected_gravity_xy",
    "Metrics/stability/roll_pitch_ang_vel",
    "Metrics/stability/vertical_velocity",
    "Metrics/control/action_rate",
    "Metrics/control/torque_mean",
    "Metrics/control/torque_peak",
    "Metrics/contact/feet_slide",
    "Metrics/contact/feet_stumble",
    "Metrics/termination/fall_rate",
    "Metrics/termination/timeout_rate",
)

BASELINE_REWARD_TENSORBOARD_TAGS = tuple(
    f"Episode_Reward/{name}"
    for name in (
        "track_lin_vel_xy",
        "track_ang_vel_z",
        "lin_vel_z",
        "ang_vel_xy",
        "flat_orientation",
        "joint_torques",
        "joint_acc",
        "action_rate",
        "dof_pos_limits",
        "feet_air_time",
        "feet_slide",
        "feet_stumble",
        "undesired_contacts",
        "termination",
    )
)

BASELINE_TERMINATION_TENSORBOARD_TAGS = (
    "Episode_Termination/time_out",
    "Episode_Termination/base_contact",
)

COMMON_TENSORBOARD_TAGS = (
    *PPO_TENSORBOARD_TAGS,
    *LOCOMOTION_TENSORBOARD_TAGS,
    *BASELINE_REWARD_TENSORBOARD_TAGS,
    *BASELINE_TERMINATION_TENSORBOARD_TAGS,
)

HIM_TENSORBOARD_TAGS = (
    "Loss/him_velocity",
    "Loss/him_swap",
)


def read_scalar_tags(run_dir: str | Path) -> set[str]:
    """Load scalar tags from one RSL-RL run directory."""

    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    run_path = Path(run_dir).expanduser().resolve()
    if not run_path.is_dir():
        raise FileNotFoundError(f"TensorBoard run directory not found: {run_path}")
    if not any(run_path.glob("events.out.tfevents.*")):
        raise FileNotFoundError(f"No TensorBoard event file found in: {run_path}")
    accumulator = EventAccumulator(str(run_path), size_guidance={"scalars": 0})
    accumulator.Reload()
    return set(accumulator.Tags()["scalars"])


def validate_scalar_tags(run_dir: str | Path, route: str = "competition") -> tuple[set[str], set[str]]:
    """Return ``(present, missing)`` for the route's required scalar tags."""

    if route not in {"competition", "him"}:
        raise ValueError(f"Unsupported route: {route}")
    present = read_scalar_tags(run_dir)
    required = set(COMMON_TENSORBOARD_TAGS)
    if route == "him":
        required.update(HIM_TENSORBOARD_TAGS)
    return present, required - present
