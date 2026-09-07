"""Export a self-describing Go2 policy deployment contract from Isaac Lab."""

from __future__ import annotations

import json
from pathlib import Path

from ..assets import GO2_JOINT_NAMES, GO2_POLICY_JOINT_NAMES
from ..assets.go2 import OFFICIAL_GO2_ACTION_SCALE


def export_deployment_config(env, output_dir: str | Path, route: str) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    asset = env.unwrapped.scene["robot"]
    joint_ids, resolved_names = asset.find_joints(list(GO2_JOINT_NAMES), preserve_order=True)
    if tuple(resolved_names) != tuple(GO2_JOINT_NAMES):
        raise RuntimeError(f"Go2 joint resolution mismatch: {resolved_names}")

    def values(tensor):
        return tensor[0, joint_ids].detach().cpu().tolist()

    actuator = asset.actuators["base_legs"]

    history_steps = 6 if route == "him" else 1
    command_ranges = env.unwrapped.cfg.commands.base_velocity.ranges
    payload = {
        "schema_version": 1,
        "route": route,
        "policy_file": "policy.onnx",
        "observation": {
            "frame_dim": 45,
            "history_steps": history_steps,
            "history_order": "oldest_to_newest",
            "terms": [
                ["base_ang_vel", 3],
                ["projected_gravity", 3],
                ["velocity_command", 3],
                ["joint_pos_rel", 12],
                ["joint_vel_rel", 12],
                ["previous_action", 12],
            ],
            "quaternion_order": "wxyz",
        },
        "control": {
            "mode": "joint_position_pd",
            "dt": float(env.unwrapped.step_dt),
            "action_scale": [OFFICIAL_GO2_ACTION_SCALE] * 12,
            "command_ranges": [
                list(command_ranges.lin_vel_x),
                list(command_ranges.lin_vel_y),
                list(command_ranges.ang_vel_z),
            ],
        },
        "robot": {
            "name": "Unitree Go2",
            "joint_names": list(GO2_POLICY_JOINT_NAMES),
            "usd_joint_names": list(GO2_JOINT_NAMES),
            "motor_sdk_indices": list(range(12)),
            "default_joint_pos": values(asset.data.default_joint_pos),
            "stiffness": values(asset.data.default_joint_stiffness),
            "damping": values(asset.data.default_joint_damping),
            "joint_pos_limits": asset.data.soft_joint_pos_limits[0, joint_ids].detach().cpu().tolist(),
            # Explicit DC motors leave PhysX's raw effort limit effectively unlimited.
            # These limits come from the active Isaac Lab actuator model instead.
            "effort_limits": values(actuator.effort_limit),
            "velocity_limits": values(actuator.velocity_limit),
        },
    }
    path = output_dir / "deploy.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path
