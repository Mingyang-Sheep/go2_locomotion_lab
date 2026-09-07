"""Canonical Unitree Go2 robot contract.

Motor limits, PD gains, and default pose are inherited verbatim from
``isaaclab_assets.robots.unitree.UNITREE_GO2_CFG``. This module deliberately
does not duplicate those values so upstream asset corrections remain visible.
"""

from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG

# Policy-facing order required by the platform contract. Isaac Lab's USD joint
# names include the ``_joint`` suffix; the semantic names omit it.
GO2_POLICY_JOINT_NAMES = (
    "FR_hip",
    "FR_thigh",
    "FR_calf",
    "FL_hip",
    "FL_thigh",
    "FL_calf",
    "RR_hip",
    "RR_thigh",
    "RR_calf",
    "RL_hip",
    "RL_thigh",
    "RL_calf",
)
GO2_JOINT_NAMES = tuple(f"{name}_joint" for name in GO2_POLICY_JOINT_NAMES)

GO2_CFG = UNITREE_GO2_CFG.copy()
"""Official Isaac Lab Go2 articulation configuration."""

# Source: Isaac Lab's UnitreeGo2RoughEnvCfg. This is the only policy-to-joint
# position multiplier used by the baseline.
OFFICIAL_GO2_ACTION_SCALE = 0.25

# COMPETITION_TODO: replace only when the competition publishes authoritative
# actuator/default-pose/action-scale values. Keep all such overrides here.
COMPETITION_TODO: dict[str, object] = {}


def official_robot_contract() -> dict[str, object]:
    """Return inspectable values from the official Go2 configuration."""

    actuator = GO2_CFG.actuators["base_legs"]
    return {
        "policy_joint_order": GO2_POLICY_JOINT_NAMES,
        "usd_joint_order": GO2_JOINT_NAMES,
        "action_scale": OFFICIAL_GO2_ACTION_SCALE,
        "default_joint_position_patterns": dict(GO2_CFG.init_state.joint_pos),
        "stiffness": actuator.stiffness,
        "damping": actuator.damping,
        "effort_limit": actuator.effort_limit,
        "velocity_limit": actuator.velocity_limit,
    }
