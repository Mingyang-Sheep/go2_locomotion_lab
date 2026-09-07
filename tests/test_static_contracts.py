"""Fast checks that do not require launching Isaac Sim."""

import ast
from pathlib import Path

ROOT = Path(__file__).parents[1]
PACKAGE = ROOT / "source" / "go2_locomotion_lab" / "go2_locomotion_lab"


def _assignments(path: Path) -> dict[str, object]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    values = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                values[node.targets[0].id] = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                pass
    return values


def test_control_contract_constants():
    values = _assignments(PACKAGE / "tasks" / "locomotion" / "go2" / "base_env_cfg.py")
    assert values["PROPRIO_OBSERVATION_DIM"] == 45
    assert values["ACTION_DIM"] == 12
    assert values["CONTROL_DT"] == 0.02
    assert values["PHYSICS_DT"] * values["DECIMATION"] == values["CONTROL_DT"]


def test_policy_joint_order():
    values = _assignments(PACKAGE / "assets" / "go2.py")
    assert values["GO2_POLICY_JOINT_NAMES"] == (
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


def test_no_out_of_scope_platform_code():
    source = "\n".join(path.read_text(encoding="utf-8") for path in PACKAGE.rglob("*.py"))
    for forbidden in ("EnvWrapper", "kaiwudrl", "RewardBridge", "NavGrid", "UWB"):
        assert forbidden not in source
