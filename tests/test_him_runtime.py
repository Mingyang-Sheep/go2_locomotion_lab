import json
from pathlib import Path

import numpy as np
from go2_locomotion_lab.deployment.runtime import DeploymentConfig, ObservationHistory, build_proprio


def test_history_is_oldest_to_newest():
    history = ObservationHistory(3, frame_dim=2)
    np.testing.assert_array_equal(history.reset(np.array([1, 2])), [[1, 2, 1, 2, 1, 2]])
    np.testing.assert_array_equal(history.append(np.array([3, 4])), [[1, 2, 1, 2, 3, 4]])


def test_upright_proprio_contract():
    frame = build_proprio(
        np.array([1, 2, 3]),
        np.array([1, 0, 0, 0]),
        np.array([0.4, 0, 0]),
        np.ones(12),
        np.zeros(12),
        np.zeros(12),
        np.ones(12),
    )
    assert frame.shape == (45,)
    np.testing.assert_array_equal(frame[3:6], [0, 0, -1])
    np.testing.assert_array_equal(frame[9:21], np.zeros(12))


def test_deployment_config_validation(tmp_path: Path):
    payload = {
        "schema_version": 1,
        "route": "him",
        "observation": {"history_steps": 6},
        "control": {
            "dt": 0.02,
            "action_scale": [0.25] * 12,
            "command_ranges": [[-1.0, 1.0], [-0.5, 0.5], [-1.0, 1.0]],
        },
        "robot": {
            "joint_names": [f"j{i}" for i in range(12)],
            "motor_sdk_indices": list(range(12)),
            "default_joint_pos": [0.0] * 12,
            "stiffness": [25.0] * 12,
            "damping": [0.5] * 12,
            "joint_pos_limits": [[-1.0, 1.0]] * 12,
            "effort_limits": [23.5] * 12,
            "velocity_limits": [30.0] * 12,
        },
    }
    path = tmp_path / "deploy.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    config = DeploymentConfig.load(path)
    np.testing.assert_array_equal(config.action_to_joint_target(np.full(12, 10.0)), np.ones(12))
    np.testing.assert_allclose(config.validate_command([0.4, 0.0, 0.0]), [0.4, 0.0, 0.0])
