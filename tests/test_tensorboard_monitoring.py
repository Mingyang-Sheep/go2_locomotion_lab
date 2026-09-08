"""TensorBoard contract tests that do not launch Isaac Sim."""

from pathlib import Path

import pytest
from go2_locomotion_lab.monitoring.event_tags import (
    COMMON_TENSORBOARD_TAGS,
    HIM_TENSORBOARD_TAGS,
    validate_scalar_tags,
)


def test_route_tag_contract_is_distinct():
    assert "Loss/value" in COMMON_TENSORBOARD_TAGS
    assert "Metrics/termination/fall_rate" in COMMON_TENSORBOARD_TAGS
    assert "Loss/him_velocity" not in COMMON_TENSORBOARD_TAGS
    assert set(HIM_TENSORBOARD_TAGS).isdisjoint(COMMON_TENSORBOARD_TAGS)


def test_event_file_validation(tmp_path: Path):
    tensorboard = pytest.importorskip("torch.utils.tensorboard")
    writer = tensorboard.SummaryWriter(tmp_path)
    for tag in (*COMMON_TENSORBOARD_TAGS, *HIM_TENSORBOARD_TAGS):
        writer.add_scalar(tag, 1.0, 0)
    writer.close()

    present, missing = validate_scalar_tags(tmp_path, route="him")
    assert not missing
    assert set(COMMON_TENSORBOARD_TAGS).issubset(present)
    assert set(HIM_TENSORBOARD_TAGS).issubset(present)
