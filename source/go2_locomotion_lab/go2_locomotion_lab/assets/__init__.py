"""Shared robot and terrain assets."""

from .go2 import GO2_CFG, GO2_JOINT_NAMES, GO2_POLICY_JOINT_NAMES
from .terrain import GO2_TERRAINS_CFG, TerrainContract

__all__ = ["GO2_CFG", "GO2_JOINT_NAMES", "GO2_POLICY_JOINT_NAMES", "GO2_TERRAINS_CFG", "TerrainContract"]
