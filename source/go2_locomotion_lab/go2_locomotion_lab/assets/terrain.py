"""Centralized terrain contract for every algorithm branch."""

import math
from dataclasses import dataclass

import isaaclab.terrains as terrain_gen
from isaaclab.terrains import TerrainGeneratorCfg


@dataclass(frozen=True)
class TerrainContract:
    """Human-readable source of truth used to build the Isaac Lab config."""

    difficulty_range: tuple[float, float] = (0.0, 1.0)
    num_rows: int = 10
    num_cols: int = 10
    curriculum: bool = True
    step_height_range: tuple[float, float] = (0.05, 0.15)
    supported_step_height_range: tuple[float, float] = (0.05, 0.23)
    slope_range: tuple[float, float] = (0.0, 0.35)
    proportions: tuple[float, float, float, float, float] = (0.10, 0.225, 0.225, 0.225, 0.225)


TERRAIN_CONTRACT = TerrainContract()


def make_go2_terrain_cfg(contract: TerrainContract = TERRAIN_CONTRACT) -> TerrainGeneratorCfg:
    """Build the five-terrain baseline generator without hidden parameters."""

    if not math.isclose(sum(contract.proportions), 1.0, rel_tol=0.0, abs_tol=1.0e-9):
        raise ValueError(f"Terrain proportions must sum to 1.0, got {sum(contract.proportions)}")
    flat, slope, slope_inv, stairs, stairs_inv = contract.proportions
    return TerrainGeneratorCfg(
        seed=None,
        curriculum=contract.curriculum,
        size=(8.0, 8.0),
        border_width=20.0,
        num_rows=contract.num_rows,
        num_cols=contract.num_cols,
        horizontal_scale=0.1,
        vertical_scale=0.005,
        slope_threshold=0.75,
        difficulty_range=contract.difficulty_range,
        use_cache=False,
        sub_terrains={
            "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=flat),
            "pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
                proportion=slope,
                slope_range=contract.slope_range,
                platform_width=2.0,
                border_width=0.25,
            ),
            "pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
                proportion=slope_inv,
                slope_range=contract.slope_range,
                platform_width=2.0,
                border_width=0.25,
            ),
            "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
                proportion=stairs,
                step_height_range=contract.step_height_range,
                step_width=0.3,
                platform_width=3.0,
                border_width=1.0,
                holes=False,
            ),
            "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
                proportion=stairs_inv,
                step_height_range=contract.step_height_range,
                step_width=0.3,
                platform_width=3.0,
                border_width=1.0,
                holes=False,
            ),
        },
    )


GO2_TERRAINS_CFG = make_go2_terrain_cfg()
