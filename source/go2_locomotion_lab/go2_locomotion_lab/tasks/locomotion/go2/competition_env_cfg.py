"""Competition PPO baseline environment specialization."""

from isaaclab.utils import configclass

from .base_env_cfg import BaseGo2LocomotionEnvCfg


@configclass
class CompetitionBaselineEnvCfg(BaseGo2LocomotionEnvCfg):
    """Competition-PPO-Baseline training environment."""

    pass


@configclass
class CompetitionBaselineEnvCfg_PLAY(CompetitionBaselineEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.env_spacing = 2.5
        self.scene.terrain.max_init_terrain_level = None
        self.scene.terrain.terrain_generator.num_rows = 5
        self.scene.terrain.terrain_generator.num_cols = 5
        self.scene.terrain.terrain_generator.curriculum = False
        self.curriculum.terrain_levels = None
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None


@configclass
class CompetitionHighStepEnvCfg(CompetitionBaselineEnvCfg):
    """Reserved for Phase B; no high-step rewards are enabled in Phase A."""

    IMPLEMENTED = False

