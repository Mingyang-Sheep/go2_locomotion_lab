"""HIM locomotion environment built on the shared Go2 physical contract."""

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from .. import mdp
from .base_env_cfg import ORDERED_JOINTS, BaseGo2LocomotionEnvCfg

HIM_HISTORY_STEPS = 6
HIM_POLICY_OBSERVATION_DIM = 45 * HIM_HISTORY_STEPS


@configclass
class HimObservationsCfg:
    """Time-major actor history and privileged critic state."""

    @configclass
    class PolicyCfg(ObsGroup):
        proprio_history = ObsTerm(
            func=mdp.proprio45,
            params={"asset_cfg": ORDERED_JOINTS},
            history_length=HIM_HISTORY_STEPS,
            flatten_history_dim=True,
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        proprio = ObsTerm(
            func=mdp.proprio45,
            params={"asset_cfg": ORDERED_JOINTS},
        )
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class HimBaselineEnvCfg(BaseGo2LocomotionEnvCfg):
    """HIM baseline with six proprioceptive frames and privileged critic data."""

    observations: HimObservationsCfg = HimObservationsCfg()


@configclass
class HimBaselineEnvCfg_PLAY(HimBaselineEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.max_init_terrain_level = None
        self.scene.terrain.terrain_generator.num_rows = 5
        self.scene.terrain.terrain_generator.num_cols = 5
        self.scene.terrain.terrain_generator.curriculum = False
        self.curriculum.terrain_levels = None
        self.events.base_external_force_torque = None
        self.events.push_robot = None
