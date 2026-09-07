"""Shared Unitree Go2 locomotion environment configuration."""

from __future__ import annotations

import math
from dataclasses import dataclass

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as UniformNoise

from ....assets import GO2_CFG, GO2_JOINT_NAMES, GO2_TERRAINS_CFG
from ....assets.go2 import OFFICIAL_GO2_ACTION_SCALE
from .. import mdp

PROPRIO_OBSERVATION_DIM = 45
ACTION_DIM = 12
CONTROL_DT = 0.02
PHYSICS_DT = 0.005
DECIMATION = 4


@dataclass(frozen=True)
class CommandContract:
    resampling_time_s: tuple[float, float] = (10.0, 10.0)
    lin_vel_x: tuple[float, float] = (-1.0, 1.0)
    lin_vel_y: tuple[float, float] = (-0.5, 0.5)
    ang_vel_z: tuple[float, float] = (-1.0, 1.0)


COMMAND_CONTRACT = CommandContract()
ORDERED_JOINTS = SceneEntityCfg("robot", joint_names=list(GO2_JOINT_NAMES), preserve_order=True)
FOOT_SENSOR = SceneEntityCfg("contact_forces", body_names=".*_foot")
FOOT_BODIES = SceneEntityCfg("robot", body_names=".*_foot")


@configclass
class Go2SceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=GO2_TERRAINS_CFG.copy(),
        max_init_terrain_level=4,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.32, 0.36, 0.32), roughness=0.8),
        debug_vis=False,
    )
    robot = GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=(1.6, 1.0)),
        mesh_prim_paths=["/World/ground"],
        debug_vis=False,
    )
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(intensity=750.0, color=(0.9, 0.9, 0.9)),
    )


@configclass
class CommandsCfg:
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=COMMAND_CONTRACT.resampling_time_s,
        rel_standing_envs=0.02,
        rel_heading_envs=0.0,
        heading_command=False,
        debug_vis=False,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=COMMAND_CONTRACT.lin_vel_x,
            lin_vel_y=COMMAND_CONTRACT.lin_vel_y,
            ang_vel_z=COMMAND_CONTRACT.ang_vel_z,
            heading=None,
        ),
    )


@configclass
class ActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=list(GO2_JOINT_NAMES),
        preserve_order=True,
        scale=OFFICIAL_GO2_ACTION_SCALE,
        use_default_offset=True,
    )


@configclass
class ObservationsCfg:
    """Competition v1 actor and asymmetric critic observations."""

    @configclass
    class PolicyCfg(ObsGroup):
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=UniformNoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=UniformNoise(n_min=-0.05, n_max=0.05))
        velocity_command = ObsTerm(func=mdp.velocity_command, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(
            func=mdp.joint_pos,
            params={"asset_cfg": ORDERED_JOINTS},
            noise=UniformNoise(n_min=-0.01, n_max=0.01),
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel,
            params={"asset_cfg": ORDERED_JOINTS},
            noise=UniformNoise(n_min=-1.5, n_max=1.5),
        )
        previous_action = ObsTerm(func=mdp.previous_action, params={"action_name": "joint_pos"})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(PolicyCfg):
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=UniformNoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            super().__post_init__()
            self.enable_corruption = False

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class EventsCfg:
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 0.8),
            "dynamic_friction_range": (0.6, 0.6),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-1.0, 3.0),
            "operation": "add",
        },
    )
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "force_range": (0.0, 0.0),
            "torque_range": (0.0, 0.0),
        },
    )
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-math.pi, math.pi)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={"position_range": (1.0, 1.0), "velocity_range": (0.0, 0.0)},
    )
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )


@configclass
class RewardsCfg:
    """Weights for functions implemented in ``mdp/rewards.py``."""

    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy,
        weight=1.5,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z,
        weight=0.75,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    lin_vel_z = RewTerm(func=mdp.lin_vel_z, weight=-2.0)
    ang_vel_xy = RewTerm(func=mdp.ang_vel_xy, weight=-0.05)
    flat_orientation = RewTerm(func=mdp.flat_orientation, weight=-0.5)
    joint_torques = RewTerm(func=mdp.joint_torques, weight=-2.0e-4, params={"asset_cfg": ORDERED_JOINTS})
    joint_acc = RewTerm(func=mdp.joint_acc, weight=-2.5e-7, params={"asset_cfg": ORDERED_JOINTS})
    action_rate = RewTerm(func=mdp.action_rate, weight=-0.01)
    dof_pos_limits = RewTerm(func=mdp.dof_pos_limits, weight=-1.0, params={"asset_cfg": ORDERED_JOINTS})
    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=0.01,
        params={"command_name": "base_velocity", "sensor_cfg": FOOT_SENSOR, "threshold": 0.5},
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.1,
        params={"sensor_cfg": FOOT_SENSOR, "asset_cfg": FOOT_BODIES},
    )
    feet_stumble = RewTerm(func=mdp.feet_stumble, weight=-0.1, params={"sensor_cfg": FOOT_SENSOR, "ratio": 4.0})
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_thigh", ".*_calf"]),
            "threshold": 1.0,
        },
    )
    termination = RewTerm(func=mdp.termination, weight=-200.0)


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0},
    )


@configclass
class CurriculumCfg:
    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)


@configclass
class BaseGo2LocomotionEnvCfg(ManagerBasedRLEnvCfg):
    """Algorithm-neutral Go2 velocity task and physical interface."""

    scene: Go2SceneCfg = Go2SceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        self.decimation = DECIMATION
        self.episode_length_s = 20.0
        self.sim.dt = PHYSICS_DT
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        self.scene.height_scanner.update_period = CONTROL_DT
        self.scene.contact_forces.update_period = PHYSICS_DT
        self.scene.terrain.terrain_generator.curriculum = self.curriculum.terrain_levels is not None


def set_fixed_command(env_cfg: BaseGo2LocomotionEnvCfg, vx: float, vy: float, wz: float) -> None:
    """Mutate a config so every environment receives one fixed velocity command."""

    ranges = env_cfg.commands.base_velocity.ranges
    ranges.lin_vel_x = (vx, vx)
    ranges.lin_vel_y = (vy, vy)
    ranges.ang_vel_z = (wz, wz)
    env_cfg.commands.base_velocity.heading_command = False
    env_cfg.commands.base_velocity.rel_heading_envs = 0.0
    env_cfg.commands.base_velocity.rel_standing_envs = 0.0
