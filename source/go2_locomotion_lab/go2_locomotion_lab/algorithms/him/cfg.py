"""Isaac Lab configuration types for HIM on RSL-RL v5."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlPpoAlgorithmCfg


@configclass
class HimActorModelCfg(RslRlMLPModelCfg):
    class_name: str = "go2_locomotion_lab.algorithms.him.model:HimActorModel"
    history_steps: int = 6
    one_step_obs_dim: int = 45
    latent_dim: int = 16
    estimator_hidden_dims: list[int] = [128, 64]
    estimator_target_hidden_dims: list[int] = [128, 64]
    estimator_num_prototypes: int = 32
    estimator_learning_rate: float = 1.0e-3


@configclass
class HimPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    class_name: str = "go2_locomotion_lab.algorithms.him.ppo:HimPPO"
