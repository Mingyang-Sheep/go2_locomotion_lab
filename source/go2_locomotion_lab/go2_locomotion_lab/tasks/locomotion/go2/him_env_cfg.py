"""HIM environment interface placeholder.

HIM will replace the observation design and actor stack while reusing the Go2
physics, action, command, terrain, reward, and evaluation contracts.
"""

from isaaclab.utils import configclass

from .base_env_cfg import BaseGo2LocomotionEnvCfg


@configclass
class HimBaselineEnvCfg(BaseGo2LocomotionEnvCfg):
    IMPLEMENTED = False

