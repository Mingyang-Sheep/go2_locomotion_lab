"""MDP terms shared by Competition and HIM environments."""

# Reuse Isaac Lab manager term configurations and standard MDP primitives.
from isaaclab.envs.mdp import *  # noqa: F401, F403

from .curriculum import *  # noqa: F401, F403
from .events import *  # noqa: F401, F403
from .observations import *  # noqa: F401, F403
from .rewards import *  # noqa: F401, F403
from .terminations import *  # noqa: F401, F403
