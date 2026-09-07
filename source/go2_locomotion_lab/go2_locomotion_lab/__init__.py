"""Go2 Locomotion Lab.

Importing the package registers its Gymnasium tasks. Isaac Sim must be launched
before constructing an environment, but configuration modules remain importable
for tooling and contract tests.
"""

from .tasks import *  # noqa: F401, F403

__version__ = "0.1.0"

