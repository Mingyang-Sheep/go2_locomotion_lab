"""Go2 Locomotion Lab with an Isaac-independent deployment runtime."""

import importlib.util

# Isaac Sim exposes pxr only after AppLauncher starts. Keeping this conditional
# lets the exported ONNX runtime run on a lightweight MuJoCo or robot computer.
if importlib.util.find_spec("pxr") is not None:
    from .tasks import *  # noqa: F401, F403

__version__ = "0.1.0"
