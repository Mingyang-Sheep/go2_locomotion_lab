"""Installation entry point for the Go2 Locomotion Lab Isaac Lab extension."""

from pathlib import Path

import toml
from setuptools import find_packages, setup

EXTENSION_ROOT = Path(__file__).parent
metadata = toml.load(EXTENSION_ROOT / "config" / "extension.toml")["package"]

setup(
    name="go2_locomotion_lab",
    version=metadata["version"],
    description=metadata["description"],
    author=metadata["author"],
    license="BSD-3-Clause",
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.10",
    install_requires=["gymnasium", "numpy", "toml"],
    extras_require={
        "deployment": ["onnxruntime>=1.20,<2"],
        "monitoring": ["tensorboard>=2.18,<3"],
        "sim2sim": ["mujoco>=3.2,<4", "onnxruntime>=1.20,<2"],
    },
    zip_safe=False,
)
