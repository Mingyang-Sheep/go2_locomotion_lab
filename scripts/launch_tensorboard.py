#!/usr/bin/env python3
"""Launch TensorBoard for all Go2 locomotion experiments."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from common import PROJECT_ROOT

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--logdir", type=Path, default=PROJECT_ROOT / "logs" / "rsl_rl")
parser.add_argument("--host", default="127.0.0.1", help="Bind host; keep loopback when using SSH forwarding.")
parser.add_argument("--port", type=int, default=6006)
parser.add_argument("--reload_interval", type=float, default=5.0)
args = parser.parse_args()


def main() -> int:
    logdir = args.logdir.expanduser().resolve()
    logdir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "tensorboard.main",
        "--logdir",
        str(logdir),
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--reload_interval",
        str(args.reload_interval),
    ]
    print(f"[INFO] TensorBoard log root: {logdir}", flush=True)
    print(f"[INFO] Server URL: http://{args.host}:{args.port}", flush=True)
    try:
        return subprocess.call(command)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
