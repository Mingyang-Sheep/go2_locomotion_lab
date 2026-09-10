#!/usr/bin/env python3
"""Serve a live Go2 MuJoCo policy simulation through mjviser/Viser."""

from __future__ import annotations

import argparse
from pathlib import Path

from go2_locomotion_lab.deployment import MujocoRuntime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--vx", type=float, default=0.4)
    parser.add_argument("--vy", type=float, default=0.0)
    parser.add_argument("--wz", type=float, default=0.0)
    parser.add_argument("--host", default="127.0.0.1", help="Bind address; keep loopback for SSH forwarding")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--headless", action="store_true", help="Run a finite batch without importing mjviser")
    parser.add_argument("--steps", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runtime = MujocoRuntime.from_paths(args.policy_dir, args.model)
    runtime.set_command((args.vx, args.vy, args.wz))
    if args.headless:
        import json

        print(json.dumps(runtime.run_headless(args.steps), indent=2))
        return

    try:
        import viser
        from mjviser import Viewer
    except ImportError as exc:
        raise RuntimeError("Install mjviser with `python -m pip install -r requirements-mjviser.txt`") from exc
    from sim2sim import add_command_controls

    server = viser.ViserServer(host=args.host, port=args.port)
    viewer = Viewer(
        runtime.model,
        runtime.data,
        step_fn=runtime.step_physics,
        reset_fn=runtime.reset,
        server=server,
    )
    add_command_controls(server, runtime)
    print(f"[INFO] mjviser URL: http://{args.host}:{args.port}", flush=True)
    viewer.run()


if __name__ == "__main__":
    main()
