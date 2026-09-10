#!/usr/bin/env python3
"""Run an exported HIM policy against Unitree's Go2 MuJoCo model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from go2_locomotion_lab.deployment import MujocoRuntime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True, help="Path to unitree_mujoco/unitree_robots/go2/scene.xml")
    parser.add_argument("--vx", type=float, default=0.4)
    parser.add_argument("--vy", type=float, default=0.0)
    parser.add_argument("--wz", type=float, default=0.0)
    parser.add_argument("--steps", type=int, default=1000, help="Number of 50 Hz policy steps")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--viewer",
        choices=("mujoco", "mjviser"),
        default="mujoco",
        help="Interactive viewer when --headless is omitted (mjviser requires the deployment extra).",
    )
    parser.add_argument("--host", default="127.0.0.1", help="mjviser bind host; use loopback with SSH forwarding")
    parser.add_argument("--port", type=int, default=8080, help="mjviser/Viser HTTP port")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def add_command_controls(server, runtime: MujocoRuntime) -> None:
    """Expose the trained command range as live Viser controls."""
    import numpy as np

    command = runtime.get_command()
    ranges = runtime.config.command_ranges
    with server.gui.add_folder("Go2 Command"):
        sliders = []
        for label, index in (("vx (m/s)", 0), ("vy (m/s)", 1), ("wz (rad/s)", 2)):
            slider = server.gui.add_slider(
                label,
                min=float(ranges[index, 0]),
                max=float(ranges[index, 1]),
                step=0.01,
                initial_value=float(command[index]),
            )
            sliders.append(slider)

        def update(_) -> None:
            runtime.set_command(np.array([slider.value for slider in sliders], dtype=np.float32))

        for slider in sliders:
            slider.on_update(update)


def run_mjviser(runtime: MujocoRuntime, args: argparse.Namespace) -> None:
    try:
        import viser
        from mjviser import Viewer
    except ImportError as exc:
        raise RuntimeError("Install mjviser with `python -m pip install -r requirements-mjviser.txt`") from exc

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


def run_native_viewer(runtime: MujocoRuntime, steps: int) -> None:
    import mujoco.viewer

    with mujoco.viewer.launch_passive(runtime.model, runtime.data) as viewer:
        for _ in range(steps):
            if not viewer.is_running():
                break
            runtime.step_control()
            viewer.sync()


def main() -> None:
    args = parse_args()
    runtime = MujocoRuntime.from_paths(args.policy_dir, args.model)
    runtime.set_command((args.vx, args.vy, args.wz))

    if args.headless:
        result = runtime.run_headless(args.steps)
        print(json.dumps(result, indent=2))
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        return

    if args.viewer == "mjviser":
        run_mjviser(runtime, args)
    else:
        run_native_viewer(runtime, args.steps)


if __name__ == "__main__":
    main()
