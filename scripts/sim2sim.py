#!/usr/bin/env python3
"""Run an exported HIM policy against Unitree's Go2 MuJoCo model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from go2_locomotion_lab.deployment import DeploymentConfig, ObservationHistory, OnnxPolicy, build_proprio


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True, help="Path to unitree_mujoco/unitree_robots/go2/scene.xml")
    parser.add_argument("--vx", type=float, default=0.4)
    parser.add_argument("--vy", type=float, default=0.0)
    parser.add_argument("--wz", type=float, default=0.0)
    parser.add_argument("--steps", type=int, default=1000, help="Number of 50 Hz policy steps")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def sensor(model, data, mujoco, name: str) -> np.ndarray:
    sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
    if sensor_id < 0:
        raise KeyError(f"MuJoCo sensor not found: {name}")
    start = model.sensor_adr[sensor_id]
    size = model.sensor_dim[sensor_id]
    return np.asarray(data.sensordata[start : start + size]).copy()


def main() -> None:
    args = parse_args()
    try:
        import mujoco
    except ImportError as exc:
        raise RuntimeError("Install the optional 'mujoco' package for Sim2Sim") from exc

    config = DeploymentConfig.load(args.policy_dir / "deploy.json")
    policy = OnnxPolicy(args.policy_dir / "policy.onnx")
    model = mujoco.MjModel.from_xml_path(str(args.model.resolve()))
    data = mujoco.MjData(model)
    if model.nkey:
        mujoco.mj_resetDataKeyframe(model, data, 0)

    qpos_addresses = []
    qvel_addresses = []
    actuator_ids = []
    for name in config.joint_names:
        xml_name = f"{name}_joint"
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, xml_name)
        actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if joint_id < 0 or actuator_id < 0:
            raise KeyError(f"Go2 joint/actuator not found in MuJoCo model: {name}")
        qpos_addresses.append(model.jnt_qposadr[joint_id])
        qvel_addresses.append(model.jnt_dofadr[joint_id])
        actuator_ids.append(actuator_id)
    data.qpos[qpos_addresses] = config.default_joint_pos
    mujoco.mj_forward(model, data)

    sim_steps_per_control = round(config.control_dt / model.opt.timestep)
    if not np.isclose(sim_steps_per_control * model.opt.timestep, config.control_dt):
        raise ValueError("MuJoCo timestep must divide the exported 50 Hz control period")
    command = config.validate_command((args.vx, args.vy, args.wz))
    previous_action = np.zeros(12, dtype=np.float32)
    history = ObservationHistory(config.history_steps)
    max_torque = 0.0

    viewer_context = None
    if not args.headless:
        import mujoco.viewer

        viewer_context = mujoco.viewer.launch_passive(model, data)
    try:
        for _ in range(args.steps):
            joint_pos = np.asarray(data.qpos[qpos_addresses], dtype=np.float32)
            joint_vel = np.asarray(data.qvel[qvel_addresses], dtype=np.float32)
            frame = build_proprio(
                sensor(model, data, mujoco, "imu_gyro"),
                sensor(model, data, mujoco, "imu_quat"),
                command,
                joint_pos,
                joint_vel,
                previous_action,
                config.default_joint_pos,
            )
            action = policy(history.append(frame))[0]
            target = config.action_to_joint_target(action)
            for _ in range(sim_steps_per_control):
                torque = (
                    config.stiffness * (target - data.qpos[qpos_addresses]) - config.damping * data.qvel[qvel_addresses]
                )
                torque = np.clip(torque, -config.effort_limits, config.effort_limits)
                data.ctrl[actuator_ids] = torque
                max_torque = max(max_torque, float(np.abs(torque).max()))
                mujoco.mj_step(model, data)
            previous_action = action
            if viewer_context is not None:
                viewer_context.sync()
    finally:
        if viewer_context is not None:
            viewer_context.close()

    result = {
        "status": "SIM2SIM_OK",
        "policy_steps": args.steps,
        "sim_time": float(data.time),
        "base_position": np.asarray(data.qpos[:3]).tolist(),
        "max_abs_torque": max_torque,
    }
    print(json.dumps(result, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
