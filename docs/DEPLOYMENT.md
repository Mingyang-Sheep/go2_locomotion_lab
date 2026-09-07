# HIM Sim2Sim And Deployment

Install optional runtime dependencies only in the isolated project environment:

```bash
conda activate go2_locomotion_lab
python -m pip install -r requirements-deployment.txt
```

## Sim2Sim

Use Unitree's unmodified Go2 MuJoCo model:

```bash
git clone https://github.com/unitreerobotics/unitree_mujoco.git third_party/unitree_mujoco
python scripts/sim2sim.py \
  --policy-dir outputs/him_policy \
  --model third_party/unitree_mujoco/unitree_robots/go2/scene.xml \
  --vx 0.4 --vy 0 --wz 0
```

Add `--headless --steps 1000 --output outputs/sim2sim.json` for a reproducible batch
run. The adapter resolves joints, actuators, IMU and gyro by name. It evaluates the
policy at 50 Hz and runs the MuJoCo PD loop at the model physics timestep.

## Runtime Check

Run ONNX inference and the complete observation/action conversion without SDK2:

```bash
python scripts/deploy_go2.py --policy-dir outputs/him_policy \
  --dry-run --duration 2 --vx 0.4
```

## Real Go2

Real output requires Unitree's `unitree_sdk2_python`. Place the robot on a support
rig with an emergency stop available, verify joint order and signs in dry-run and
MuJoCo, disable the sport controller, and begin with zero velocity command. The
runtime refuses to publish unless `--arm` is supplied, the sport motion service is
already disabled, a fresh low-state stream is present, the start pose is close to the
exported default pose, commands are within training ranges, and tilt remains below the
configured threshold.

```bash
python scripts/deploy_go2.py --policy-dir outputs/him_policy \
  --network eth0 --arm --duration 10 --vx 0 --vy 0 --wz 0
```

The real-robot adapter is code-complete but cannot be hardware-validated on this
server. Treat first hardware use as commissioning, not as a normal training run.
