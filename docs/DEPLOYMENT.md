# HIM Sim2Sim And Deployment

Install optional headless runtime dependencies in the isolated project environment:

```bash
conda activate go2_locomotion_lab
python -m pip install -r requirements-deployment.txt
```

The Web Viewer has a separate dependency file because current `mjviser` releases
require Pillow `>=12.2`, while Isaac Lab 0.54.x pins Pillow `11.3.0`. Keep the
viewer in a lightweight environment so installing it cannot alter the Isaac Lab
training environment:

```bash
conda create -n go2_mjviser python=3.10 pip -y
conda activate go2_mjviser
cd "/home/fenglab/lmy/RL/Go2 Locomotion Platform/go2_locomotion_lab"
python -m pip install -r requirements-mjviser.txt
python -m pip install --no-deps -e source/go2_locomotion_lab
```

## Sim2Sim

Use Unitree's unmodified Go2 MuJoCo model:

```bash
git clone https://github.com/unitreerobotics/unitree_mujoco.git third_party/unitree_mujoco
python scripts/sim2sim.py \
  --policy-dir outputs/him_policy \
  --model third_party/unitree_mujoco/unitree_robots/go2/scene.xml \
  --vx 0.4 --vy 0 --wz 0 --headless \
  --steps 1000 --output outputs/sim2sim.json
```

The shared `MujocoRuntime` resolves joints, actuators, IMU and gyro by name. It
evaluates the unchanged ONNX policy at 50 Hz and runs the unchanged deployment
contract's PD loop at the model physics timestep. Headless execution does not import
the viewer stack and remains suitable for reproducible batch evaluation.

## mjviser Web Viewer

The live path is:

```text
Isaac Lab train -> policy.onnx -> MujocoRuntime -> MuJoCo -> mjviser
                -> VS Code Remote SSH port forwarding -> Windows browser
```

Start the server in a second VS Code Remote SSH terminal while training continues:

```bash
conda activate go2_mjviser
cd "/home/fenglab/lmy/RL/Go2 Locomotion Platform/go2_locomotion_lab"
python scripts/mjviser_viewer.py \
  --policy-dir outputs/him_policy \
  --model third_party/unitree_mujoco/unitree_robots/go2/scene.xml \
  --host 127.0.0.1 --port 8080 \
  --vx 0.4 --vy 0 --wz 0
```

In VS Code, open the **Ports** view, forward remote port `8080`, keep its visibility
private, and open the forwarded address (normally `http://127.0.0.1:8080`) in the
local Windows browser. If VS Code assigns another local port, open the exact address
shown in the Ports view. Keep the remote viewer terminal running.

The **Go2 Command** controls update `vx`, `vy`, and `wz` live within the exported
training ranges. mjviser's controls provide pause/play, single-step, reset, playback
speed, camera tracking, and visualization overlays. Reset also clears the HIM
observation history and previous action.

The same web front end is also available through the original entry point:

```bash
python scripts/sim2sim.py \
  --policy-dir outputs/him_policy \
  --model third_party/unitree_mujoco/unitree_robots/go2/scene.xml \
  --viewer mjviser --host 127.0.0.1 --port 8080
```

Use loopback plus SSH forwarding instead of exposing the viewer on a public server
interface. If local port `8080` is occupied, map the remote `8080` to another local
port such as `18080` and open `http://127.0.0.1:18080`.

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
