# Go2 Locomotion Lab

Independent Unitree Go2 locomotion training platform built on Isaac Lab and
RSL-RL. It contains separate Competition PPO and HIM locomotion routes, both with
training, checkpoint playback, export and unified evaluation. HIM additionally has
MuJoCo Sim2Sim and a safety-gated Unitree SDK2 deployment adapter.

Competition and HIM are parallel routes. They share the robot, terrain, command,
control, and evaluation contracts; they do not share an ActorCritic implementation.
HIM owns a separate estimator, actor and PPO extension. This repository contains no
KaiWu workflow wrappers and no navigation stack.

## Verified server baseline

| Component | Version / status |
|---|---|
| GPU | 2 x NVIDIA GeForce RTX 4090, 24 GiB each |
| NVIDIA driver | 550.144.03 |
| Driver CUDA capability | 12.4 |
| CUDA toolkit (`nvcc`) | 12.4 / V12.4.99 |
| Conda | 23.7.4 |
| Project Python | 3.10.20 |
| PyTorch | 2.7.0+cu128 |
| Isaac Sim | 4.5.0-rc.36 |
| Isaac Lab | 0.54.3, clean `/home/fenglab/IsaacLab` checkout |
| RSL-RL | 5.0.1 |
| Git | 2.25.1 |

The project environment is isolated at
`/home/fenglab/anaconda3/envs/go2_locomotion_lab`. It was cloned from the known
working `islab` environment, then its editable Isaac Lab packages were pointed at
the clean checkout. No system CUDA, driver, existing environment, or Isaac Lab
source was modified.

## Layout

```text
go2_locomotion_lab/
├── source/go2_locomotion_lab/
│   ├── config/extension.toml
│   └── go2_locomotion_lab/
│       ├── assets/                  # official Go2 + centralized terrain
│       ├── tasks/locomotion/        # shared environment and MDP terms
│       ├── algorithms/competition/  # RSL-RL PPO configuration
│       ├── algorithms/him/          # estimator, actor and PPO integration
│       ├── evaluation/              # route-independent metrics/reports
│       └── monitoring/              # shared TensorBoard metrics/tag contract
├── scripts/                         # train, play, evaluate, export
├── configs/                         # experiment inventory
├── tests/
├── docs/
├── requirements-deployment.txt      # optional ONNX/MuJoCo runtime
└── third_party/
```

## Setup

On this server the environment and extension are prepared with:

```bash
conda activate go2_locomotion_lab
cd "/home/fenglab/lmy/RL/Go2 Locomotion Platform/go2_locomotion_lab"
python -m pip install --no-deps -e source/go2_locomotion_lab
python -m pip install -r requirements-monitoring.txt
```

This server uses a standalone Isaac Sim installation. In each new shell, inject
its official Python and shared-library paths before running train/play/evaluate:

```bash
source /home/fenglab/IsaacSim/setup_conda_env.sh
```

TensorBoard itself does not require Isaac Sim to be running.

## Validation

```bash
python -m pytest -q tests/test_static_contracts.py
python tests/smoke_official.py --headless --device cuda:0
python tests/smoke_env.py --headless --device cuda:0 --num_envs 128
```

The Phase A acceptance run completed official Go2 reset/step, project task registration,
128/1024/4096-env finite stepping, 128-env PPO for two iterations with final checkpoint save, checkpoint
playback, JSON/CSV evaluation, and TorchScript/ONNX export. The checkpoint and smoke
artifacts are under ignored `logs/` and `outputs/` directories.

For another machine, first install the Isaac Sim/Isaac Lab versions supported by
that machine in a fresh Conda environment. Do not install from `environment.yml`
alone: Isaac Sim binaries and CUDA-compatible PyTorch must come from the selected
Isaac Lab installation workflow.

## Train and resume

```bash
python scripts/train.py --headless --num_envs 4096 --seed 42 --max_iterations 1500

python scripts/train.py --headless --resume \
  --load_run 2026-09-07_12-00-00_baseline \
  --load_checkpoint model_500.pt
```

Use `--device cuda:0` to select a device. Fixed commands are accepted by training,
play, and evaluation:

```bash
python scripts/train.py --headless --vx 0.4 --vy 0.0 --wz 0.0
python scripts/play.py --checkpoint PATH/model_1500.pt --vx 0.4 --vy 0.0 --wz 0.0
python scripts/evaluate.py --headless --checkpoint PATH/model_1500.pt \
  --num_envs 128 --vx 0.4 --vy 0.0 --wz 0.0
```

Training uses RSL-RL's TensorBoard writer and records PPO/HIM losses, velocity
tracking, stability, control, contact, reward components, and termination rates.
Start the shared comparison server with:

```bash
python scripts/launch_tensorboard.py
```

Forward remote port `6006` in VS Code and open `http://127.0.0.1:6006` on the
local Windows machine. See [TensorBoard monitoring](docs/TENSORBOARD.md) for the
complete tag contract, SSH forwarding alternatives, and event-file validation.

Evaluation writes `outputs/evaluation/evaluation.json` and `.csv`. Export uses:

```bash
python scripts/export.py --headless --checkpoint PATH/model_1500.pt
```

## HIM train to deployment

```bash
python scripts/train.py --task Go2-Locomotion-HIM-Baseline-v0 \
  --headless --num_envs 4096 --max_iterations 1500
python scripts/export.py --task Go2-Locomotion-HIM-Baseline-Play-v0 \
  --headless --checkpoint PATH/model_1500.pt --output_dir outputs/him_policy
python scripts/sim2sim.py --policy-dir outputs/him_policy \
  --model PATH/unitree_mujoco/unitree_robots/go2/scene.xml --headless
python scripts/deploy_go2.py --policy-dir outputs/him_policy --dry-run
```

See [HIM baseline](docs/HIM_BASELINE.md) and the mandatory
[deployment checklist](docs/DEPLOYMENT.md) before connecting a robot.

## Smoke tests

```bash
python -m pytest -q tests/test_static_contracts.py
python tests/smoke_env.py --headless --num_envs 128 --num_steps 8
python scripts/train.py --headless --num_envs 128 --max_iterations 2 \
  --run_name acceptance_smoke
```

See [architecture](docs/ARCHITECTURE.md),
[baseline contract](docs/COMPETITION_BASELINE.md), [HIM plan](docs/HIM_PLAN.md),
[TensorBoard monitoring](docs/TENSORBOARD.md), and [roadmap](docs/ROADMAP.md).
