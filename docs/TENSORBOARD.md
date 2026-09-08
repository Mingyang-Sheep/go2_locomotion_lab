# TensorBoard Training Monitoring

Competition PPO and HIM are logged through the same RSL-RL TensorBoard writer and
the same physics-metric wrapper. Their policies and algorithms remain independent.
Every run is stored below:

```text
logs/rsl_rl/
├── competition_ppo_baseline/<timestamp>_<run_name>/
└── him_ppo_baseline/<timestamp>_<run_name>/
```

Point TensorBoard at `logs/rsl_rl`, not at a single run, to compare both routes and
multiple seeds. Give runs descriptive names when training:

```bash
conda activate go2_locomotion_lab
source /home/fenglab/IsaacSim/setup_conda_env.sh
python scripts/train.py --headless --run_name flat_seed42 --seed 42
python scripts/train.py --task Go2-Locomotion-HIM-Baseline-v0 \
  --headless --run_name rough_seed42 --seed 42
```

## Start the server

Run this in a VS Code Remote SSH terminal from the repository root:

```bash
conda activate go2_locomotion_lab
cd "/home/fenglab/lmy/RL/Go2 Locomotion Platform/go2_locomotion_lab"
python scripts/launch_tensorboard.py --host 127.0.0.1 --port 6006
```

On a fresh environment, install the standalone monitoring dependencies once with
`python -m pip install -r requirements-monitoring.txt`. TensorBoard does not need
the Isaac Sim setup script or an active GPU simulation process.

The server binds to `127.0.0.1:6006` by default and reloads event files every five
seconds. Binding only to loopback keeps it unavailable on the server's public
network interfaces. Alternatives are explicit:

```bash
python scripts/launch_tensorboard.py --port 16006 --reload_interval 10
python scripts/launch_tensorboard.py --logdir logs/rsl_rl/competition_ppo_baseline
```

## VS Code Remote SSH forwarding

1. Connect to the server with the VS Code **Remote - SSH** extension.
2. Open the **Ports** view (`View` -> `Terminal`, then select the **Ports** tab).
3. Select **Forward a Port**, enter server port `6006`, and keep visibility private.
4. Open `http://127.0.0.1:6006` in the local Windows browser.

If the local port is already occupied, edit the forwarded port in the Ports view,
for example mapping remote `6006` to local `16006`, then open
`http://127.0.0.1:16006`.

The equivalent PowerShell tunnel, used instead of the VS Code Ports view, is:

```powershell
ssh -N -L 6006:127.0.0.1:6006 USER@SERVER
```

Keep that PowerShell window open while viewing TensorBoard. Do not combine a VS
Code forward and the PowerShell tunnel on the same local port.

## Scalar contract

RSL-RL writes these groups without a second logging service:

| Group | Representative tags | Meaning |
|---|---|---|
| PPO | `Loss/value`, `Loss/surrogate`, `Loss/entropy`, `Loss/learning_rate` | optimization |
| policy | `Policy/mean_std` | exploration scale |
| training | `Train/mean_reward`, `Train/mean_episode_length` | episode outcome |
| HIM estimator | `Loss/him_velocity`, `Loss/him_swap` | estimator regression and embedding |
| velocity | `Metrics/velocity/lin_xy_error`, `Metrics/velocity/yaw_error` | command tracking error |
| stability | `Metrics/stability/*` | base height, tilt, roll/pitch rate, vertical velocity |
| control | `Metrics/control/*` | action change and absolute joint torque |
| contact | `Metrics/contact/*` | foot sliding and stumbling |
| rewards | `Episode_Reward/<term>` | every configured weighted reward component |
| termination | `Metrics/termination/*`, `Episode_Termination/<term>` | cumulative rates and causes |
| performance | `Perf/*` | FPS and rollout/update time |

`Metrics/termination/fall_rate` is the cumulative fraction of completed episodes
that ended through a non-timeout termination. `timeout_rate` is its timeout
counterpart. A fall coincident with a timeout is counted as a timeout. This makes
the curves meaningful even when a rollout step ends no episodes.

The base height is measured relative to the finite height-scanner hits beneath the
robot. `torque_mean` is the mean absolute joint torque and `torque_peak` is the
per-environment maximum absolute joint torque; RSL-RL then averages step samples
within each logged iteration.

The native Isaac Lab command metrics and termination tags remain available for
backward compatibility. The `Metrics/...` tags are the stable cross-route names to
use in dashboards.

## Compare and validate runs

In TensorBoard, enable multiple runs in the left run selector. Use a regular
expression such as `competition_ppo_baseline.*seed42` or
`him_ppo_baseline.*seed42` to compare like-for-like commands, terrains and seeds.
TensorBoard reads new event data during training; no restart is required.

After a smoke run, validate that all common tags exist:

```bash
python scripts/validate_tensorboard.py \
  logs/rsl_rl/competition_ppo_baseline/<RUN_DIRECTORY> \
  --route competition

python scripts/validate_tensorboard.py \
  logs/rsl_rl/him_ppo_baseline/<RUN_DIRECTORY> \
  --route him
```

The validator checks every Competition/HIM baseline reward component and both
baseline termination causes in addition to PPO and shared locomotion metrics.
Future experiment-specific terms may add tags without invalidating this baseline
contract.

## Troubleshooting

- No runs: verify training printed a path under `logs/rsl_rl` and that TensorBoard
  was launched from this repository or received an absolute `--logdir`.
- Browser cannot connect: confirm TensorBoard is still running and VS Code shows
  remote `6006` as forwarded.
- Curves update slowly: the RSL-RL writer flushes every 10 seconds; the server
  reload interval defaults to 5 seconds.
- Runs are hard to distinguish: set a unique `--run_name` and `--seed`; both agent
  and environment configs are saved inside each run's `params/` directory.
