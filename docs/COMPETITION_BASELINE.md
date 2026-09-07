# Competition PPO Baseline

`Competition-PPO-Baseline` is a manager-based Isaac Lab velocity locomotion task
using the official Unitree Go2 USD and actuator configuration and RSL-RL PPO.

## Contracts

Actor observation `proprio45`, in this exact order:

| Term | Size |
|---|---:|
| base angular velocity | 3 |
| projected gravity | 3 |
| velocity command `(vx, vy, wz)` | 3 |
| relative joint position | 12 |
| relative joint velocity | 12 |
| previous action | 12 |
| Total | 45 |

The critic receives the same proprioception plus the native 187-value Isaac Lab height scan (232 total on this
Isaac Lab version).
The scanner is not reshaped or padded to 256. Actor terrain input/latent alignment is
a later experiment and cannot change `proprio45` silently.

Actions are 12 normalized policy values in the documented `FR, FL, RR, RL` order.
They are scaled by the official Go2 rough-task value `0.25`, offset by the official
default pose, and sent as joint position targets to the official PD/DC motor config.
Physics runs at 200 Hz and policy control at 50 Hz.

The baseline reads these values from Isaac Lab's official `UNITREE_GO2_CFG`:

| Parameter | Official value |
|---|---:|
| stiffness | 25.0 Nm/rad |
| damping | 0.5 Nms/rad |
| effort limit / saturation effort | 23.5 Nm |
| actuator velocity limit | 30.0 rad/s |
| soft joint position limit factor | 0.9 |

Official default joint-position patterns are `left hip=+0.1`, `right hip=-0.1`,
`front thigh=0.8`, `rear thigh=1.0`, and `calf=-1.5` radians. Competition-specific
values are not guessed: future authoritative overrides belong only in
`assets/go2.py` under `COMPETITION_TODO`.

Reward implementations live in `tasks/locomotion/mdp/rewards.py`; their weights live
only in `base_env_cfg.py`. No high-step-specific reward is active.

Evaluation `episode_length` is reported in seconds. If an evaluation horizon contains
no completed episode, both `episode_length` and `fall_rate` are `null` rather than
silently reporting a successful zero-fall trial.

## Reproducible commands

```bash
conda activate go2_locomotion_lab
cd go2_locomotion_lab
python scripts/train.py --task Go2-Locomotion-Competition-Baseline-v0 \
  --headless --num_envs 4096 --seed 42 --max_iterations 1500

python scripts/train.py --headless --resume --load_run RUN_DIRECTORY \
  --load_checkpoint model_500.pt

python scripts/play.py --checkpoint CHECKPOINT --vx 0.4 --vy 0.0 --wz 0.0
python scripts/evaluate.py --headless --checkpoint CHECKPOINT \
  --num_envs 128 --vx 0.4 --vy 0.0 --wz 0.0
```
