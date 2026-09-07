# HIM Baseline

HIM is a route parallel to Competition PPO. It has its own observation history,
estimator, actor, PPO extension, checkpoint state and export path. It does not reuse
the Competition actor.

Implemented data flow:

```text
proprio history
      |
HIM estimator
      |
hybrid embedding
      |
HIM actor
      |
action12 -> shared Go2 joint-position/PD interface
```

The implemented actor input is six oldest-to-newest `proprio45` frames (270 values).
The estimator predicts body-frame linear velocity (3) and a normalized hybrid
embedding (16). The actor consumes the newest `proprio45 + velocity3 + latent16`.
The privileged critic consumes current proprioception, true body linear velocity,
and the native 187-ray height scan (235 values).

The estimator uses next-state velocity regression and swapped prototype assignment,
excluding terminal transitions whose next observation has already reset. PPO remains
the upstream RSL-RL implementation; `HimPPO` only adds estimator optimization and its
checkpoint state. This is an intentional adaptation to RSL-RL 5 rather than a copy of
the legacy custom runner/storage stack.

HIM shares robot, terrain, command, physical stepping, reward and evaluation contracts
with Competition. Sim2Sim and deployment use the exported `deploy.json`, generated
from the live Isaac Lab articulation and actuator rather than duplicated values.
