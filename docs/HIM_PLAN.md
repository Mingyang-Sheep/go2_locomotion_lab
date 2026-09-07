# HIM Plan

HIM is a route parallel to Competition PPO. Phase A creates only this boundary and
`HimBaselineEnvCfg`; it does not train HIM and does not reuse the Competition actor.

Future data flow:

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

HIM will share robot, terrain, command, physical stepping, and evaluation contracts.
Its observation history, estimator loss, embedding, actor, and checkpoint schema will
remain under `algorithms/him/` and a HIM-specific environment specialization.

