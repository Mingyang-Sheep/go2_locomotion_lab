# Architecture

Competition and HIM are independent, parallel routes. HIM is not a later stage of
Competition, and its estimator or actor must never be inserted into Competition PPO.

```text
Competition PPO                         HIM
  PPO Loco Teacher                       HIM estimator + actor (future)
  High-Step Loco (future)                HIM Sim2Real (future)
  LBC (future)                              |
          \                               /
           shared Go2 environment contract
             robot | terrain | command
             action | reward | termination
                        |
               unified evaluation
```

## Ownership

- `assets/`: official Go2 articulation reference, joint order, and terrain parameters.
- `tasks/locomotion/`: algorithm-neutral physics and MDP implementations.
- `algorithms/competition/`: RSL-RL PPO configuration only.
- `algorithms/him/`: reserved boundary; no implementation in Phase A.
- `evaluation/`: metrics and report serialization shared by every future route.
- `scripts/`: lifecycle entry points which launch Isaac Sim before importing task configs.

The project imports Isaac Lab as a dependency and does not modify its checkout or
site-packages source. Gym registration points to this extension's configuration.

