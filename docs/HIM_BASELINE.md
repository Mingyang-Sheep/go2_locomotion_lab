# HIM Locomotion Baseline

The HIM route is registered as:

```text
Go2-Locomotion-HIM-Baseline-v0
Go2-Locomotion-HIM-Baseline-Play-v0
```

Train and resume with the same route-neutral scripts used by Competition:

```bash
python scripts/train.py --task Go2-Locomotion-HIM-Baseline-v0 \
  --headless --device cuda:0 --num_envs 4096 --max_iterations 1500

python scripts/train.py --task Go2-Locomotion-HIM-Baseline-v0 \
  --headless --resume --checkpoint PATH/model_500.pt --max_iterations 1000
```

The actor receives six chronological `proprio45` frames. Isaac Lab initializes a
new episode's history with the first frame, matching `ObservationHistory.reset()` in
the standalone runtime. The critic layout is `proprio45 + base_lin_vel3 +
height_scan187`.

The estimator encoder maps 270 values to velocity3 and latent16. Its target branch
maps the next privileged, noise-free proprio45 to latent16. Both branches learn through
next-state velocity MSE plus balanced swapped prototype assignments. Terminal
transitions are excluded because Isaac Lab has already reset their next observation. Estimator outputs are
detached from PPO, so Competition and HIM optimization remain separate.

Export creates an end-to-end policy. The estimator is embedded in the graph, so
deployment needs only one history tensor and receives one action12 tensor:

```bash
python scripts/export.py --task Go2-Locomotion-HIM-Baseline-Play-v0 \
  --headless --checkpoint PATH/model_1500.pt --output_dir outputs/him_policy
```

Files are `policy.pt`, `policy.onnx`, and `deploy.json`.
