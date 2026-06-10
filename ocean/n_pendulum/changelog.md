# n_pendulum Changelog

This file tracks the meaningful `n_pendulum` training, reward, architecture,
and performance changes, plus their observed impact.

## Current Active Setup

The active config is a fixed 5-link run based on the successful 5-pendulum
hyperparameters, with a smaller model and stabilized PPO settings.

```ini
[vec]
total_agents = 4096
num_buffers = 2
num_threads = 16

[env]
num_links = 5
cart_mass = 1.0
link_mass = 0.1
link_length = 0.5
gravity = 9.8
force_mag = 8.622977087274194
dt = 0.02
min_episode_steps = 1000
max_episode_steps = 2000
upright_reset_prob = 0.38920166390016675
upright_angle_noise = 0.034932828396558764
upright_vel_noise = 0.02436711547896266

[policy]
hidden_size = 144
num_layers = 4
expansion_factor = 1

[torch]
network = MinGRU
encoder = DefaultEncoder
decoder = DefaultDecoder

[train]
total_timesteps = 100000000000
learning_rate = 0.0009
gamma = 0.995
gae_lambda = 0.95
clip_coef = 0.15
vf_coef = 0.5
vf_clip_coef = 1.0
max_grad_norm = 0.5
ent_coef = 0.0012
minibatch_size = 16384
horizon = 128
```

Current model size with `DP_MAX_LINKS = 7`: approximately `257k` params.

Active reward is the non-delta shaped reward path from the successful
5-pendulum work:

```text
upright link shaping
+ gated right-angle swing-up shaping
+ slow-above/stillness shaping
+ cart centering
+ tip centering
+ balance and hold bonuses
```

The later `delta_height` term is not active.

## 2026-06-10

### Removed Link-Count Curriculum

Removed the active-link curriculum code and config surface:

```text
curriculum_steps
link_curriculum_min_links
link_curriculum_final_random_prob
max_link_hold_time
max_link_n
upright_reset_prob_start
upright_reset_prob_end
```

Impact:

- Restored fixed-link training behavior.
- Removed ambiguity between `hold_time` and `max_link_hold_time`.
- Sweep metric is back to plain `hold_time`.
- Current runs use exactly `num_links` links every episode.

### Reduced Observation Padding

Changed the compile-time max link cap:

```c
#define DP_MAX_LINKS 7
```

Impact:

- Observation size dropped from `386` floats to `44` floats.
- `128 x 2` MinGRU dropped from roughly `149k` params to roughly `106k`.
- Current `144 x 4` MinGRU is roughly `257k` params.
- Old checkpoints from the 64-link padded observation shape are not compatible.
- Physics target remains 5 links in the active config, with room to run up to 7.

### Optimized Env Hot Path

Changed `compute_observations` and `integrate_physics` to avoid unnecessary
full-buffer clears.

Impact:

- Removed full padded observation clearing when only the active obs tail needs
  zeroing.
- Removed full dense matrix zero-init now that the active `dof x dof` block is
  assigned directly.
- Helped reduce CPU env overhead after rebuilding the native backend.

### Tuned Rollout Threading

Tested several rollout worker splits on an R9 7900X:

```text
2 buffers / 16 threads: best observed balance so far
2 buffers / 24 threads: worse, likely SMT contention
2 buffers / 12 threads: worse, underfed env workers
1 buffer / 8 threads: less overlap
```

Current setting:

```ini
num_buffers = 2
num_threads = 16
```

Impact:

- Gives `2048` agents per buffer.
- Gives `8` CPU workers per buffer.
- Best observed env timing was around `75ms`, down from roughly `160ms`.

### Stabilized PPO After Collapse

A `200 x 4` MinGRU run reached strong early hold behavior but collapsed as KL
rose.

Observed before collapse:

```text
params: 491.2K
steps: 2.9B
score: 382.289
hold_time: 70.578
kl: 0.072
clipfrac: 0.119
```

Then KL continued rising:

```text
steps: 3.3B
score: 371.284
hold_time: 73.240
kl: 0.099
x_threshold_termination: 0.062
```

Changes:

```ini
learning_rate = 0.0009
clip_coef = 0.15
ent_coef = 0.0012
```

Impact:

- Intended to reduce policy drift and clip pressure.
- Keeps exploration slightly higher.
- Leaves the successful 5-link env settings intact.

### Network Size Tests

Useful tested shapes:

```text
272 x 4: documented successful 5-link model, about 995K dashboard params
200 x 4: about 491K with DP_MAX_LINKS=7, strong early hold, collapsed from high KL
144 x 4: about 257K with DP_MAX_LINKS=7, current active test
128 x 4: about 204K with DP_MAX_LINKS=7
128 x 2: about 106K with DP_MAX_LINKS=7
64 x 12: about 173K before latest config changes, decent but deep/narrow
50 x 4: about 50K, likely too small
```

Impact:

- Wider 4-layer MinGRU shapes look more promising than very deep narrow shapes.
- `200 x 4` learned quickly but needed PPO stabilization.
- `144 x 4` is the current compromise between speed and capacity.

### Hybrid Delta Reward Test

Pure delta-height reward performed weakly. We switched back to a hybrid reward:

```text
weighted height delta
+ upright link shaping
+ gated right-angle swing-up shaping
+ slow-above/stillness shaping
+ cart centering
+ tip centering
+ balance and hold bonuses
```

Best short-run hybrid sweep candidate:

```text
log: logs/n_pendulum/1781125830783.json
score: 112.389
perf: 0.571
hold_time: 4.911
steps: 174.6M
episode_length: 1500.101
x_threshold_termination: 0.000
max_steps_termination: 1.000
```

Impact:

- Hybrid reward beat pure delta reward in short 5-link sweeps.
- Pure delta reward gave very low score scale and weak progress.
- This is no longer active. The current reward removes the `delta_height` term
  and uses the non-delta shaped reward path.

## Successful 5-Pendulum Run

The first strong fixed 5-link run reached:

```text
num_links: 5
steps: 12.4B
score: 738.876
perf: 0.866
hold_time: 246.276
episode_length: 1408.470
x_threshold_termination: 0.126
max_steps_termination: 0.874
```

The run used the low-upright-start hold-time sweep winner, not the highest-score
sweep winner.

Successful long-run values:

```ini
[env]
num_links = 5
cart_mass = 1.0
link_mass = 0.1
link_length = 0.5
gravity = 9.8
force_mag = 8.622977087274194
dt = 0.02
min_episode_steps = 1000
max_episode_steps = 2000
upright_reset_prob = 0.38920166390016675
upright_angle_noise = 0.034932828396558764
upright_vel_noise = 0.02436711547896266

[policy]
hidden_size = 272
num_layers = 4
expansion_factor = 1

[torch]
network = MinGRU
encoder = DefaultEncoder
decoder = DefaultDecoder

[train]
total_timesteps = 100000000000
learning_rate = 0.0012467834141555744
gamma = 0.995
gae_lambda = 0.95
replay_ratio = 1
clip_coef = 0.2
vf_coef = 0.5
vf_clip_coef = 1.0
max_grad_norm = 0.5
ent_coef = 0.0010354715948189064
beta1 = 0.9
beta2 = 0.999
eps = 1e-8
minibatch_size = 16384
horizon = 128
prio_alpha = 0.5
prio_beta0 = 0.5
```

Source sweep candidate:

```text
log: logs/n_pendulum/1781111700197.json
score: 42.864
perf: 0.507
hold_time: 3.411
steps: 79.7M
episode_length: 1495.064
x_threshold_termination: 0.00137
max_steps_termination: 0.99863
```

Impact:

- Selecting by `hold_time` transferred better than selecting by shaped score.
- The breakthrough happened late. The run was around `hold_time = 33` at 2.4B
  steps, then reached `hold_time = 246` by 12.4B steps.
- `upright_reset_prob = 0.389...` was a useful fixed reset mix.
- `upright_reset_prob = 0.9` gave good short-run hold but transferred poorly to
  downward-start behavior.
- `upright_reset_prob = 0.0` could produce shaped score without actual hold.

## Sweep Template

Useful sweep ranges for this env:

```ini
[sweep]
method = Protein
metric = hold_time
metric_distribution = linear
goal = maximize
max_suggestion_cost = 450
max_runs = 16
gpus = 1
downsample = 5
use_gpu = True
prune_pareto = True
early_stop_quantile = 0.3
sweep_only = total_timesteps,upright_reset_prob,upright_angle_noise,upright_vel_noise,force_mag,learning_rate,ent_coef

[sweep.train.total_timesteps]
distribution = log_normal
min = 8e7
max = 3e8
scale = time

[sweep.env.upright_reset_prob]
distribution = uniform
min = 0.0
max = 0.5
scale = auto

[sweep.env.upright_angle_noise]
distribution = uniform
min = 0.025
max = 0.055
scale = auto

[sweep.env.upright_vel_noise]
distribution = uniform
min = 0.02
max = 0.055
scale = auto

[sweep.env.force_mag]
distribution = uniform
min = 8.5
max = 10.5
scale = auto

[sweep.train.learning_rate]
distribution = log_normal
min = 0.0005
max = 0.002
scale = 0.35

[sweep.train.ent_coef]
distribution = log_normal
min = 0.001
max = 0.0025
scale = auto
```
