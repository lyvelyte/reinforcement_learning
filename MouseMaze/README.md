# MouseMaze

MouseMaze trains recurrent PPO agents on procedurally generated perfect mazes.
The default is a continuous recurrent PPO agent using a centered 5×5 `local`
observation, a 512-unit GRU, graph-aware progress shaping, 0.25-cell control
steps, a `-0.2` invalid-move penalty, and a small RND reward. Walls block line of
sight; `full` mode exposes the complete map. Local observations include an
egocentric crop with optional visit-count history.

Default full-map observations contain wall, mouse, goal, and visit-count
channels. Default local observations contain wall, goal, and visit-count
channels only. Discrete counts saturate at five visits by default. Continuous
counts record the occupied cell after every control step, including collisions,
and use `log1p(count) / log1p(max_episode_steps + 1)`. This remains informative
throughout the episode instead of saturating after 20 samples. The legacy
step-scaled clipped encoding remains available for schema-v10 inference and
ablations.

The remaining-time channel is disabled by default. Configure these inputs with
`--remaining-time-channel`, `--no-visit-count-channel`,
`--visit-count-clip COUNT`, `--visit-count-encoding clipped|episode_log`, and
`--no-wall-occlusion`; each Boolean option also
accepts its inverse form. Reward shaping is explicitly selectable with
`--distance-shaping-mode none|fractional|potential|progress` and is never silently
overridden by the observation mode.

The default `progress` reward uses a wall-aware route from the occupied cell to
the goal and rewards only actual reductions in that distance. A stationary
action therefore receives the step penalty instead of a positive discounted
potential reward. RND novelty is computed from the reached observation.
Use `--continuous-distance-mode start_path` only for legacy ablations; it can
underestimate off-route distance when walls separate a cell from the original
start-to-goal path.

Continuous recurrent PPO is the default. Use `--action-space discrete` for the
legacy four-direction policy. Continuous mode supplies normalized within-cell
row/column offsets and the previous collision flag as a compact three-value
proprioception vector directly to the GRU rather than as spatial image channels.
Its policy is a tanh-squashed two-dimensional Gaussian, so
sampled and deterministic actions stay in `[-1, 1]`. The environment scales
them to 0.25 cell per control step by default, checks the swept segment for
walls/corners, and supplies the executed displacement—not a rejected requested
action—to the GRU. Configure the scale with `--continuous-step-scale` and
continuous entropy independently with `--continuous-entropy-coef`.

## Environment

Use the repository's shared conda environment. Repository automation and checks
should select it explicitly:

```bash
conda run -n ml pytest MouseMaze/test_mouse_agent.py
```

From the repository root, start the recommended continuous training profile
with:

```bash
conda run -n ml python MouseMaze/MouseAgent.py \
  --train --no-resume --no-infer --remaining-time-channel
```

## RTX 3090 training

The `auto` profile selects `rtx3090-fast` on a 24GB RTX 3090. Recurrent PPO
automatically uses 768 environments with that profile and 256 with portable or
strict profiles; an explicit `--num-envs` always wins. The fast profile also
uses BF16/TF32, compiled recurrent kernels, fused Adam, GPU-resident rollouts,
vectorized exact continuous-maze geometry, and eight deterministic
maze-generation workers. A newly encountered
model/batch shape can spend roughly 1–2 minutes compiling; the compiled kernels
are cached and subsequent launches are much faster.

The RTX 3090 default was revalidated on 2026-08-26 with the 11×11 continuous
local configuration and eight 128-step rollouts at each candidate size. After
discarding three compile/warm-up rollouts, median collector-plus-learner
throughput over the remaining five was 6,532 steps/s at 512 environments,
7,816 steps/s at 768 (+19.7%), and 7,185 steps/s at 1,024. At 768 environments,
eight maze workers reached 7,816 steps/s versus 7,708 with 12 and 7,662 with 16;
generation blocking was already negligible with eight.

A CUDA-resident environment prototype matched observations, terminal flags,
collisions, and executed actions exactly, with a maximum reward difference of
`2.98e-8`. It reduced isolated 128-step environment time from 0.703 seconds to
0.115 seconds, but its projected end-to-end gain was only 4.7% with four PPO
epochs and approximately 16.6% before integration overhead with one epoch.
The production trainer therefore retains the simpler vectorized CPU environment.

Recurrent PPO defaults to finite fresh training. The 11×11 local budget is 100
million transitions and scales linearly with the final maximum dimension (136
million for 15×15 and 191 million for 21×21). Explicit `--max-env-steps`,
`--resume`, and `--target-only-stop` flags override those defaults. Target-only
candidates are still checked on three deterministic, seed-separated suites.

The resolved budget controls every recurrent schedule. RND defaults to `0.05`
and cosine-decays to zero by two-thirds progress. For the default 100-million
transition run, the learning rate decreases linearly from `3e-4` to `1.2e-4`
over the first 50 million transitions, then cosine-decays monotonically to
`3e-6`. Default continuous entropy stays at `0.003` until that precision phase
and then anneals to `0.0003`. Explicit target-only runs retain those floors.

Local episode limits use the maximum of 20 steps, four times shortest path, and
twice the traversable-cell count. Continuous mode scales the path and
exploration allowances by the inverse control-step scale. This provides a full
depth-first exploration allowance independently of oracle path length. There
is no default hard cap; `--max-episode-steps` adds one.

Hard-maze replay defaults to 5%. Timed-out training mazes and their unique
symmetries feed shape-specific bounded reservoirs; validation and benchmark
mazes are never added. Replay ramps from zero at a 60% validation solve rate to
its configured maximum at 80%.

Automatic curriculum is the default. It builds an odd size ladder in increments
of four, probes 2,048 deterministic mazes per size, and stages the lower third,
lower two-thirds, then unrestricted distribution using an eight-order
depth-first discovery score. Three consecutive 70% validations promote a stage.
Performance can promote sooner, but the default reserves the final 20% of a
finite budget for unrestricted mazes. Configure or disable that boundary with
`--curriculum-final-stage-fraction`. Use `--curriculum-mode manual` for the
legacy easy and medium distance ranges.

Maze generation uses Wilson's loop-erased random-walk algorithm. RTX workers
return 64 grids per future and retain four queued batches per worker while
preserving seeded submission order. Curriculum validation runs every 50,000
transitions; final-distribution evaluation changes to every 500,000 transitions.

Recurrent PPO uses a 512-unit GRU and 128-step truncated-BPTT sequences. The GRU
hidden state persists across the complete episode; sequence length controls the
gradient-history window, not inference memory lifetime. Minibatches contain 16
sequences, preserving a 2,048-transition effective minibatch. Learner telemetry
records that effective size and the optimizer updates performed by every rollout.
Target-KL stopping uses the mean
nonnegative approximate KL over a complete epoch, so every rollout receives at
least one full optimization epoch before later epochs can be skipped.

Rollback-based precision recovery is enabled during final-stage precision
training after 20 evaluations without improvement. Use
`--no-precision-recovery` to disable it; the existing
`--precision-plateau-evals`, `--precision-recovery-steps`, and
`--precision-recovery-lr-fraction` controls remain available for experiments.

Target-only runs continue evaluating by transition cadence after passing the
nominal episode count; that count no longer causes evaluation after every
learner update.

Full-map finite training:

```bash
conda run -n ml python MouseMaze/MouseAgent.py \
  --train --no-resume --no-infer --no-dashboard \
  --algorithm recurrent_ppo --observation-mode full \
  --performance-profile rtx3090-fast
```

Local 5×5 discrete finite training:

```bash
conda run -n ml python MouseMaze/MouseAgent.py \
  --train --no-resume --no-infer --no-dashboard \
  --algorithm recurrent_ppo --action-space discrete \
  --observation-mode local --view-size 5 \
  --performance-profile rtx3090-fast
```

Continuous ablation without distance shaping:

```bash
conda run -n ml python MouseMaze/MouseAgent.py \
  --train --no-resume --no-infer --no-dashboard \
  --algorithm recurrent_ppo --action-space continuous \
  --observation-mode local --view-size 5 \
  --distance-shaping-mode none --continuous-step-scale 0.25 \
  --performance-profile rtx3090-fast
```

Continuous mode is rejected for DQN and feed-forward PPO because those agents
do not implement a continuous action head.

Use `--target-only-stop` to opt out of the finite transition cap. DQN and
feed-forward PPO retain their existing bounded behavior.

Fresh training uses one UTC timestamp for paired artifacts under
`MouseMaze/results/models/` and `MouseMaze/results/logs/`. The base
`<timestamp>_mousemaze.pth` file is the frozen best policy used by inference and
benchmarks. Its `<timestamp>_mousemaze.latest.pth` sidecar contains the current
resumable policy, optimizer, RND, RNG, curriculum, counters, and sampler state.
The sidecar is written atomically every one million transitions by default and
on graceful interruption; configure the cadence with
`--latest-checkpoint-every-steps`.

Explicit resume prefers the sidecar and falls back to the frozen-best checkpoint.
Process speeds use only counters accumulated since the resume, while lifetime
totals remain in their own fields. Post-budget
target-only ETA is reported as unknown. Each JSONL record also contains UTC
`timestamp` and `time_unix` fields, resolved configuration, runtime and Git
provenance, utilization, and checkpoint events. Explicit `--save-path` and
`--training-log-path` values are honored exactly. Deprecated `--eval-every`
controls dashboard cadence only; `--eval-every-steps` controls evaluation. The
legacy `--train-updates-per-step N` alias maps to `N / num_envs` updates per
transition and cannot be combined with `--updates-per-transition`.

Continuous evaluation records low-action rate, longest same-cell dwell, visit
saturation for legacy clipped inputs, mean reward and shaping contribution, in
addition to collisions, loops, action statistics, and difficulty solve rates.

The optional dashboard retains a bounded whole-run chart history, reports
percentages to two decimal places, and suppresses overlapping episode-axis
labels. Pygame and its event loop run in a separate process; training publishes
only the newest frame through a bounded, non-blocking queue, so a slow or closed
window cannot stall collection. Use `--no-dashboard` for fully headless runs.

For deterministic debugging, replace the performance profile with `strict`
and normally reduce `--num-envs`.

## Evaluation

Render one fresh maze with the saved policy:

```bash
conda run -n ml python MouseMaze/MouseAgent.py \
  --no-train --inference-mazes 1
```

Render several fresh mazes in sequence, or use `0`/`infinite` to continue
until the inference window is closed:

```bash
conda run -n ml python MouseMaze/MouseAgent.py \
  --no-train --inference-mazes 10

conda run -n ml python MouseMaze/MouseAgent.py \
  --no-train --inference-mazes infinite
```

Inference renders at 30 FPS by default. Use `--inference-fps` to change the
speed, for example `--inference-fps 60` for faster playback. The same pygame
window remains open across fresh mazes, preserving its size and input-panel
visibility.

The inference window is resizable and keeps the maze centered with square
cells. The HUD shows the current step, episode limit, and consecutive blocked
steps so a policy waiting for its timeout can be distinguished from a frozen
application. In `local` observation mode, the policy's current observation
footprint is outlined while the rest of the maze is dimmed. Press Escape or
close the window to stop inference. Use `--show-input-channels` to start with a
labeled side panel showing the exact spatial channels supplied to the policy.
Press `I` to toggle that panel while inference is running. The module-level
`DEFAULT_SHOW_INPUT_CHANNELS` setting near the top of `MouseAgent.py` controls
the default when neither CLI form is supplied; `--no-show-input-channels`
overrides it for one run.

The panel shows the channels enabled for the current model: full-map inputs use
`Walls`, `Mouse`, and `Goal`; local inputs use `Walls` and `Goal`. Both modes
may additionally include `Time remaining` and `Visit count`. Continuous inputs
also show labeled `-1` to `+1` gauges and exact values for `Within-cell row` and
`Within-cell col`, plus a color-coded `Previous collision` indicator below the
spatial thumbnails. These are the same proprioception values supplied to the
policy for the displayed action. Recurrent hidden state, previous executed
displacement, previous reward, and episode-reset state are policy context rather
than environment observations and are not included in the panel.

Run the three deterministic held-out suites without training:

```bash
conda run -n ml python MouseMaze/MouseAgent.py \
  --no-train --no-infer --benchmark --benchmark-episodes 2000 \
  --algorithm recurrent_ppo --observation-mode full
```

Inference and benchmark commands without `--save-path` select the newest valid
timestamped frozen-best model and ignore `.latest.pth` sidecars. Pass an
explicit path to load an archived or specific model. Curriculum-stage scores do
not compete for the frozen-best checkpoint; the main `.pth` is created from
unrestricted evaluation, while `.latest.pth` always retains resumable current
training state.

New checkpoints and logs use schema v11. Schema-v10 checkpoints remain
inference- and benchmark-compatible with their clipped visit counts, potential
reward feedback, and existing proprioception, but cannot be resumed for
training after the reward and observation semantics changed. Loading infers the
algorithm, action space, observation encoding, and reward shaping before
constructing the model; an explicit CLI mismatch is rejected.

The exact full-map BFS planner remains available only as an environment sanity
check:

```bash
conda run -n ml python MouseMaze/MouseAgent.py --no-train --planner-fallback
```
