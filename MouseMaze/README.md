# MouseMaze

MouseMaze trains masked recurrent PPO agents on procedurally generated perfect
mazes. The default `local` observation exposes a centered 7×7 view and uses GRU
memory plus random-network-distillation exploration; `full` mode exposes the
complete map.

Observations contain wall, mouse, goal, and remaining-time channels. Local
training never uses the hidden BFS distance reward.

## Environment

Use the repository's shared conda environment. Do not invoke Python or pytest
directly.

```bash
conda run -n ml pytest MouseMaze/test_mouse_agent.py
```

## RTX 3090 training

The `auto` profile selects `rtx3090-fast` on a 24GB RTX 3090. Recurrent PPO
automatically uses 512 environments with that profile and 256 with portable or
strict profiles; an explicit `--num-envs` always wins. The fast profile also
uses BF16/TF32, compiled recurrent kernels, fused Adam, GPU-resident rollouts,
and eight deterministic maze-generation workers. A newly encountered
model/batch shape can spend roughly 1–2 minutes compiling; the compiled kernels
are cached and subsequent launches are much faster.

Recurrent PPO defaults to target-only stopping. After the curriculum completes,
a policy that reaches the target solve rate is frozen and checked on three
deterministic, seed-separated suites. Training stops only when that unchanged
candidate passes every suite. Resume is the default: the trainer loads the
newest timestamped experiment and appends to its paired JSONL log. Pass
`--no-resume` to create a fresh timestamped experiment.

The configured transition budget controls the main learning-rate and RND
schedules. With the default five-million-transition local budget, the learning
rate linearly anneals from `3e-4` to `3e-5`, RND reaches zero independently, and
the learning rate then remains at `3e-5`. The entropy coefficient remains at
`0.01` before and after the budget so target-only training does not silently
lose optimizer intensity or exploration pressure.

Episode limits default to at least 20 steps or four times the maze's shortest
path, whichever is larger, with a 300-step hard cap controlled by
`--max-episode-steps`.

Hard-maze replay is disabled by default. Setting `--hard-maze-fraction` above
zero feeds transformed held-out failures into a bounded deterministic reservoir;
the exact held-out grids are never used for training. Replay ramps from zero at
a 90% validation solve rate to its configured maximum at 99%. No hard pool is
collected or checkpointed while the fraction is zero.

Recurrent PPO uses a 128-unit GRU and 64-step truncated-BPTT sequences. The GRU
hidden state persists across the complete episode; sequence length controls the
gradient-history window, not inference memory lifetime. Minibatches contain 32
sequences, preserving a 2,048-transition effective minibatch and the proven
optimizer-update density. Learner telemetry records that effective size and the
optimizer updates performed by every rollout.

Rollback-based precision recovery is disabled by default, including when a
legacy checkpoint contains an active recovery window. Use
`--precision-recovery` to opt in; the existing
`--precision-plateau-evals`, `--precision-recovery-steps`, and
`--precision-recovery-lr-fraction` controls remain available for experiments.

Target-only runs continue evaluating by transition cadence after passing the
nominal episode count; that count no longer causes evaluation after every
learner update.

Full-map target-only training:

```bash
conda run -n ml python MouseMaze/MouseAgent.py \
  --train --no-resume --no-infer --no-dashboard \
  --algorithm recurrent_ppo --observation-mode full \
  --performance-profile rtx3090-fast
```

Local 7×7 target-only training:

```bash
conda run -n ml python MouseMaze/MouseAgent.py \
  --train --no-resume --no-infer --no-dashboard \
  --algorithm recurrent_ppo --observation-mode local --view-size 7 \
  --performance-profile rtx3090-fast
```

To restore the episode/transition caps, pass `--no-target-only-stop` with
`--episodes` and/or `--max-env-steps`. DQN and feed-forward PPO always retain
their existing caps.

Fresh training uses one UTC timestamp for paired artifacts under
`MouseMaze/results/models/` and `MouseMaze/results/logs/`. The base
`<timestamp>_mousemaze.pth` file is the frozen best policy used by inference and
benchmarks. Its `<timestamp>_mousemaze.latest.pth` sidecar contains the current
resumable policy, optimizer, RND, RNG, curriculum, counters, and sampler state.
The sidecar is written atomically every one million transitions by default and
on graceful interruption; configure the cadence with
`--latest-checkpoint-every-steps`.

Default resume prefers the sidecar and falls back to the frozen-best checkpoint
for legacy experiments. Process speeds use only counters accumulated since the
resume, while lifetime totals remain in their own fields. Post-budget
target-only ETA is reported as unknown. Each JSONL record also contains UTC
`timestamp` and `time_unix` fields, resolved configuration, runtime and Git
provenance, utilization, and checkpoint events. Explicit `--save-path` and
`--training-log-path` values are honored exactly. Deprecated `--eval-every`
controls dashboard cadence only; `--eval-every-steps` controls evaluation. The
legacy `--train-updates-per-step N` alias maps to `N / num_envs` updates per
transition and cannot be combined with `--updates-per-transition`.

The optional dashboard retains a bounded whole-run chart history, reports
percentages to two decimal places, and suppresses overlapping episode-axis
labels. It remains useful interactively; use `--no-dashboard` for the highest
headless throughput.

For deterministic debugging, replace the performance profile with `strict`
and normally reduce `--num-envs`.

## Evaluation

Render one fresh maze with the saved policy (the default is one maze):

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

The inference window is resizable and keeps the maze centered with square
cells. In `local` observation mode, the policy's current observation footprint
is outlined while the rest of the maze is dimmed. Press Escape or close the
window to stop inference.

Run the three deterministic held-out suites without training:

```bash
conda run -n ml python MouseMaze/MouseAgent.py \
  --no-train --no-infer --benchmark --benchmark-episodes 2000 \
  --algorithm recurrent_ppo --observation-mode full
```

Inference and benchmark commands without `--save-path` select the newest valid
timestamped frozen-best model and ignore `.latest.pth` sidecars. Pass an
explicit path to load an archived or specific model.

Checkpoints use schema v2. Earlier root-level `agent_weights*.pth` files and
their JSONL logs remain archived experiments; compatible checkpoints can still
be loaded with an explicit path.

The exact full-map BFS planner remains available only as an environment sanity
check:

```bash
conda run -n ml python MouseMaze/MouseAgent.py --no-train --planner-fallback
```
