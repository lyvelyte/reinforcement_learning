# MouseMaze

MouseMaze trains masked recurrent PPO agents on procedurally generated perfect
mazes. The default observation is the full map; `local` mode exposes a centered
7×7 view and uses GRU memory plus random-network-distillation exploration.

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
candidate passes every suite. Fresh training is the default; pass `--resume`
to load the newest timestamped checkpoint and append to its paired JSONL log.

The configured transition budget controls the main learning-rate and RND
schedules. RND reaches zero at that budget. Target-only training beyond the
budget enters a precision phase that progressively halves the learning rate and
entropy coefficient down to stable floors, reducing late policy drift.

Episode limits default to at least 60 steps or six times the maze's shortest
path, whichever is larger, capped by `--max-episode-steps`. This gives the local
agent time to recover from a wrong turn while retaining a finite task budget.

Failed held-out evaluations feed a bounded hard-maze replay pool through
shape-preserving rotations and reflections; the exact held-out grids are never
used for training. The pool uses deterministic reservoir sampling rather than
continually replacing its oldest examples. Its configured 15% fraction is a
maximum: replay is disabled below a 90% validation solve rate, ramps to the
maximum between 90% and 99%, and remains there afterward.
Recurrent PPO trains on 64-step sequences so wrong-turn and recovery behavior
can receive credit across longer local-observation episodes. Configure these
features with `--hard-maze-fraction`, `--hard-maze-pool-size`, and
`--recurrent-sequence-length`.

After the original transition budget, 20 regular evaluations without a new
best policy trigger guarded recovery. Training restores the frozen best policy
and optimizer, uses 5% of the initial learning rate for one million
transitions, and retains the existing entropy floor. An improvement ends
recovery immediately; otherwise the policy rolls back to the frozen best again.
Tune this with `--precision-plateau-evals`, `--precision-recovery-steps`, and
`--precision-recovery-lr-fraction`.

Target-only runs continue evaluating by transition cadence after passing the
nominal episode count; that count no longer causes evaluation after every
learner update.

Full-map target-only training:

```bash
conda run -n ml python MouseMaze/MouseAgent.py \
  --train --no-infer --no-dashboard \
  --algorithm recurrent_ppo --observation-mode full \
  --performance-profile rtx3090-fast
```

Local 7×7 target-only training:

```bash
conda run -n ml python MouseMaze/MouseAgent.py \
  --train --no-infer --no-dashboard \
  --algorithm recurrent_ppo --observation-mode local --view-size 7 \
  --performance-profile rtx3090-fast
```

To restore the episode/transition caps, pass `--no-target-only-stop` with
`--episodes` and/or `--max-env-steps`. DQN and feed-forward PPO always retain
their existing caps.

Fresh training uses one UTC timestamp for paired artifacts under
`MouseMaze/results/models/` and `MouseMaze/results/logs/`. A run requested with
`--resume` reuses the newest model and its matching log. Each JSONL record
contains UTC `timestamp` and `time_unix` fields alongside resolved configuration,
runtime and Git provenance, metrics, utilization, and checkpoint events.
Explicit `--save-path` and `--training-log-path` values are honored exactly.

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
timestamped model. Pass an explicit path to load an archived or specific model.

Checkpoints use schema v2. Earlier root-level `agent_weights*.pth` files and
their JSONL logs remain archived experiments; compatible checkpoints can still
be loaded with an explicit path.

The exact full-map BFS planner remains available only as an environment sanity
check:

```bash
conda run -n ml python MouseMaze/MouseAgent.py --no-train --planner-fallback
```
