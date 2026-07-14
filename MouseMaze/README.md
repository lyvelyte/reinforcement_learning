# MouseMaze

MouseMaze trains masked recurrent PPO agents on procedurally generated perfect
mazes. The default `local` observation exposes a centered 7×7 view and uses GRU
memory plus random-network-distillation exploration; `full` mode exposes the
complete map. Local observations add a fifth channel containing an egocentric
crop of the episode's explicit visited-cell map.

Observations contain wall, mouse, goal, and remaining-time channels, plus the
local-only visited channel. Local training never uses hidden BFS reward shaping.

## Environment

Use the repository's shared conda environment. Do not invoke Python or pytest
directly.

```bash
conda run -n ml pytest MouseMaze/test_mouse_agent.py
```

## RTX 3090 training

The `auto` profile selects `rtx3090-fast` on a 24GB RTX 3090. Recurrent PPO
automatically uses 768 environments with that profile and 256 with portable or
strict profiles; an explicit `--num-envs` always wins. The fast profile also
uses BF16/TF32, compiled recurrent kernels, fused Adam, GPU-resident rollouts,
and eight deterministic maze-generation workers. A newly encountered
model/batch shape can spend roughly 1–2 minutes compiling; the compiled kernels
are cached and subsequent launches are much faster.

The RTX 3090 default was selected with seed `20260714`, the same 11×11 local
configuration, and eight 128-step rollouts at each candidate size. After
discarding three compile/warm-up rollouts, median end-to-end training throughput
over the remaining five was 5,714 steps/s at 512 environments, 6,041 steps/s at
768 (+5.72%), and 5,999 steps/s at 1,024 (+4.98% versus 512). The prepared maze
queue remained populated throughout the measured rollouts.

Recurrent PPO defaults to finite fresh training. The 11×11 local budget is 100
million transitions and scales linearly with the final maximum dimension (136
million for 15×15 and 191 million for 21×21). Explicit `--max-env-steps`,
`--resume`, and `--target-only-stop` flags override those defaults. Target-only
candidates are still checked on three deterministic, seed-separated suites.

The resolved budget controls every recurrent schedule. RND cosine-decays to
zero across the full run. The learning rate reaches `3e-5` at 80% progress;
during the final 20% precision phase it cosine-decays to `3e-6`, while entropy
anneals from `0.01` to `0.001`. Explicit target-only runs retain those floors.

Local episode limits use the maximum of 20 steps, four times shortest path, and
twice the traversable-cell count. This provides a full depth-first exploration
allowance independently of oracle path length. There is no default hard cap;
`--max-episode-steps` adds one.

Hard-maze replay defaults to 5%. Timed-out training mazes and their unique
symmetries feed shape-specific bounded reservoirs; validation and benchmark
mazes are never added. Replay ramps from zero at a 90% validation solve rate to
its configured maximum at 99%.

Automatic curriculum is the default. It builds an odd size ladder in increments
of four, probes 2,048 deterministic mazes per size, and stages the lower third,
lower two-thirds, then unrestricted distribution using an eight-order
depth-first discovery score. Three consecutive 90% validations promote a stage.
Use `--curriculum-mode manual` for the legacy easy and medium distance ranges.

Maze generation uses Wilson's loop-erased random-walk algorithm. RTX workers
return 64 grids per future and retain four queued batches per worker while
preserving seeded submission order. Curriculum validation runs every 50,000
transitions; final-distribution evaluation changes to every 500,000 transitions.

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

Full-map finite training:

```bash
conda run -n ml python MouseMaze/MouseAgent.py \
  --train --no-resume --no-infer --no-dashboard \
  --algorithm recurrent_ppo --observation-mode full \
  --performance-profile rtx3090-fast
```

Local 7×7 finite training:

```bash
conda run -n ml python MouseMaze/MouseAgent.py \
  --train --no-resume --no-infer --no-dashboard \
  --algorithm recurrent_ppo --observation-mode local --view-size 7 \
  --performance-profile rtx3090-fast
```

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

Checkpoints and logs use schema v3. Schema-v2 and earlier model artifacts remain
archived but are intentionally rejected because they lack visited-map weights;
train a fresh schema-v3 policy before inference or benchmarking.

The exact full-map BFS planner remains available only as an environment sanity
check:

```bash
conda run -n ml python MouseMaze/MouseAgent.py --no-train --planner-fallback
```
