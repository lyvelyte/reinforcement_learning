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

The `auto` profile selects `rtx3090-fast` on a 24GB RTX 3090. It uses 256
environments, BF16/TF32, compiled recurrent kernels, fused Adam, GPU-resident
rollouts, and eight deterministic maze-generation workers. A newly encountered
model/batch shape can spend roughly 1–2 minutes compiling; the compiled kernels
are cached and subsequent launches are much faster.

Full-map training, capped at three million transitions:

```bash
conda run -n ml python MouseMaze/MouseAgent.py \
  --train --no-infer --no-dashboard \
  --algorithm recurrent_ppo --observation-mode full \
  --performance-profile rtx3090-fast --max-env-steps 3000000 \
  --save-path MouseMaze/agent_weights_v2.pth \
  --training-log-path MouseMaze/training_log_v2.jsonl
```

Local 7×7 training, capped at five million transitions:

```bash
conda run -n ml python MouseMaze/MouseAgent.py \
  --train --no-infer --no-dashboard \
  --algorithm recurrent_ppo --observation-mode local --view-size 7 \
  --performance-profile rtx3090-fast --max-env-steps 5000000 \
  --save-path MouseMaze/local_agent_weights_v2.pth \
  --training-log-path MouseMaze/local_training_log_v2.jsonl
```

For deterministic debugging, replace the performance profile with `strict`
and normally reduce `--num-envs`.

## Evaluation

Render one fresh maze with the saved policy (the default is one maze):

```bash
conda run -n ml python MouseMaze/MouseAgent.py \
  --no-train --inference-mazes 1 \
  --save-path MouseMaze/agent_weights_v2.pth
```

Render several fresh mazes in sequence, or use `0`/`infinite` to continue
until the inference window is closed:

```bash
conda run -n ml python MouseMaze/MouseAgent.py \
  --no-train --inference-mazes 10 \
  --save-path MouseMaze/agent_weights_v2.pth

conda run -n ml python MouseMaze/MouseAgent.py \
  --no-train --inference-mazes infinite \
  --save-path MouseMaze/agent_weights_v2.pth
```

Run the three deterministic held-out suites without training:

```bash
conda run -n ml python MouseMaze/MouseAgent.py \
  --no-train --no-infer --benchmark --benchmark-episodes 2000 \
  --algorithm recurrent_ppo --observation-mode full \
  --save-path MouseMaze/agent_weights_v2.pth
```

Checkpoints use schema v2. Earlier `agent_weights*.pth` files and their JSONL
logs remain archived experiments and are intentionally rejected by the new
loader.

The exact full-map BFS planner remains available only as an environment sanity
check:

```bash
conda run -n ml python MouseMaze/MouseAgent.py --no-train --planner-fallback
```
