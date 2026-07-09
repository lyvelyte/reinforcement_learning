# AGENTS.md

## Purpose
- Keep this file slim, practical, and current. Update this file or a nested
  `AGENTS.md` when repo workflows, entrypoints, tests, version rules, or layout
  change in ways future agents should know.
- These instructions apply repo-wide. Follow any deeper `AGENTS.md` for
  directory-specific guidance.

## Project Layout
Two independent RL projects live side-by-side under this repo root:

| Directory | What it does | Algorithm | Framework |
|-----------|-------------|-----------|-----------|
| `marioppo/alveybj/` | Train and evaluate PPO agents on Atari/ALE envs (Breakout-v5) | PPO + CNN (LSTM variant commented out) | stable-baselines3 |
| `marioppo/clarity_coder/` | Alternative PPO trainer with custom callback, GPU support | PPO | stable-baselines3 |
| `MouseMaze/` | DQN agent navigating a grid maze via 5×5 local view window | DQN + experience replay | PyTorch |

### Key entrypoints
- **`marioppo/alveybj/Train.py`** — Train PPO (10 M timesteps, Breakout-v5, video every 24 episodes). LSTM path is commented out.
- **`marioppo/alveybj/Test.py`** — Evaluate a loaded model (50 evals × 100 steps). Video export available via wrapper flag.
- **`marioppo/clarity_coder/Train.py`** — Train PPO (2 M timesteps, custom `SaveOnBestTrainingRewardCallback`, TensorBoard under `./board/`).
- **`marioppo/clarity_coder/Test.py`** — Evaluate `ppo_level_1_1.zip` (40 episodes × 200 steps).
- **`MouseMaze/MouseAgent.py`** — All-in-one: `ReplayBuffer`, `Maze` env, `QNetwork` (MLP), `MouseAgent` (epsilon-greedy DQN), `MetricsTracker`, matplotlib `Dashboard`, and `train()` function.
- **`MouseMaze/gen_maze.py`** — Maze generator (Wilson's algorithm, perfect mazes).

### Model artifacts
- `marioppo/alveybj/models/best_model.zip` — CNN-PPO weights.
- `marioppo/alveybj/lstm_models/best_model.zip` — RecurrentPPO weights.
- `marioppo/clarity_coder/ppo_level_1_1.zip` — PPO level-1 model.
- `MouseMaze/agent_weights.pth` — DQN PyTorch state dict.

### Logs & telemetry
- `marioppo/alveybj/tensorboard_logs/` — TensorBoard runs for PPO training.
- `marioppo/alveybj/logs/monitor.csv` — OpenAI monitor data.
- `marioppo/alveybj/evaluations.npz` — Saved evaluation metrics.

## Environment & Commands
- Use the shared conda environment named `ml`.
- Run Python scripts with `conda run -n ml python <script>.py ...`.
- Run tests with `conda run -n ml pytest <path-or-nodeid>`.
- Do not invoke `python`, `pip`, or `pytest` directly.
- Do not install or upgrade dependencies unless explicitly requested.

## Change Guidelines
- Only read and modify files within this project root.
- Make minimal, focused changes and preserve existing behavior unless the task
  requires a behavior change.
- Prefer existing project patterns over new abstractions.
- Follow PEP 8. Use descriptive names, small focused functions, and type hints
  for new or modified functions when reasonable.
- Add docstrings for new public functions/classes. Keep inline comments concise
  and only where they explain non-obvious intent.
- Use existing logging/error patterns. Do not leave debug print statements.
- Preserve deterministic behavior; do not introduce randomness without a seed.
- Do not modify environment configuration files or global/user settings.

## Testing
- Run relevant tests for code changes with
  `conda run -n ml pytest <path-or-nodeid>`.
- For plotting or GUI-adjacent tests, use `MPLBACKEND=Agg` when needed.
- Documentation-only changes do not require runtime tests; state that clearly.
- Do not delete, weaken, or add trivial tests. Update or add tests when behavior
  changes enough to need coverage.

## Versions & Commits
- After finishing edits, always commit all changes (even those un-related to the changes Codex made, I just want to keep the repo continously updated with all changes tracked):
  `git add -A`
  `git commit -m "<brief top-level description>"`
