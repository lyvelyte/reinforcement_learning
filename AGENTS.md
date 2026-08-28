# AGENTS.md

## Scope

* These instructions apply repository-wide.
* Follow deeper `AGENTS.md` files for directory-specific rules.
* When instructions conflict, follow the most specific applicable instruction.
* Keep this file practical and update it when workflows, entrypoints, tests,
  versions, or repository layout materially change.

## Repository Map

| Directory                 | Purpose                                                                    | Stack                                                    |
| ------------------------- | -------------------------------------------------------------------------- | -------------------------------------------------------- |
| `marioppo/alveybj/`       | PPO training and evaluation for Atari/ALE environments such as Breakout-v5 | stable-baselines3, CNN PPO; retained commented LSTM path |
| `marioppo/clarity_coder/` | Alternative PPO trainer with a custom callback and GPU support             | stable-baselines3 PPO                                    |
| `MouseMaze/`              | Recurrent agent for procedural mazes with full or 7x7 local observations   | PyTorch recurrent PPO + GRU; retained DQN baselines      |

Key entrypoints:

* `marioppo/alveybj/Train.py`: 10-million-step Breakout-v5 PPO training.
* `marioppo/alveybj/Test.py`: model evaluation and optional video export.
* `marioppo/clarity_coder/Train.py`: 2-million-step PPO training with
  `SaveOnBestTrainingRewardCallback`.
* `marioppo/clarity_coder/Test.py`: evaluates `ppo_level_1_1.zip`.
* `MouseMaze/MouseAgent.py`: recurrent PPO/DQN training, curriculum, RND,
  benchmarking, pygame dashboard, and RTX 3090 profiles.
* `MouseMaze/gen_maze.py`: Wilson's-algorithm perfect-maze generator.

Large or generated artifacts include model checkpoints, `*.pth`, `*.pt`,
`*.zip`, TensorBoard output, evaluation files, and `MouseMaze/results/`. Do not
read or modify them unless the task specifically requires it.

## Environment and Commands

* Use the shared Conda environment named `ml`.

* Run Python scripts with:

  ```bash
  conda run -n ml python <script>.py ...
  ```

* Run tests with:

  ```bash
  conda run -n ml pytest <path-or-nodeid>
  ```

* Do not invoke `python`, `pip`, or `pytest` directly.

* Do not install, remove, or upgrade dependencies unless explicitly requested.

* Do not modify global or user-level environment configuration.

* Use `MPLBACKEND=Agg` for plotting or GUI-adjacent tests when necessary.

* For MouseMaze production training on the RTX 3090, follow
  `MouseMaze/README.md` and use `rtx3090-fast`.

* Use the MouseMaze `strict` profile for deterministic debugging.

* Avoid full production training unless explicitly requested; use a small,
  deterministic smoke configuration for validation.

## Change Rules

* Work only within this project root unless explicitly instructed otherwise.
* Before editing, inspect the relevant code, tests, and nearby conventions.
* Make minimal, focused changes and preserve existing behavior unless the task
  requires a behavior change.
* Prefer existing patterns over new abstractions or dependencies.
* Avoid unrelated refactors, formatting changes, renames, or cleanup.
* Follow PEP 8 and use descriptive names and small functions.
* Add or update type hints when reasonable.
* Add docstrings for new public functions and classes.
* Keep comments concise and limited to non-obvious intent.
* Preserve existing logging, error handling, and deterministic behavior.
* Do not introduce randomness without an explicit seed.
* Do not leave debug prints, temporary instrumentation, or commented-out
  experiments unless explicitly requested.
* Do not overwrite, discard, revert, or clean pre-existing user changes.
* Do not edit checkpoints, logs, datasets, archives, or generated outputs unless
  the task specifically requires it.

## Task Continuation

* Continue until every requested deliverable is implemented and validated, or a
  concrete blocker makes further progress impossible.
* Never end a turn immediately after stating an intended action. When an action
  can be performed with an available tool, perform it in the same turn.
* Do not treat analysis, a plan, a progress summary, or a description of the
  next step as task completion.
* Do not wait for the user to say `continue` while unfinished work remains.
* For tasks with more than two meaningful steps, maintain a concise todo list:

  * Use 3 to 8 outcome-oriented items.
  * Keep exactly one item in progress.
  * Mark an item complete only after implementation or validation succeeds.
* After each tool result:

  1. Record important results, errors, changed files, and decisions.
  2. Decide whether the active todo item is complete.
  3. Continue immediately with the next unfinished action.
* Before changing direction, record the current finding, reason, remaining work,
  and next concrete action.

### Tool or command failures

If a tool call is empty, malformed, interrupted, timed out, or returns a
transport error:

1. Do not assume the action completed.
2. Preserve the exact abnormal result or error.
3. Retry once with a smaller or more targeted invocation.
4. Use another safe method when available.
5. Continue unaffected work instead of ending prematurely.

If a command, edit, or test fails:

1. Read the complete relevant error.
2. Identify the most likely cause.
3. Attempt a focused fix.
4. Re-run the narrowest relevant validation.
5. Do not claim success unless tool output confirms it.

### Recovery after compaction or interruption

After compaction, session restoration, or a long interruption:

1. Re-read the user's latest request.
2. Review the todo list and identify the first unfinished item.
3. Run `git status --short`.
4. Inspect targeted diffs for files already changed.
5. Recover the latest validation result and blocker.
6. Resume the first unfinished item immediately.

Preserve this continuation state:

* Requested outcome and constraints.
* Completed, active, and remaining work.
* Files created, modified, or inspected.
* Important decisions and rationale.
* Exact failures and relevant error messages.
* Validation commands and results.
* Pre-existing changes that must not be overwritten.
* The next concrete tool action.

## Context Discipline

* Search for symbols, filenames, and call sites before reading large files.
* Read the smallest useful range and expand only when focused evidence is
  insufficient.
* Do not repeatedly read unchanged content.
* Prefer focused `grep`, `glob`, symbol search, narrow file reads, and narrow
  tests.
* Do not dump complete logs, generated data, checkpoints, archives, model
  weights, or broad diffs into context.
* For large logs, search for the relevant error, warning, timestamp, or metric,
  then read a narrow surrounding range.
* Summarize command results with the command, exit status, and important output
  or exact failure.
* Preserve exact paths, errors, failing tests, changed files, decisions, and
  validation results in the todo state.
* Use targeted diffs instead of unrestricted `git diff` on a large tree:

  ```bash
  git diff -- <path>
  git diff --cached -- <path>
  ```

## Testing

* Start with the smallest test that directly validates the change.
* Run a broader relevant suite when practical.
* Documentation-only changes do not require runtime tests; state that clearly.
* Do not delete or weaken tests to make a change pass.
* Avoid trivial coverage-only tests, brittle implementation-detail assertions,
  and unnecessary mocking.
* If tests cannot be added or run, explain why and describe the remaining risk.

For changed behavior:

* New feature: test the main path and at least one meaningful edge case.
* Bug fix: add a regression test that fails without the fix.
* Rendering or pygame: use a headless `pygame.Surface`, verify output differs
  from the background, and cover relevant sizes or modes.
* Data flow: assert important shapes, types, ranges, or invariants.
* Performance-sensitive training: run a small deterministic smoke test covering
  initialization, a short update, checkpoint behavior, and expected metrics or
  tensor shapes.

## Git Safety

At the start of an editing task:

```bash
git status --short
```

Record pre-existing changes and never discard or overwrite them.

After editing and validation, inspect:

```bash
git status --short
git diff --stat
git diff -- <relevant-paths>
```

* Do not stage or commit unless the user explicitly requests a commit.

* When a commit is requested, stage only intended paths:

  ```bash
  git add -- <path-1> <path-2>
  ```

* Do not use `git add -A` unless the user explicitly requests the entire working
  tree.

* Do not include unrelated pre-existing changes without explicit approval.

* Do not create an empty commit.

* Do not amend, rewrite history, force-push, or push unless explicitly requested.

* Never stage secrets, credentials, private keys, core dumps, checkpoints,
  archives, logs, datasets, or unintended generated artifacts.

* Leave suspicious files untouched and report them clearly.

## Completion Criteria

A task is complete only when one of these is true:

1. All requested changes are implemented, relevant validation passes, and the
   result is reported.
2. No repository change was needed and the requested analysis or answer is
   complete.
3. A concrete external blocker prevents progress and the blocker, failed
   operation, exact error, completed work, remaining work, and safest next
   action are reported.

A plan, progress checkpoint, announced next action, or partial completion does
not satisfy completion.

Before finishing:

1. Re-read the request and verify every requested item.
2. Inspect `git status --short`, `git diff --stat`, and targeted diffs.
3. Check for accidental or unrelated edits.
4. Run the smallest relevant validation, then broader validation when practical.
5. Fix failures caused by the change.
6. Report changed files, validation commands and results, assumptions, risks,
   blockers, and relevant pre-existing changes.

Ask a question only when missing information materially changes the
implementation and no safe, reversible, project-consistent default exists.
Otherwise, make the safest reasonable choice and report the assumption.

## External Information

* Use current external information only when the task requires it.
* Do not answer time-sensitive questions from memory.
* Prefer sources in this order:

  1. Official project documentation.
  2. Upstream repositories and release notes.
  3. Primary papers or specifications.
  4. Maintainer issue discussions.
  5. Community reports as supporting evidence only.
* Record the relevant source, version, release, or date when external
  information materially affects a decision.
* Do not download or execute untrusted scripts, binaries, models, or installers
  without explicit permission.
* Review external commands before executing them.

## Final Report

Keep the final response concise and factual. State:

* What changed and which files changed.
* Which validation commands ran and their results.
* Assumptions made.
* Unresolved risks, blockers, or incomplete work.
* Relevant pre-existing changes observed but not modified.
