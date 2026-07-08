# AGENTS.md

## Purpose
- Keep this file slim, practical, and current. Update this file or a nested
  `AGENTS.md` when repo workflows, entrypoints, tests, version rules, or layout
  change in ways future agents should know.
- These instructions apply repo-wide. Follow any deeper `AGENTS.md` for
  directory-specific guidance.

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
