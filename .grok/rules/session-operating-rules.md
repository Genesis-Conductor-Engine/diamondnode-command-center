# Session operating rules

These rules apply to every agent session on this machine — Claude Code, Grok, Codex, Cursor, Gemini, Copilot, OpenCode, OpenClaw, and any other harness. They are not Claude-specific.

## Truthfulness Over Optimism

Never report a finding from a still-running command or an unverified tool response. If an MCP or integration tool returns an empty confirmation, treat it as FAILED — re-run or fall back to a direct file write, and say so explicitly. Never invent URLs, citations, or contract addresses; if a source is unknown, mark it `[UNVERIFIED]`.

## Disk & Environment Preflight

Before any long build, package install, or test run: check free disk with `df -h /` and `df -h /tmp`. If `/private/tmp` exists (macOS), also check `df -h /private/tmp`. If under 5GB free, run `npm cache clean --force`, prune Docker, and report before proceeding. ENOSPC has silently broken shell and file-edit tools in past sessions.

## Async / non-blocking I/O

Every session is asynchronous. Never block the conversation, the event loop, or a parent agent on I/O.

- Start long commands (installs, builds, tests, deploys, SSH, compiles, downloads) in the background and keep working. Do not wait on them in the foreground.
- Never poll status tools in a loop. Use one sleep-then-check, or a single bounded timeout. Subagents must write their outputs to disk before returning.
- Every network and subprocess call must have a timeout. No unbounded `sleep`/retry loops, no blocking `subprocess` without a bound, no `curl` without `--max-time`.
- If a foreground command is still running after a few seconds, background it and continue.
- Checkpoint partial findings to disk after each phase so a session-limit hit does not lose work.

## Git Conventions

- Use separate `-m` flags for multi-line commit messages; heredoc/multi-line strings have failed repeatedly.
- Before committing, run `git log origin/main..HEAD --oneline` — if the branch carries unrelated commits, STOP and report rather than pushing.
- Scope commits narrowly; never commit files containing PII, clearance, or credential content.

## Contract Verification (Foundry/Base)

Always verify with relative source paths and matching compiler settings; absolute paths cause metadata-hash mismatches. Never use jq-based greps to determine verification status — query the explorer API directly and confirm per-contract. Base 8453 is the only chain with live deployments.
