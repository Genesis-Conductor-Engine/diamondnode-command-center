# Mirage workspace — Claude Code integration

[Mirage](https://github.com/strukto-ai/mirage) (`mirage-ai` 0.0.2) mounts many
backends — disk, Notion, GitHub, Google Drive, S3/R2, Redis, Slack, Postgres, … —
as a single FUSE filesystem. Claude Code then reads/writes/greps/pipes across all of
them with ordinary bash, instead of juggling separate SDKs/MCPs.

## Install (already done)
- `uv` project here, pinned to Python 3.12 (mirage-ai requires >=3.12).
- `mirage-ai[fuse]` installed into `.venv` (FUSE via `/usr/bin/fusermount`).

## Run
```bash
mirage-claude              # mount, then launch `claude` inside the mount
mirage-claude --no-claude  # mount, print the path, wait (cd in from another shell)
```
`mirage-claude` is a wrapper on PATH (~/bin) that sources ~/.env and runs
`mirage_claude.py`. The mountpoint is a temp dir under /tmp; the writable `/`
is backed by `~/mirage-workspace/diskroot` (commits lazily on unmount).

## Mounts
| Path     | Backend       | Mode  | Enabled when |
|----------|---------------|-------|--------------|
| `/`      | DiskResource  | WRITE | always (diskroot) |
| `/notion`| NotionResource| READ  | real `NOTION_TOKEN` in env |
| `/github`| GitHubResource| READ  | `GITHUB_TOKEN`/`GH_TOKEN` in env |

Add more backends in `build_resources()` of `mirage_claude.py` — every resource in
`mirage.resource.*` follows the same `Resource(Config(...))` shape, e.g.:
```python
from mirage.resource.r2 import R2Resource, R2Config
resources["/r2"] = (R2Resource(R2Config(...)), MountMode.READ)
```

## Notes
- The published 0.0.2 API differs from the docs site (no `Mount`/`agents.claude_code`
  yet); this setup uses the real `Workspace({path:(resource,mode)}, fuse=True)` API.
- A RAM-only root currently fails OS-level writes in 0.0.2; the Disk root works, so
  that's the default.
