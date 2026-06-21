# GitHub Branch & PR Instructions

## Branch
```
feature/yennefer-diamondnode-mcp-integration-20260621
```

## Files committed
- `artifacts/yennefer/Containerfile.yennefer-daemon` — production Podman image
- `artifacts/yennefer/docker/entrypoint.sh` — container entrypoint
- `artifacts/patches/neurohumogenistic-yennefer-soul-crystal-wiring.patch` — orchestrator wiring
- `artifacts/diamondnode-cli/yennefer_cli_commands.py` — CLI soul-crystal / coherence / state
- `artifacts/READY-TO-RUN-COMMANDS-20260621.md` — operational runbook
- `Yennefer/genesis-q-mem/yennefer_persistence.py` — production SoulCrystallizer
- `Yennefer/genesis-q-mem/requirements.txt` — +structlog

## Apply orchestrator patch
```bash
cd ~/diamondnode-unified-inference  # or wherever workspace_orchestrator.py lives
patch -p1 < artifacts/patches/neurohumogenistic-yennefer-soul-crystal-wiring.patch
```

## PR title
feat(yennefer): Production persistence layer + full MCP integration
