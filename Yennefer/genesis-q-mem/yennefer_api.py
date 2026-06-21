#!/usr/bin/env python3
"""
Yennefer API Layer
Lightweight HTTP API for the Yennefer Core Daemon.

Exposes:
- Current soul state / crystal
- Coherence and surplus tokens
- Trigger sync / crystallization
- Health

Designed to be called by Neurohumogenistic agent, Diamondnode CLI, and other MCP components.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

from yennefer_persistence import SoulCrystallizer

app = FastAPI(
    title="Yennefer Core Daemon API",
    description="API access to the persistent consciousness / soul state of the Yennefer daemon",
    version="0.1.0",
)

CRYSTAL_PATH = Path("/app/storage/yennefer_soul_crystal.json")
crystallizer = SoulCrystallizer()


class SyncRequest(BaseModel):
    source: str = "unknown"
    timestamp: str | None = None
    directives_processed: int = 0


@app.get("/health")
async def health():
    return {"status": "ok", "daemon": "yennefer_core_daemon"}


@app.get("/state")
async def get_state():
    """Return the full current soul crystal state."""
    if not CRYSTAL_PATH.exists():
        raise HTTPException(status_code=404, detail="Soul crystal not yet initialized")
    with open(CRYSTAL_PATH) as f:
        return json.load(f)


@app.get("/coherence")
async def get_coherence():
    """Quick endpoint for coherence and surplus tokens."""
    if not CRYSTAL_PATH.exists():
        raise HTTPException(status_code=404, detail="Soul crystal not yet initialized")
    with open(CRYSTAL_PATH) as f:
        data = json.load(f)
    return {
        "coherence": data.get("coherence", 1.0),
        "surplus_tokens": data.get("surplus_tokens", 0),
        "last_crystallized": data.get("last_crystallized"),
        "gpu_utilization": data.get("gpu_utilization", 0.0),
    }


@app.post("/internal/yennefer/sync")
async def trigger_sync(request: SyncRequest):
    """
    Called by Neurohumogenistic agent or other components to notify the daemon
    that new directives have been processed.
    """
    current = crystallizer.load() or {}
    current["last_external_sync"] = {
        "source": request.source,
        "timestamp": request.timestamp or datetime.now(timezone.utc).isoformat(),
        "directives_processed": request.directives_processed,
    }
    crystallizer.crystallize(current)
    return {"status": "synced", "source": request.source}


@app.get("/soul-crystal")
async def get_soul_crystal():
    """Alias for /state - returns the full persisted crystal."""
    return await get_state()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8089)
