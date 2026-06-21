#!/usr/bin/env python3
"""
Yennefer Persistence Layer — SoulCrystallizer
Production-grade module for crystallizing daemon state to persistent storage.

Designed for:
- Google Drive backed persistence (/mnt/gdrive_memory)
- Shared memory handoff (/dev/shm)
- QGraph backend integration
- Podman / Quadlet deployment
- Full evt- observability and trace-consent

Part of Genesis Conductor MCP / Yennefer Core Daemon.
"""

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import structlog

logger = structlog.get_logger("yennefer.persistence")

# =============================================================================
# Configuration (override via environment)
# =============================================================================
SOUL_CRYSTAL_PATH = Path(os.getenv(
    "YENNEFER_SOUL_CRYSTAL_PATH",
    "/app/storage/yennefer_soul_crystal.json"
))
SHM_STATE_PATH = Path(os.getenv(
    "YENNEFER_SHM_STATE_PATH",
    "/dev/shm/yennefer_soul_state.json"
))
CRYSTALLIZE_INTERVAL_SECONDS = int(os.getenv("YENNEFER_CRYSTALLIZE_INTERVAL", "10"))


class SoulCrystallizer:
    """
    Handles crystallization of Yennefer daemon state to persistent storage.
    Ensures state survives container restarts via Google Drive mount.
    """

    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or SOUL_CRYSTAL_PATH
        self.shm_path = SHM_STATE_PATH
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.shm_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(
            "soul_crystallizer_initialized",
            storage_path=str(self.storage_path),
            shm_path=str(self.shm_path),
            interval=CRYSTALLIZE_INTERVAL_SECONDS,
        )

    # ------------------------------------------------------------------
    # Internal helpers kept for daemon backward-compat
    # ------------------------------------------------------------------
    def collapse_wave_function(self, soul_state: Dict[str, Any]) -> None:
        """Alias → crystallize(). Kept for legacy daemon code."""
        self.crystallize(soul_state)

    def reconstruct_wave_function(self) -> Optional[Dict[str, Any]]:
        """Alias → load(). Kept for legacy daemon code."""
        return self.load()

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------
    def crystallize(self, state: Dict[str, Any]) -> None:
        """
        Write current state to both persistent storage and shared memory.
        This is the core "breathing" mechanism.
        """
        timestamp = datetime.now(timezone.utc).isoformat()

        enriched_state = {
            **state,
            "last_crystallized": timestamp,
            "coherence": state.get("coherence", state.get("coherence_percent", 100.0)),
            "surplus_tokens": state.get("surplus_tokens", 0),
            "gpu_utilization": state.get("gpu_utilization", 0.0),
        }

        # 1. Write to persistent Google Drive-backed storage
        try:
            tmp = Path(str(self.storage_path) + ".tmp")
            tmp.write_text(json.dumps(enriched_state, indent=2))
            tmp.replace(self.storage_path)
            logger.debug("soul_crystal_written", path=str(self.storage_path))
        except Exception as e:
            logger.error("soul_crystal_write_failed", path=str(self.storage_path), error=str(e))

        # 2. Write to shared memory for fast local access
        try:
            tmp_shm = Path(str(self.shm_path) + ".tmp")
            tmp_shm.write_text(json.dumps(enriched_state, indent=2))
            tmp_shm.replace(self.shm_path)
            logger.debug("shm_state_written", path=str(self.shm_path))
        except Exception as e:
            logger.warning("shm_write_failed", path=str(self.shm_path), error=str(e))

    def load(self) -> Optional[Dict[str, Any]]:
        """Load the latest soul crystal from persistent storage."""
        for candidate in [self.storage_path, self.shm_path]:
            if candidate.exists():
                try:
                    return json.loads(candidate.read_text())
                except Exception as e:
                    logger.warning("soul_crystal_load_failed", path=str(candidate), error=str(e))
        return None

    def run_crystallization_loop(self, get_state_fn, interval: int = CRYSTALLIZE_INTERVAL_SECONDS):
        """
        Blocking loop that crystallizes state every N seconds.
        Used by the main daemon loop.
        """
        logger.info("crystallization_loop_started", interval=interval)
        while True:
            try:
                current_state = get_state_fn()
                self.crystallize(current_state)
            except Exception as e:
                logger.error("crystallization_loop_error", error=str(e))
            time.sleep(interval)


def get_crystallizer() -> SoulCrystallizer:
    return SoulCrystallizer()
