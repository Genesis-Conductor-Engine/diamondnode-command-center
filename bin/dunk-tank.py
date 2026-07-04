#!/usr/bin/env python3
"""Dunk Tank — space hosts set vote-out threshold; audience dunks occupants they dislike.

If dunk_votes / (dunk + save) >= threshold → occupant voted out (DUNKED).
Settlement: wQFLOP unliquid stakes pool; optional Alchemy proof.

Writes: /tmp/sota-livestream/dunk-tank.json
Persists: ~/digital-assets/dunk-tank/state.json
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

SPACES_FILE = Path.home() / "digital-assets/dunk-tank/spaces.json"
STATE_FILE = Path.home() / "digital-assets/dunk-tank/state.json"
OUT = Path("/tmp/sota-livestream/dunk-tank.json")
SHAKE_OUT = Path("/tmp/sota-livestream/connection-shake.json")
INTENT = Path("/tmp/sota-livestream/intent-signal.json")
CAST_ROTATION = ["fox", "j1", "j0", "j2", "med"]
SHAKE_DURATION_S = 8

# Cast member → upstream connection that splashes when voted out
CAST_CONNECTIONS = {
    "fox": {"agent": "grok-cli", "connection": "xai-mcp", "label": "Grok / xAI MCP"},
    "j1": {"agent": "vibe-cli", "connection": "vibe", "label": "Mistral Vibe"},
    "j0": {"agent": "gc-mcp-beta", "connection": "gc-mcp-beta", "label": "GC MCP Beta"},
    "j2": {"agent": "openclaw", "connection": "openclaw", "label": "OpenClaw :18789"},
    "med": {"agent": "unified-inference", "connection": "unified-inference", "label": "Unified Inference :8080"},
}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(p: Path, default=None):
    if not p.is_file():
        return default
    try:
        return json.loads(p.read_text())
    except Exception:
        return default


def write_json(p: Path, data: dict):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2) + "\n")


def load_state() -> dict:
    state = read_json(STATE_FILE, {})
    if not state:
        spaces_def = read_json(SPACES_FILE, {})
        state = {
            "updated_at": now_iso(),
            "spaces": {k: dict(v) for k, v in spaces_def.get("spaces", {}).items()},
            "history": [],
        }
    return state


def save_state(state: dict):
    state["updated_at"] = now_iso()
    write_json(STATE_FILE, state)


def dunk_ratio(space: dict) -> float:
    d = int(space.get("dunk_votes", 0))
    s = int(space.get("save_votes", 0))
    total = d + s
    return round(d / total, 3) if total > 0 else 0.0


def next_occupant(current: str) -> str:
    try:
        i = CAST_ROTATION.index(current)
        return CAST_ROTATION[(i + 1) % len(CAST_ROTATION)]
    except ValueError:
        return CAST_ROTATION[0]


def parse_iso(ts: str) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None


def get_active_shake() -> dict | None:
    shake = read_json(SHAKE_OUT, {})
    if not shake or not shake.get("active"):
        return None
    expires = parse_iso(shake.get("expires_at", ""))
    if expires and datetime.now(timezone.utc) >= expires:
        shake["active"] = False
        shake["recovered_at"] = now_iso()
        write_json(SHAKE_OUT, shake)
        return None
    return shake


def emit_connection_shake(occupant: str, space_id: str) -> dict:
    meta = CAST_CONNECTIONS.get(
        occupant,
        {"agent": occupant, "connection": occupant, "label": occupant},
    )
    started = datetime.now(timezone.utc)
    shake = {
        "type": "CONNECTION_SHAKE",
        "active": True,
        "target": occupant,
        "space_id": space_id,
        "agent": meta["agent"],
        "connection": meta["connection"],
        "label": meta["label"],
        "intensity": 0.85,
        "duration_s": SHAKE_DURATION_S,
        "started_at": started.isoformat(),
        "expires_at": (started + timedelta(seconds=SHAKE_DURATION_S)).isoformat(),
        "line": f"CONNECTION SHAKE: {occupant}'s {meta['label']} link splashes out",
        "provenance": "dunk-tank:connection-shake",
    }
    write_json(SHAKE_OUT, shake)
    return shake


def apply_oracle_nudge(state: dict) -> None:
    """Intent signal nudges dunk tank (drop/conflict → dunk pressure)."""
    intent = read_json(INTENT, {})
    signal = intent.get("signal", "silence")
    who = intent.get("who", "")
    default_id = read_json(SPACES_FILE, {}).get("default_space", "hearth-stage")
    space = state["spaces"].get(default_id)
    if not space or space.get("status") != "active":
        return
    if who and space.get("occupant") == who:
        if signal in ("drop", "conflict", "storm"):
            space["dunk_votes"] = int(space.get("dunk_votes", 0)) + 1
            space["stakes_wqflop"] = round(float(space.get("stakes_wqflop", 0)) + 0.5, 2)
        elif signal in ("surge", "shift"):
            space["save_votes"] = int(space.get("save_votes", 0)) + 1
            space["stakes_wqflop"] = round(float(space.get("stakes_wqflop", 0)) + 0.25, 2)


def check_resolution(space: dict) -> dict | None:
    ratio = dunk_ratio(space)
    threshold = float(space.get("threshold", 0.55))
    total = int(space.get("dunk_votes", 0)) + int(space.get("save_votes", 0))
    if total < 3:
        return None
    if ratio >= threshold:
        return {"outcome": "dunked", "ratio": ratio, "threshold": threshold}
    if total >= 10 and ratio <= (1 - threshold):
        return {"outcome": "saved", "ratio": ratio, "threshold": threshold}
    return None


def resolve_space(state: dict, space_id: str, outcome: str) -> None:
    space = state["spaces"][space_id]
    occupant = space.get("occupant", "?")
    host = space.get("host", "?")
    event = {
        "at": now_iso(),
        "space_id": space_id,
        "occupant": occupant,
        "outcome": outcome,
        "ratio": dunk_ratio(space),
        "threshold": space.get("threshold"),
        "stakes_wqflop": space.get("stakes_wqflop"),
        "provenance": "dunk-tank:resolution",
    }
    state.setdefault("history", []).insert(0, event)
    state["history"] = state["history"][:20]

    if outcome == "dunked":
        space["status"] = "dunked"
        space["last_dunked"] = occupant
        shake = emit_connection_shake(occupant, space_id)
        event["connection_shake"] = shake
        state["active_shake"] = shake
        space["occupant"] = next_occupant(occupant)
        space["dunk_votes"] = 0
        space["save_votes"] = 0
        space["stakes_wqflop"] = 0
        space["status"] = "active"
        event["line"] = (
            f"SPLASH! {occupant} voted out — {host}'s tank threshold hit · "
            f"{shake['line']}"
        )
    else:
        space["status"] = "saved"
        space["dunk_votes"] = max(0, int(space.get("dunk_votes", 0)) - 2)
        event["line"] = f"{occupant} holds the mic — crowd saves them"


def cast_vote(space_id: str, vote: str, stake: float = 1.0, voter: str = "anonymous") -> dict:
    state = load_state()
    space = state["spaces"].get(space_id)
    if not space:
        raise ValueError(f"unknown space: {space_id}")
    if space.get("status") not in ("active", "saved"):
        raise ValueError("space not accepting votes")

    min_stake = float(space.get("min_stake_wqflop", 0.5))
    stake = max(min_stake, float(stake))

    if vote == "dunk":
        space["dunk_votes"] = int(space.get("dunk_votes", 0)) + 1
    elif vote == "save":
        space["save_votes"] = int(space.get("save_votes", 0)) + 1
    else:
        raise ValueError("vote must be dunk or save")

    space["stakes_wqflop"] = round(float(space.get("stakes_wqflop", 0)) + stake, 2)
    space["status"] = "active"

    resolution = check_resolution(space)
    if resolution:
        resolve_space(state, space_id, resolution["outcome"])

    save_state(state)
    payload = build_payload(state)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n")

    shake = get_active_shake()
    return {
        "ok": True,
        "space_id": space_id,
        "vote": vote,
        "voter": voter,
        "stake_wqflop": stake,
        "ratio": dunk_ratio(space),
        "threshold": space.get("threshold"),
        "occupant": space.get("occupant"),
        "resolution": resolution,
        "connection_shake": shake,
    }


def set_threshold(space_id: str, threshold: float, set_by: str) -> dict:
    if not 0.1 <= threshold <= 0.95:
        raise ValueError("threshold must be 0.1–0.95")
    state = load_state()
    space = state["spaces"].get(space_id)
    if not space:
        raise ValueError(f"unknown space: {space_id}")
    if set_by != space.get("host") and set_by != "admin":
        raise ValueError("only space host can set threshold")
    space["threshold"] = round(threshold, 2)
    space["threshold_set_by"] = set_by
    save_state(state)
    payload = build_payload(state)
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    return {"ok": True, "space_id": space_id, "threshold": space["threshold"], "set_by": set_by}


def build_payload(state: dict | None = None) -> dict:
    state = state or load_state()
    apply_oracle_nudge(state)

    spaces_out = []
    for sid, space in state.get("spaces", {}).items():
        ratio = dunk_ratio(space)
        threshold = float(space.get("threshold", 0.55))
        total = int(space.get("dunk_votes", 0)) + int(space.get("save_votes", 0))
        pct_to_dunk = max(0, round((threshold - ratio) * 100, 1)) if ratio < threshold else 0
        spaces_out.append({
            **space,
            "dunk_ratio": ratio,
            "votes_total": total,
            "pct_to_dunk": pct_to_dunk if total > 0 else 100,
            "verdict_preview": "DUNK IMMINENT" if ratio >= threshold and total >= 3 else (
                "BUILDING" if total < 3 else f"{int(ratio*100)}% dunk · need {int(threshold*100)}%"
            ),
        })
        res = check_resolution(space)
        if res and space.get("status") == "active":
            resolve_space(state, sid, res["outcome"])

    save_state(state)

    default_id = read_json(SPACES_FILE, {}).get("default_space", "hearth-stage")
    primary = next((s for s in spaces_out if s["id"] == default_id), spaces_out[0] if spaces_out else {})

    active_shake = get_active_shake()
    if not active_shake:
        state.pop("active_shake", None)

    return {
        "type": "DUNK_TANK",
        "organizing_principle": "@Coalition",
        "subtitle": "Space host sets threshold · crowd votes out · wQFLOP stakes",
        "updated_at": now_iso(),
        "default_space": default_id,
        "spaces": spaces_out,
        "primary": primary,
        "history": state.get("history", [])[:5],
        "active_shake": active_shake,
        "cast_connections": CAST_CONNECTIONS,
        "settlement": "wqflop-unliquid",
        "rules": {
            "dunk": "dunk_votes / total >= host threshold → occupant voted out + connection shake",
            "save": "crowd can counter-vote to keep occupant on the mic",
            "host_power": "space host sets threshold (0.1–0.95)",
            "shake": f"voted-out occupant's upstream link shakes for {SHAKE_DURATION_S}s",
        },
        "trace_consent": True,
    }


def main():
    payload = build_payload()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    write_json(Path.home() / "digital-assets/dunk-tank/latest.json", payload)
    p = payload.get("primary", {})
    print(json.dumps({
        "ok": True,
        "occupant": p.get("occupant"),
        "ratio": p.get("dunk_ratio"),
        "threshold": p.get("threshold"),
        "verdict": p.get("verdict_preview"),
    }))


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "vote":
        # CLI: dunk-tank.py vote <space> dunk|save [stake]
        sid = sys.argv[2] if len(sys.argv) > 2 else "hearth-stage"
        vote = sys.argv[3] if len(sys.argv) > 3 else "dunk"
        stake = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
        print(json.dumps(cast_vote(sid, vote, stake)))
    elif len(sys.argv) > 1 and sys.argv[1] == "threshold":
        sid = sys.argv[2]
        thr = float(sys.argv[3])
        who = sys.argv[4] if len(sys.argv) > 4 else "fox"
        print(json.dumps(set_threshold(sid, thr, who)))
    else:
        main()