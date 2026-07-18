#!/usr/bin/env python3
"""Write self vs observer understanding state for WebGPU livestream HUD."""
from __future__ import annotations

import json
import math
import re
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

OUT = Path("/tmp/sota-livestream/self-observer-state.json")
STATE_DIR = Path("/tmp/sota-livestream")
INFERENCE_STATE = Path("/home/diamondnode/diamond-node/state/thrml_ebm_state.json")
FRESHNESS_LIMIT_S = 6 * 60 * 60
MAX_SAMPLES_PER_S = 1_000_000
MAX_ABS_ENERGY = 1_000_000
ALLOWED_ENGINES = {"thrml-0.1.3/jax-gpu"}
SENSITIVE_CONTENT_WITHHELD = "[sensitive content withheld]"
PUBLIC_CHAT_FIELD_LIMITS = {
    "question": 280,
    "reply": 120,
    "host": 40,
    "host_label": 80,
    "viewer": 64,
    "source": 64,
}
SENSITIVE_CHAT_PATTERN = re.compile(
    r"(?:\bstream[\s_-]+key\b|rtmps?://|\bbearer(?:\s+|\s*[:=]\s*)\S+|"
    r"(?:^|[^a-z0-9])[\"']?[\w.-]*(?:token|secret)[\w.-]*[\"']?\s*[:=]\s*\S+)",
    re.IGNORECASE,
)

UNAVAILABLE_INFERENCE = {
    "status": "unavailable",
    "engine": None,
    "age_s": None,
    "gate": None,
    "samples_per_s": None,
    "min_energy": None,
}


def public_chat_text(value: object, limit: int) -> str:
    """Project one untrusted chat value without logging rejected content."""
    if not isinstance(value, str):
        return "—"
    if SENSITIVE_CHAT_PATTERN.search(value):
        return SENSITIVE_CONTENT_WITHHELD
    return value[:limit]


def project_chat_exchange(exchange: object) -> dict:
    if not isinstance(exchange, dict):
        exchange = {}
    return {
        field: public_chat_text(exchange.get(field, "—"), limit)
        for field, limit in PUBLIC_CHAT_FIELD_LIMITS.items()
    }


def project_public_chat(chat_response: object, lines: object) -> dict:
    """Return the strict public projection of recent and latest chat state."""
    if not isinstance(chat_response, dict):
        chat_response = {}
    exchanges = chat_response.get("exchanges")
    if not isinstance(exchanges, list):
        exchanges = []
    return {
        "exchanges": [project_chat_exchange(exchange) for exchange in exchanges[-3:]],
        "last_reply": project_chat_exchange(chat_response.get("last_reply")),
        "lines": public_chat_text(lines, 120),
        "count": len(exchanges),
    }


def read_inference_state(path: Path, now: float) -> dict:
    """Return bounded, credential-blind inference state for the public HUD."""
    try:
        raw = json.loads(Path(path).read_text())
        if not isinstance(raw, dict) or raw.get("ok") is not True:
            return UNAVAILABLE_INFERENCE.copy()

        timestamp = raw.get("ts")
        gate = raw.get("energy_gate")
        result = raw.get("result")
        if (
            not _bounded_number(timestamp, 0, now)
            or not isinstance(gate, dict)
            or not isinstance(gate.get("gpu_ok"), bool)
            or not isinstance(result, dict)
        ):
            return UNAVAILABLE_INFERENCE.copy()

        engine = result.get("engine")
        samples_per_s = result.get("samples_per_s")
        min_energy = result.get("min_energy")
        if (
            engine not in ALLOWED_ENGINES
            or not _bounded_number(samples_per_s, 0, MAX_SAMPLES_PER_S)
            or not _bounded_number(min_energy, -MAX_ABS_ENERGY, MAX_ABS_ENERGY)
        ):
            return UNAVAILABLE_INFERENCE.copy()

        age = now - timestamp
        age_s = int(age)
        return {
            "status": "fresh" if age <= FRESHNESS_LIMIT_S else "stale",
            "engine": engine,
            "age_s": age_s,
            "gate": "open" if gate["gpu_ok"] else "closed",
            "samples_per_s": samples_per_s,
            "min_energy": min_energy,
        }
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return UNAVAILABLE_INFERENCE.copy()


def _bounded_number(value: object, lower: float, upper: float) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and lower <= value <= upper
    )


def read_txt(name: str, default: str = "") -> str:
    p = STATE_DIR / name
    return p.read_text().strip() if p.is_file() else default


def fetch_json(url: str) -> dict:
    try:
        with urllib.request.urlopen(url, timeout=2) as r:
            return json.loads(r.read().decode())
    except Exception:
        return {}


def main():
    thermo = fetch_json("http://127.0.0.1:9100/state")
    intent = fetch_json("http://127.0.0.1:8789/intent-signal.json")
    epoch = {}
    ep = Path("/tmp/thermo-epoch/epoch_latest.json")
    if ep.is_file():
        try:
            epoch = json.loads(ep.read_text())
        except Exception:
            pass

    x_pulse = {}
    xp = STATE_DIR / "x-observer-pulse.json"
    if xp.is_file():
        try:
            x_pulse = json.loads(xp.read_text())
        except Exception:
            pass

    chat_resp = {}
    cr = STATE_DIR / "chat-responses.json"
    if cr.is_file():
        try:
            chat_resp = json.loads(cr.read_text())
        except Exception:
            pass

    viewer_intents = {}
    vi = STATE_DIR / "x-viewer-intents.json"
    if vi.is_file():
        try:
            viewer_intents = json.loads(vi.read_text())
        except Exception:
            pass

    viewers = read_txt("live-viewers.txt", "0")
    inference = read_inference_state(INFERENCE_STATE, datetime.now(timezone.utc).timestamp())
    signal = intent.get("signal", read_txt("exhibit-status.txt", "silence").split("intent:")[-1].strip()[:20])
    occupant = read_txt("exhibit-who.txt", "—")
    verdict = read_txt("exhibit-intent-lab.txt", "—")

    self_reduc = float(epoch.get("reducibility_score", 0.5) or 0.5)
    local_obs = min(1.0, (int("".join(c for c in viewers if c.isdigit()) or "0") % 5000) / 5000.0)
    x_obs = float(x_pulse.get("observer_activity", 0) or 0)
    obs_activity = round(max(local_obs, x_obs * 0.7 + local_obs * 0.3), 4) if x_obs else local_obs
    gap = round(abs(self_reduc - obs_activity), 4)
    x_agg = x_pulse.get("aggregate_metrics") or {}
    top_post = (x_pulse.get("top_posts") or [{}])[0] if x_pulse.get("top_posts") else {}
    top_ann = (x_pulse.get("top_annotations") or [{}])[0]
    public_chat = project_public_chat(chat_resp, read_txt("chat-response-latest.txt", "—"))
    last_chat = public_chat["last_reply"]
    recent = public_chat["exchanges"]

    payload = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "gap": gap,
        "self": {
            "intent": intent.get("line", read_txt("exhibit-intent-line.txt", "—"))[:80],
            "signal": signal,
            "temp_c": thermo.get("temperature_c"),
            "vram_pct": (thermo.get("vram") or {}).get("used_pct"),
            "epoch_id": epoch.get("epoch_id", "—"),
            "reducibility": epoch.get("reducibility_score"),
            "self_model": "thermo daemon + Opux epoch + intent oracle",
        },
        "observers": {
            "viewers": viewers,
            "chat_rate": read_txt("chat-activity.txt", "—"),
            "occupant": occupant,
            "verdict": verdict[:60],
            "perceived_signal": read_txt("exhibit-grok-out.txt", "—").split("\n")[0][:80],
            "broadcast": "VA RTMP [credential sealed] · live encoder",
            "x_source": x_pulse.get("source", "—"),
            "x_posts": x_pulse.get("post_count", 0),
            "x_likes": x_agg.get("like_count", 0),
            "x_impressions": x_agg.get("impression_count", 0),
            "x_top_post": (top_post.get("text") or "—")[:80],
            "x_top_author": top_post.get("author", "—"),
            "x_observer_activity": x_pulse.get("observer_activity", 0),
            "x_retweets": x_agg.get("retweet_count", 0),
            "x_replies": x_agg.get("reply_count", 0),
            "x_top_topic": top_ann.get("name", "—") if top_ann else "—",
            "x_communities": x_pulse.get("community_posts", 0),
            "chat_last_reply": (last_chat.get("reply") or "—")[:80],
            "chat_last_host": last_chat.get("host_label", "—"),
            "chat_last_viewer": last_chat.get("viewer", "—"),
            "chat_count": public_chat["count"],
            "chat_source": last_chat.get("source", "—"),
        },
        "chat": {
            "exchanges": recent,
            "lines": public_chat["lines"],
        },
        "inference": inference,
        "x_pulse": x_pulse,
        "viewer_intents": viewer_intents,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"ok": True, "gap": gap, "viewers": viewers}))


if __name__ == "__main__":
    main()
