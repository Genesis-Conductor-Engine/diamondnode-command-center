#!/usr/bin/env python3
"""Write self vs observer understanding state for WebGPU livestream HUD."""
from __future__ import annotations

import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

OUT = Path("/tmp/sota-livestream/self-observer-state.json")
STATE_DIR = Path("/tmp/sota-livestream")


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
    last_chat = chat_resp.get("last_reply") or {}
    recent = (chat_resp.get("exchanges") or [])[-3:]

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
            "broadcast": "VA RTMP k2atpt1e4x6v · live encoder",
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
            "chat_count": len(chat_resp.get("exchanges") or []),
            "chat_source": last_chat.get("source", "—"),
        },
        "chat": {
            "exchanges": recent,
            "lines": read_txt("chat-response-latest.txt", "—")[:120],
        },
        "x_pulse": x_pulse,
        "viewer_intents": viewer_intents,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"ok": True, "gap": gap, "viewers": viewers}))


if __name__ == "__main__":
    main()