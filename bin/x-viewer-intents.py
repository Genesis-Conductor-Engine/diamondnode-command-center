#!/usr/bin/env python3
"""Build X Web Intent URLs for livestream viewers (tweet, follow, like, RT, reply).

Docs: https://docs.x.com/x-for-websites/web-intents/overview
Writes: /tmp/sota-livestream/x-viewer-intents.json
"""
from __future__ import annotations

import json
import os
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path

CFG = Path(__file__).resolve().parent / "x-viewer-intents.json"
PULSE = Path("/tmp/sota-livestream/x-observer-pulse.json")
OUT = Path("/tmp/sota-livestream/x-viewer-intents.json")


def load_json(path: Path, default: dict | None = None) -> dict:
    if not path.is_file():
        return default or {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return default or {}


def intent_link(kind: str, params: dict) -> str:
    base = f"https://x.com/intent/{kind}"
    q = {k: v for k, v in params.items() if v is not None and v != ""}
    return f"{base}?{urllib.parse.urlencode(q)}" if q else base


def build(cfg: dict, pulse: dict) -> dict:
    broadcast_id = os.environ.get("X_BROADCAST_ID", cfg.get("broadcast_id", "1XxyggNlQjbGM"))
    screen_name = os.environ.get("X_VIEWER_SCREEN_NAME", cfg.get("screen_name", "kovachenterprises"))
    exhibit_url = cfg.get("exhibit_url", "https://yennefer.quest/fable5-exhibit-app.html")
    hearth_url = cfg.get("hearth_url", "https://yennefer.quest/")
    lang = cfg.get("lang", "en")
    related = cfg.get("related", screen_name)
    hashtags = cfg.get("hashtags", "")

    broadcast_url = f"https://x.com/i/broadcasts/{broadcast_id}"
    tweet_text = os.environ.get(
        "X_VIEWER_TWEET_TEXT",
        cfg.get("tweet_text", "Watching Fable 5 live — self vs observers"),
    )

    tweet_params = {
        "text": tweet_text,
        "url": broadcast_url,
        "related": related,
        "hashtags": hashtags,
        "lang": lang,
    }
    follow_params = {"screen_name": screen_name, "lang": lang}
    profile_params = {"screen_name": screen_name}

    intents: dict[str, dict] = {
        "tweet": {
            "kind": "tweet",
            "label": "Post about stream",
            "url": intent_link("tweet", tweet_params),
        },
        "follow": {
            "kind": "follow",
            "label": f"Follow @{screen_name}",
            "url": intent_link("follow", follow_params),
        },
        "profile": {
            "kind": "user",
            "label": f"View @{screen_name}",
            "url": intent_link("user", profile_params),
        },
        "broadcast": {
            "kind": "broadcast",
            "label": "Open broadcast",
            "url": broadcast_url,
        },
        "exhibit": {
            "kind": "companion",
            "label": "Exhibit companion",
            "url": exhibit_url,
        },
        "hearth": {
            "kind": "companion",
            "label": "The Hearth",
            "url": hearth_url,
        },
    }

    top = (pulse.get("top_posts") or [{}])[0] if pulse else {}
    tweet_id = top.get("id")
    if tweet_id:
        intents["like"] = {
            "kind": "like",
            "label": "Like top post",
            "url": intent_link("like", {"tweet_id": tweet_id, "related": related}),
            "tweet_id": tweet_id,
            "author": top.get("author"),
        }
        intents["retweet"] = {
            "kind": "retweet",
            "label": "Repost top pulse",
            "url": intent_link("retweet", {"tweet_id": tweet_id, "related": related}),
            "tweet_id": tweet_id,
        }
        intents["reply"] = {
            "kind": "reply",
            "label": "Reply to top post",
            "url": intent_link("tweet", {"in_reply_to": tweet_id, "related": related, "lang": lang}),
            "tweet_id": tweet_id,
        }

    cta = f"viewers · Post · Follow @{screen_name}"
    if tweet_id:
        cta += f" · ♥ RT @{top.get('author', '?')}"

    return {
        "schema": "x-viewer-intents-v1",
        "ts": datetime.now(timezone.utc).isoformat(),
        "broadcast_id": broadcast_id,
        "screen_name": screen_name,
        "broadcast_url": broadcast_url,
        "exhibit_url": exhibit_url,
        "x_pulse_source": pulse.get("source"),
        "top_post_id": tweet_id,
        "top_post_author": top.get("author"),
        "intents": intents,
        "cta_overlay": cta[:120],
        "companion_hint": f"Companion: {exhibit_url}",
    }


def main():
    cfg = load_json(CFG)
    pulse = load_json(PULSE)
    payload = build(cfg, pulse)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"ok": True, "intents": len(payload.get("intents", {})), "top_post": payload.get("top_post_id")}))


if __name__ == "__main__":
    main()