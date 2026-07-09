#!/usr/bin/env python3
"""X API v2 observer pulse — public_metrics + context for live viewer field.

Uses data dictionary fields:
  tweet.fields: created_at,public_metrics,context_annotations,conversation_id,
                lang,entities,author_id,text
  user.fields: username,name,public_metrics,verified
  expansions: author_id

Writes: /tmp/sota-livestream/x-observer-pulse.json
Requires: X_BEARER_TOKEN (or TWITTER_BEARER_TOKEN) in environment.
"""
from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

OUT = Path("/tmp/sota-livestream/x-observer-pulse.json")
API = "https://api.x.com/2/tweets/search/recent"

TWEET_FIELDS = ",".join([
    "created_at", "public_metrics", "context_annotations", "conversation_id",
    "lang", "entities", "author_id", "text", "possibly_sensitive",
    "referenced_tweets", "reply_settings", "note_tweet", "attachments",
    "community_id", "paid_partnership",
])
USER_FIELDS = "username,name,public_metrics,verified,description,created_at"
EXPANSIONS = "author_id"


def bearer() -> str | None:
    for k in ("X_BEARER_TOKEN", "TWITTER_BEARER_TOKEN", "BEARER_TOKEN"):
        v = os.environ.get(k, "").strip()
        if v:
            return v
    return None


def synthetic_pulse() -> dict:
    return {
        "source": "synthetic",
        "reason": "X_BEARER_TOKEN not configured",
        "ts": datetime.now(timezone.utc).isoformat(),
        "query": "(live OR periscope OR broadcast) lang:en -is:retweet",
        "post_count": 0,
        "aggregate_metrics": {
            "like_count": 0, "reply_count": 0, "retweet_count": 0,
            "quote_count": 0, "impression_count": 0, "bookmark_count": 0,
        },
        "observer_activity": 0.15,
        "top_posts": [],
        "top_annotations": [],
        "rate_limit": {},
    }


def fetch_pulse(query: str, max_results: int = 25) -> dict:
    token = bearer()
    if not token:
        return synthetic_pulse()

    params = urllib.parse.urlencode({
        "query": query,
        "max_results": str(min(100, max(max_results, 10))),
        "tweet.fields": TWEET_FIELDS,
        "user.fields": USER_FIELDS,
        "expansions": EXPANSIONS,
    })
    url = f"{API}?{params}"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    try:
        with urllib.request.urlopen(req, timeout=12) as resp:
            body = json.loads(resp.read().decode())
            headers = {k.lower(): v for k, v in resp.headers.items()}
    except Exception as e:
        out = synthetic_pulse()
        out["reason"] = str(e)
        out["source"] = "error"
        return out

    posts = body.get("data") or []
    users = {u["id"]: u for u in (body.get("includes") or {}).get("users") or []}
    agg = {k: 0 for k in ("like_count", "reply_count", "retweet_count", "quote_count", "impression_count", "bookmark_count")}
    annotations: dict[str, int] = {}
    top_posts = []

    for p in posts:
        pm = p.get("public_metrics") or {}
        for k in agg:
            agg[k] += int(pm.get(k, 0) or 0)
        for ann in p.get("context_annotations") or []:
            name = (ann.get("entity") or {}).get("name") or (ann.get("domain") or {}).get("name")
            if name:
                annotations[name] = annotations.get(name, 0) + 1
        author = users.get(p.get("author_id", ""), {})
        note = (p.get("note_tweet") or {}).get("text")
        display_text = note or p.get("text") or ""
        refs = p.get("referenced_tweets") or []
        top_posts.append({
            "id": p.get("id"),
            "text": display_text[:120],
            "author": author.get("username", "?"),
            "verified": author.get("verified", False),
            "public_metrics": pm,
            "author_metrics": author.get("public_metrics") or {},
            "lang": p.get("lang"),
            "created_at": p.get("created_at"),
            "reply_settings": p.get("reply_settings"),
            "referenced_tweets": refs,
            "has_media": bool((p.get("attachments") or {}).get("media_keys")),
            "community_id": p.get("community_id"),
            "paid_partnership": p.get("paid_partnership", False),
        })

    top_posts.sort(
        key=lambda x: int((x.get("public_metrics") or {}).get("like_count", 0)),
        reverse=True,
    )
    top_ann = sorted(annotations.items(), key=lambda x: -x[1])[:5]
    engagement = sum(agg.values()) or 1
    observer_activity = min(1.0, engagement / 50000.0)

    communities = sum(1 for p in posts if p.get("community_id"))
    return {
        "source": "x_api_v2",
        "data_dictionary": "x-api-v2-post-user",
        "ts": datetime.now(timezone.utc).isoformat(),
        "query": query,
        "post_count": len(posts),
        "aggregate_metrics": agg,
        "observer_activity": round(observer_activity, 4),
        "top_posts": top_posts[:5],
        "top_annotations": [{"name": n, "count": c} for n, c in top_ann],
        "community_posts": communities,
        "rate_limit": {
            "remaining": headers.get("x-rate-limit-remaining"),
            "reset": headers.get("x-rate-limit-reset"),
            "limit": headers.get("x-rate-limit-limit"),
        },
        "meta": body.get("meta") or {},
    }


def main():
    query = os.environ.get(
        "X_OBSERVER_QUERY",
        "(live OR periscope OR broadcast OR space) lang:en -is:retweet",
    )
    pulse = fetch_pulse(query)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(pulse, indent=2) + "\n")
    print(json.dumps({
        "ok": True,
        "source": pulse.get("source"),
        "post_count": pulse.get("post_count"),
        "observer_activity": pulse.get("observer_activity"),
    }))


if __name__ == "__main__":
    main()