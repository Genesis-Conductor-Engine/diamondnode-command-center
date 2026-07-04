#!/usr/bin/env python3
"""Configure and generate live stream chat responses for RTMP + WebGPU overlay.

Uses xAI Responses API (Labs v2) when XAI_API_KEY is set; falls back to canned
replies from sota-chat-responses.json. Viewer messages arrive via:
  - POST http://127.0.0.1:8789/chat/inbox  (intent relay)
  - chat-inbox.jsonl append
  - synthetic rotation from X observer pulse + config templates

Writes:
  /tmp/sota-livestream/chat-responses.json
  /tmp/sota-livestream/chat-1.txt … chat-5.txt  (▸ viewer / ◂ host pairs)
"""
from __future__ import annotations

import json
import os
import random
import re
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

STATE = Path("/tmp/sota-livestream")
CONFIG_PATH = Path(__file__).resolve().parent / "sota-chat-responses.json"
INBOX = STATE / "chat-inbox.jsonl"
OUT = STATE / "chat-responses.json"
PROCESSED = STATE / "chat-processed.ids"
RESPONSES_URL = "https://api.x.ai/v1/responses"
CHAT_COMPLETIONS_URL = "https://api.x.ai/v1/chat/completions"
XAI_MCP = "https://xai-mcp.iholt.workers.dev/mcp"

REPLY_SCHEMA = {
    "type": "json_schema",
    "name": "live_chat_reply",
    "schema": {
        "type": "object",
        "properties": {
            "reply": {"type": "string"},
            "host": {"type": "string"},
            "tone": {"type": "string"},
        },
        "required": ["reply", "host"],
        "additionalProperties": False,
    },
    "strict": True,
}


def load_config() -> dict:
    if CONFIG_PATH.is_file():
        try:
            return json.loads(CONFIG_PATH.read_text())
        except Exception:
            pass
    return {}


def load_processed() -> set[str]:
    if not PROCESSED.is_file():
        return set()
    return {ln.strip() for ln in PROCESSED.read_text().splitlines() if ln.strip()}


def mark_processed(msg_id: str) -> None:
    STATE.mkdir(parents=True, exist_ok=True)
    with PROCESSED.open("a") as f:
        f.write(msg_id + "\n")


def read_inbox(limit: int = 20) -> list[dict]:
    if not INBOX.is_file():
        return []
    done = load_processed()
    msgs = []
    for i, line in enumerate(INBOX.read_text().splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            m = json.loads(line)
        except json.JSONDecodeError:
            continue
        mid = m.get("id") or f"inbox-{i}"
        if mid in done:
            continue
        m["id"] = mid
        msgs.append(m)
        if len(msgs) >= limit:
            break
    return msgs


def append_synthetic(cfg: dict, x_pulse: dict, intent: dict) -> dict | None:
    viewers = cfg.get("synthetic_viewers") or []
    if not viewers:
        return None
    pick = random.choice(viewers)
    templates = pick.get("templates") or ["what's happening on stream?"]
    text = random.choice(templates)
    top = (x_pulse.get("top_posts") or [{}])[0]
    if top.get("text") and random.random() < 0.35:
        text = f"re: {(top.get('text') or '')[:60]}"
    return {
        "id": f"synth-{int(time.time())}-{random.randint(1000, 9999)}",
        "user": pick.get("user", "viewer"),
        "text": text,
        "source": "synthetic",
    }


def canned_reply(cfg: dict, text: str, intent: dict, ctx: dict | None = None) -> str:
    canned = cfg.get("canned") or {}
    lower = text.lower()
    ctx = ctx or {}
    epoch = ctx.get("epoch") or {}
    x_pulse = ctx.get("x_pulse") or {}
    reduc = epoch.get("reducibility_score")
    sig = intent.get("signal", "listening")
    line = intent.get("line", "")

    for pattern, key in (cfg.get("triggers") or {}).items():
        if re.search(pattern, lower):
            base = canned.get(key, canned.get("default", "The grove listens."))
            if key == "thermo" and reduc is not None:
                return f"Thermo epoch breathes — reducibility {reduc:.2f}. {base}"[:110]
            if key == "what":
                return f"Self encodes intent ({sig}); observers see the Δ gap on screen."[:110]
            if key == "dunk":
                return canned.get("dunk", base)
            return base[:110]

    if sig in ("storm", "drop", "surge", "shift", "conflict", "silence"):
        return (line or canned.get(sig, canned.get("default")))[:110]

    x_posts = x_pulse.get("post_count", 0)
    if x_posts:
        return f"X field shows {x_posts} live posts — {canned.get('default', 'the grove listens.')}"[:110]

    return canned.get("default", "The grove listens.")[:110]


def parse_response_text(payload: dict) -> dict | None:
    text = (
        payload.get("output_text")
        or ""
    )
    if not text:
        out = payload.get("output") or []
        for item in out:
            if item.get("type") == "message":
                content = item.get("content") or []
                if content and isinstance(content[0], dict):
                    text = content[0].get("text", "")
                    break
    text = (text or "").strip()
    if not text:
        return None
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return {"reply": text[:120], "host": "fox", "tone": "playful"}


def call_responses_api(cfg: dict, ctx: dict, viewer: dict, api_key: str) -> dict | None:
    hosts = cfg.get("hosts") or {}
    persona = intent_host = ctx.get("intent", {}).get("who", cfg.get("persona", "fox"))
    instructions = cfg.get("instructions", "")
    prompt = (
        f"Live broadcast chat reply.\n"
        f"Viewer @{viewer.get('user', 'anon')}: {viewer.get('text', '')}\n"
        f"Intent signal: {ctx.get('intent', {}).get('signal', 'listening')}\n"
        f"Intent line: {ctx.get('intent', {}).get('line', '')}\n"
        f"X pulse posts: {ctx.get('x_pulse', {}).get('post_count', 0)}\n"
        f"Self reducibility: {ctx.get('epoch', {}).get('reducibility_score', '—')}\n"
        f"Default host persona: {persona} ({hosts.get(persona, persona)})\n"
        "Return JSON with reply (max 110 chars), host (fox|j1|j0|j2|med), tone."
    )
    body = {
        "model": cfg.get("model", "grok-4.3"),
        "input": [{"role": "user", "content": prompt}],
        "instructions": instructions,
        "store": False,
        "temperature": 0.75,
        "text": {"format": REPLY_SCHEMA},
    }
    req = urllib.request.Request(
        RESPONSES_URL,
        data=json.dumps(body).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return parse_response_text(json.loads(resp.read().decode()))
    except Exception:
        return None


def call_chat_completions(cfg: dict, ctx: dict, viewer: dict, api_key: str) -> dict | None:
    hosts = cfg.get("hosts") or {}
    persona = ctx.get("intent", {}).get("who", cfg.get("persona", "fox"))
    prompt = (
        f"Reply as Fable 5 host {persona}. Viewer: {viewer.get('text', '')}. "
        f"Signal: {ctx.get('intent', {}).get('signal', '')}. "
        "Return ONLY JSON: {\"reply\":\"...\",\"host\":\"fox\",\"tone\":\"playful\"}"
    )
    body = {
        "model": cfg.get("model", "grok-4.3"),
        "messages": [
            {"role": "system", "content": cfg.get("instructions", "")},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 200,
        "temperature": 0.75,
    }
    req = urllib.request.Request(
        CHAT_COMPLETIONS_URL,
        data=json.dumps(body).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=25) as resp:
            data = json.loads(resp.read().decode())
            text = (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
            return parse_response_text({"output_text": text})
    except Exception:
        return None


def call_xai_mcp(cfg: dict, ctx: dict, viewer: dict, api_key: str) -> dict | None:
    prompt = (
        f"Live stream chat. Viewer @{viewer.get('user')}: {viewer.get('text')}\n"
        f"Context: {json.dumps({'intent': ctx.get('intent'), 'signal': ctx.get('intent', {}).get('signal')}, default=str)[:400]}\n"
        'Return ONLY JSON: {"reply":"max 110 chars","host":"fox","tone":"playful"}'
    )
    body = {
        "jsonrpc": "2.0",
        "id": "chat-1",
        "method": "tools/call",
        "params": {
            "name": "xai_chat",
            "arguments": {
                "prompt": prompt,
                "system_prompt": cfg.get("instructions", ""),
                "model": cfg.get("model", "grok-4.3"),
                "temperature": 0.75,
                "max_tokens": 200,
            },
        },
    }
    req = urllib.request.Request(
        XAI_MCP,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "X-XAI-API-Key": api_key},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=25) as resp:
            payload = json.loads(resp.read().decode())
            result = payload.get("result") or {}
            text = ""
            content = result.get("content")
            if isinstance(content, list) and content:
                text = content[0].get("text", "") if isinstance(content[0], dict) else str(content[0])
            return parse_response_text({"output_text": text})
    except Exception:
        return None


def generate_reply(cfg: dict, ctx: dict, viewer: dict) -> dict:
    api_key = os.environ.get("XAI_API_KEY", "").strip()
    max_len = int(cfg.get("max_reply_chars", 110))
    hosts = cfg.get("hosts") or {}

    parsed = None
    source = "canned"
    if api_key:
        parsed = call_responses_api(cfg, ctx, viewer, api_key)
        if parsed:
            source = "responses_api"
        if not parsed:
            parsed = call_chat_completions(cfg, ctx, viewer, api_key)
            if parsed:
                source = "chat_completions"
        if not parsed:
            parsed = call_xai_mcp(cfg, ctx, viewer, api_key)
            if parsed:
                source = "xai_mcp"

    if not parsed:
        reply = canned_reply(cfg, viewer.get("text", ""), ctx.get("intent") or {}, ctx)
        host = ctx.get("intent", {}).get("who", cfg.get("persona", "fox"))
        parsed = {"reply": reply, "host": host, "tone": "canned"}

    host_key = str(parsed.get("host", cfg.get("persona", "fox"))).lower()
    host_label = hosts.get(host_key, host_key)
    reply = str(parsed.get("reply", ""))[:max_len]

    return {
        "id": viewer.get("id"),
        "viewer": viewer.get("user", "viewer"),
        "question": viewer.get("text", ""),
        "reply": reply,
        "host": host_key,
        "host_label": host_label,
        "tone": parsed.get("tone", "playful"),
        "source": source,
        "ts": datetime.now(timezone.utc).isoformat(),
    }


def load_context() -> dict:
    ctx: dict = {}
    for name, key in (
        ("intent-signal.json", "intent"),
        ("x-observer-pulse.json", "x_pulse"),
    ):
        p = STATE / name
        if p.is_file():
            try:
                ctx[key] = json.loads(p.read_text())
            except Exception:
                pass
    ep = Path("/tmp/thermo-epoch/epoch_latest.json")
    if ep.is_file():
        try:
            ctx["epoch"] = json.loads(ep.read_text())
        except Exception:
            pass
    return ctx


def should_call_api(cfg: dict) -> bool:
    every = int(cfg.get("api_every_seconds", 45))
    stamp = STATE / "chat-last-api.ts"
    now = time.time()
    if stamp.is_file():
        try:
            last = float(stamp.read_text().strip())
            if now - last < every:
                return False
        except ValueError:
            pass
    stamp.write_text(str(now))
    return True


def write_overlay(exchanges: list[dict], cfg: dict) -> None:
    STATE.mkdir(parents=True, exist_ok=True)
    lines = []
    for ex in exchanges[-3:]:
        lines.append(f"▸ @{ex['viewer']}: {ex['question'][:72]}")
        lines.append(f"◂ {ex['host_label']}: {ex['reply'][:72]}")
    while len(lines) < 5:
        lines.append("")
    for i in range(5):
        (STATE / f"chat-{i + 1}.txt").write_text((lines[i] if i < len(lines) else "") + "\n")
    if exchanges:
        latest = exchanges[-1]
        (STATE / "chat-response-latest.txt").write_text(
            f"◂ {latest['host_label']}: {latest['reply']}\n"
        )
        activity = f"chat:{len(exchanges)} replies · {latest['source']} · {latest['host']}"
        (STATE / "chat-activity.txt").write_text(activity + "\n")


def run_once(force_api: bool = False) -> dict:
    cfg = load_config()
    ctx = load_context()
    x_pulse = ctx.get("x_pulse") or {}
    intent = ctx.get("intent") or {}

    prev: dict = {}
    if OUT.is_file():
        try:
            prev = json.loads(OUT.read_text())
        except Exception:
            pass
    exchanges: list[dict] = list(prev.get("exchanges") or [])[-20:]

    inbox = read_inbox()
    if not inbox:
        synth = append_synthetic(cfg, x_pulse, intent)
        if synth:
            inbox = [synth]

    use_api = force_api or should_call_api(cfg)
    api_key = os.environ.get("XAI_API_KEY", "").strip()

    for viewer in inbox[: int(cfg.get("max_exchanges", 3))]:
        if use_api or not api_key:
            ex = generate_reply(cfg, ctx, viewer)
        else:
            ex = {
                "id": viewer.get("id"),
                "viewer": viewer.get("user", "viewer"),
                "question": viewer.get("text", ""),
                "reply": canned_reply(cfg, viewer.get("text", ""), intent, ctx),
                "host": intent.get("who", cfg.get("persona", "fox")),
                "host_label": (cfg.get("hosts") or {}).get(
                    intent.get("who", "fox"), "🦊 fox"
                ),
                "tone": "canned",
                "source": "canned_throttle",
                "ts": datetime.now(timezone.utc).isoformat(),
            }
        exchanges.append(ex)
        mark_processed(str(viewer.get("id", "")))

    payload = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "broadcast_id": cfg.get("broadcast_id"),
        "model": cfg.get("model"),
        "exchanges": exchanges[-15:],
        "last_reply": exchanges[-1] if exchanges else None,
        "config": str(CONFIG_PATH),
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    write_overlay(exchanges, cfg)
    return payload


def main():
    import sys
    force = "--force" in sys.argv
    result = run_once(force_api=force)
    print(json.dumps({
        "ok": True,
        "exchanges": len(result.get("exchanges") or []),
        "last": (result.get("last_reply") or {}).get("reply", "")[:80],
    }))


if __name__ == "__main__":
    main()