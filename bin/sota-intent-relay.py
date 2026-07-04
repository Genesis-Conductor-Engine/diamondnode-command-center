#!/usr/bin/env python3
"""Serve livestream + exhibit relay: intent, coalition, mic-drop, dunk-tank."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

HOST = "127.0.0.1"
PORT = 8789
STATE = Path("/tmp/sota-livestream")
SIGNAL = STATE / "intent-signal.json"
COALITION = STATE / "coalition-signal.json"
MIC_DROP = STATE / "mic-drop.json"
DUNK_TANK = STATE / "dunk-tank.json"
CONNECTION_SHAKE = STATE / "connection-shake.json"
MANIFEST = Path.home() / "digital-assets" / "manifest.json"
DUNK_PY = Path.home() / "bin/dunk-tank.py"
CHAT_INBOX = STATE / "chat-inbox.jsonl"
CHAT_RESPONSES = STATE / "chat-responses.json"
CHAT_RESPONDER = Path.home() / "bin/sota-chat-responder.py"


def load_dunk_module():
    spec = importlib.util.spec_from_file_location("dunk_tank", DUNK_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        return

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        routes = {
            "/intent-signal.json": SIGNAL,
            "/coalition-signal.json": COALITION,
            "/mic-drop.json": MIC_DROP,
            "/dunk-tank.json": DUNK_TANK,
            "/connection-shake.json": CONNECTION_SHAKE,
            "/digital-assets/manifest.json": MANIFEST,
            "/chat/responses.json": CHAT_RESPONSES,
        }
        if self.path == "/health":
            self._json({
                "status": "ok",
                "relay": "sota-intent",
                "routes": list(routes) + ["/dunk-tank/action", "/chat/inbox"],
            })
            return
        target = routes.get(self.path)
        if not target:
            self.send_error(404)
            return
        if not target.exists():
            subprocess.run([sys.executable, str(DUNK_PY)], check=False) if target == DUNK_TANK else None
        if not target.exists():
            self._json({"error": "not found", "path": self.path}, 404)
            return
        try:
            data = json.loads(target.read_text())
        except Exception as e:
            self._json({"error": str(e)}, 500)
            return
        self._json(data)

    def do_POST(self):
        if self.path == "/chat/inbox":
            self._chat_inbox()
            return
        if self.path != "/dunk-tank/action":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length).decode("utf-8", errors="replace")
        try:
            data = json.loads(raw or "{}")
        except json.JSONDecodeError:
            self._json({"error": "invalid JSON"}, 400)
            return
        try:
            dunk = load_dunk_module()
            action = data.get("action", "vote")
            space_id = data.get("space_id", "hearth-stage")
            if action == "threshold":
                result = dunk.set_threshold(
                    space_id,
                    float(data.get("threshold", 0.55)),
                    str(data.get("set_by", "fox")),
                )
            else:
                result = dunk.cast_vote(
                    space_id,
                    str(data.get("vote", "dunk")),
                    float(data.get("stake_wqflop", 1)),
                    str(data.get("voter", "anonymous")),
                )
            self._json(result)
        except Exception as e:
            self._json({"error": str(e)}, 400)

    def _chat_inbox(self):
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length).decode("utf-8", errors="replace")
        try:
            data = json.loads(raw or "{}")
        except json.JSONDecodeError:
            self._json({"error": "invalid JSON"}, 400)
            return
        text = str(data.get("text", "")).strip()
        if not text:
            self._json({"error": "text required"}, 400)
            return
        STATE.mkdir(parents=True, exist_ok=True)
        import time
        msg = {
            "id": data.get("id") or f"inbox-{int(time.time() * 1000)}",
            "user": str(data.get("user", "viewer"))[:32],
            "text": text[:280],
            "source": data.get("source", "relay"),
        }
        with CHAT_INBOX.open("a") as f:
            f.write(json.dumps(msg) + "\n")
        subprocess.run(
            [sys.executable, str(CHAT_RESPONDER), "--force"],
            check=False,
            capture_output=True,
        )
        self._json({"ok": True, "queued": msg["id"], "text": msg["text"][:80]})

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main():
    server = HTTPServer((HOST, PORT), Handler)
    print(f"sota-intent-relay on http://{HOST}:{PORT}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()