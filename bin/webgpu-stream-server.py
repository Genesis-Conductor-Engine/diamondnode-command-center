#!/usr/bin/env python3
"""HTTP surface for WebGPU self-vs-observers stream (page + state JSON)."""
from __future__ import annotations

import json
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HOST = "127.0.0.1"
PORT = 8790
HTML = Path(__file__).resolve().parent / "webgpu-self-observers.html"
STATE = Path("/tmp/sota-livestream/self-observer-state.json")
STATE_PY = Path(__file__).resolve().parent / "webgpu-self-observer-state.py"


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        return

    def do_GET(self):
        if self.path in ("/", "/index.html", "/webgpu-self-observers.html"):
            body = HTML.read_bytes()
            self._bytes(body, "text/html; charset=utf-8")
            return
        if self.path == "/self-observer-state.json":
            if not STATE.is_file():
                subprocess.run([sys.executable, str(STATE_PY)], check=False)
            if STATE.is_file():
                self._bytes(STATE.read_bytes(), "application/json")
            else:
                self._json({"error": "no state"}, 404)
            return
        if self.path == "/health":
            self._json({"status": "ok", "service": "webgpu-stream-server"})
            return
        self._json({"error": "not found"}, 404)

    def _json(self, obj, code=200):
        self._bytes(json.dumps(obj).encode(), "application/json", code)

    def _bytes(self, body: bytes, ctype: str, code=200):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main():
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"webgpu-stream-server http://{HOST}:{PORT}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()