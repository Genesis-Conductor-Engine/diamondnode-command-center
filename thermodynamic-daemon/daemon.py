#!/usr/bin/env python3
"""Thermodynamic Daemon — network surface for the NVML Energy Governor.

Exposes the live thermodynamic state of the GTX 1650 that the
diamond-governor.service (enforce_thermodynamic_state) is bounding.
Read-only NVML queries; runs unprivileged. Bound to 127.0.0.1:9100 and
served publicly via cloudflared tunnel at dn.genesisconductor.io.

Extended: evolutionary epoch (JAX/HyperNEAT Opux), knowledge nodes,
attestation witnesses, Alchemy story logs.
"""
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pynvml

DAEMON_DIR = Path(__file__).resolve().parent
if str(DAEMON_DIR) not in sys.path:
    sys.path.insert(0, str(DAEMON_DIR))

PORT = 9100
HOST = "127.0.0.1"

# Mirror the governor's envelope logic (diamond-energy-governor.py)
HIGH_ENVELOPE_MW = 50000   # 50 W
LOW_ENVELOPE_MW = 45000    # 45 W
VRAM_THRESHOLD_PCT = 40.0
GPU_UTIL_THRESHOLD = 25

_handle = None


def init():
    global _handle
    pynvml.nvmlInit()
    _handle = pynvml.nvmlDeviceGetHandleByIndex(0)


def _safe(fn, *a):
    try:
        return fn(_handle, *a)
    except Exception as e:
        return None


def read_state():
    mem = pynvml.nvmlDeviceGetMemoryInfo(_handle)
    util = pynvml.nvmlDeviceGetUtilizationRates(_handle)
    vram_pct = (mem.used / mem.total) * 100.0 if mem.total else 0.0
    power_limit_mw = _safe(pynvml.nvmlDeviceGetPowerManagementLimit)
    # Envelope the governor should be enforcing right now
    if vram_pct > VRAM_THRESHOLD_PCT or util.gpu > GPU_UTIL_THRESHOLD:
        target_envelope_mw = HIGH_ENVELOPE_MW
    else:
        target_envelope_mw = LOW_ENVELOPE_MW
    raw_name = pynvml.nvmlDeviceGetName(_handle)
    device_name = raw_name.decode("utf-8", errors="replace") if isinstance(raw_name, bytes) else str(raw_name)
    return {
        "ts": time.time(),
        "device": device_name,
        "temperature_c": pynvml.nvmlDeviceGetTemperature(_handle, pynvml.NVML_TEMPERATURE_GPU),
        "power_limit_mw": power_limit_mw,
        "power_limit_w": (power_limit_mw / 1000.0) if power_limit_mw is not None else None,
        "governor_envelope": {
            "target_mw": target_envelope_mw,
            "target_w": target_envelope_mw / 1000.0,
            "actual_matches_target": (power_limit_mw == target_envelope_mw) if power_limit_mw is not None else None,
            "high_envelope_mw": HIGH_ENVELOPE_MW,
            "low_envelope_mw": LOW_ENVELOPE_MW,
            "thresholds": {"vram_pct": VRAM_THRESHOLD_PCT, "gpu_util": GPU_UTIL_THRESHOLD},
        },
        "utilization": {"gpu_pct": util.gpu, "memory_pct": util.memory},
        "vram": {
            "used_mib": mem.used // 1048576,
            "total_mib": mem.total // 1048576,
            "used_pct": round(vram_pct, 2),
        },
        "clocks_mhz": {
            "sm": _safe(pynvml.nvmlDeviceGetClockInfo, pynvml.NVML_CLOCK_SM),
            "mem": _safe(pynvml.nvmlDeviceGetClockInfo, pynvml.NVML_CLOCK_MEM),
        },
        "fan_pct": _safe(pynvml.nvmlDeviceGetFanSpeed),
    }


def prometheus_metrics(s):
    lines = [
        "# HELP thermodynamic_temperature_celsius GPU temperature in Celsius",
        "# TYPE thermodynamic_temperature_celsius gauge",
        f'thermodynamic_temperature_celsius{{device="gtx1650"}} {s["temperature_c"]}',
        "# HELP thermodynamic_power_limit_watts NVML power management limit in Watts",
        "# TYPE thermodynamic_power_limit_watts gauge",
        f'thermodynamic_power_limit_watts{{device="gtx1650"}} {s["power_limit_w"]}',
        "# HELP thermodynamic_gpu_utilization_percent GPU compute utilization percent",
        "# TYPE thermodynamic_gpu_utilization_percent gauge",
        f'thermodynamic_gpu_utilization_percent{{device="gtx1650"}} {s["utilization"]["gpu_pct"]}',
        "# HELP thermodynamic_vram_used_mib VRAM used in MiB",
        "# TYPE thermodynamic_vram_used_mib gauge",
        f'thermodynamic_vram_used_mib{{device="gtx1650"}} {s["vram"]["used_mib"]}',
        "# HELP thermodynamic_vram_total_mib VRAM total in MiB",
        "# TYPE thermodynamic_vram_total_mib gauge",
        f'thermodynamic_vram_total_mib{{device="gtx1650"}} {s["vram"]["total_mib"]}',
        "# HELP thermodynamic_governor_envelope_target_watts Governor target power envelope in Watts",
        "# TYPE thermodynamic_governor_envelope_target_watts gauge",
        f'thermodynamic_governor_envelope_target_watts{{device="gtx1650"}} {s["governor_envelope"]["target_w"]}',
        "",
    ]
    return "\n".join(lines)


LANDING = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Thermodynamic Daemon — dn.genesisconductor.io</title>
<style>
body{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;background:#0a0a0a;color:#e6e6e6;margin:0;padding:2rem;max-width:920px}
h1{color:#7df9ff;border-bottom:1px solid #333;padding-bottom:.4rem}
code{background:#161616;padding:.1rem .3rem;border-radius:3px}
a{color:#7df9ff}
.row{display:flex;gap:1rem;flex-wrap:wrap}
.card{background:#111;padding:1rem;border:1px solid #222;border-radius:6px;flex:1;min-width:260px}
.k{color:#888}.v{color:#7df9ff;font-weight:600}
</style></head><body>
<h1>♨ Thermodynamic Daemon</h1>
<p>Live NVML thermodynamic state of the GTX 1650, bounded by
<code>diamond-governor.service</code> (<code>enforce_thermodynamic_state</code>).
Tunneled from <code>dn.genesisconductor.io</code> via cloudflared → <code>127.0.0.1:9100</code>.</p>
<div class="row">
 <div class="card"><h3>Endpoints</h3>
  <p><a href="/health">/health</a> — liveness</p>
  <p><a href="/state">/state</a> — full thermodynamic state (JSON)</p>
  <p><a href="/metrics">/metrics</a> — Prometheus exposition</p>
  <p><a href="/knowledge-nodes">/knowledge-nodes</a> — three goals + thermo nodes</p>
  <p><a href="/epoch/latest">/epoch/latest</a> — Opux HyperNEAT epoch</p>
  <p><a href="/attestation/latest">/attestation/latest</a> — .sol witness</p>
  <p><a href="/story/latest">/story/latest</a> — Alchemy story log</p>
  <p><a href="/ag15/research">/ag15/research</a> — AG15 openFDA substrate manifest</p>
  <p><a href="/ag15/verification">/ag15/verification</a> — double-loop authority evt</p>
  <p><a href="/hermes/simulation">/hermes/simulation</a> — pinned diamondnodebot swarm state</p>
 </div>
 <div class="card"><h3>Landauer envelope</h3>
  <p><span class="k">high:</span> <span class="v">50 W</span> (VRAM &gt; 40% or GPU &gt; 25%)</p>
  <p><span class="k">low:</span> <span class="v">45 W</span> (idle)</p>
  <p><span class="k">polling:</span> 0.5 s (governor loop)</p>
 </div>
</div>
</body></html>
"""


class Handler(BaseHTTPRequestHandler):
    def _json(self, obj, code=200):
        body = json.dumps(obj, indent=2).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self, body, ctype, code=200):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body.encode())

    def do_GET(self):
        try:
            if self.path in ("/", "/index.html"):
                return self._body(LANDING, "text/html; charset=utf-8")
            if self.path == "/health":
                return self._json({"status": "ok", "service": "thermodynamic-daemon", "port": PORT})
            if self.path == "/state":
                return self._json(read_state())
            if self.path == "/metrics":
                return self._body(prometheus_metrics(read_state()), "text/plain; version=0.0.4")
            if self.path == "/knowledge-nodes":
                kn = DAEMON_DIR / "knowledge_nodes.json"
                if kn.is_file():
                    return self._json(json.loads(kn.read_text()))
                return self._json({"error": "knowledge_nodes.json missing"}, 404)
            if self.path in ("/epoch/latest", "/epoch"):
                from evolutionary_epoch import load_latest_epoch
                ep = load_latest_epoch()
                return self._json(ep or {"error": "no epoch yet — run epoch_orchestrator.py"})
            if self.path == "/attestation/latest":
                att = Path("/tmp/thermo-epoch/attestations/attestation_latest.json")
                if att.is_file():
                    return self._json(json.loads(att.read_text()))
                return self._json({"error": "no attestation yet"})
            if self.path == "/story/latest":
                story = Path("/tmp/thermo-epoch/story_logs/story_latest.json")
                if story.is_file():
                    return self._json(json.loads(story.read_text()))
                return self._json({"error": "no story log yet"})
            if self.path == "/epoch/run":
                from epoch_orchestrator import run_pipeline
                result = run_pipeline(use_alphagenome=True)
                return self._json(result)
            if self.path in ("/ag15/research", "/ag15/manifest"):
                p = Path("/tmp/ag15-research/manifest.json")
                if not p.is_file():
                    from ag15_openfda_research import run_pipeline as ag15_run
                    return self._json(ag15_run())
                return self._json(json.loads(p.read_text()))
            if self.path == "/ag15/verification":
                p = Path("/tmp/ag15-research/verification/latest.json")
                if p.is_file():
                    return self._json(json.loads(p.read_text()))
                return self._json({"error": "no verification yet — GET /ag15/verify/run"}, 404)
            if self.path == "/ag15/verify/run":
                from ag15_openfda_research import run_pipeline as ag15_run
                from ag15_double_loop_verifier import verify as ag15_verify
                ag15_run()
                return self._json(ag15_verify())
            if self.path == "/hermes/simulation":
                swarm_cfg = Path.home() / "genesis_conductor_engine/swarm/ag15_diamondnodebot_swarm.json"
                manifest = Path("/tmp/ag15-research/manifest.json")
                verification = Path("/tmp/ag15-research/verification/latest.json")
                substrate = Path.home() / "yennefer-breath/state/substrate_hermes.jsonl"
                last_substrate = None
                if substrate.is_file():
                    lines = [ln for ln in substrate.read_text().splitlines() if ln.strip()]
                    if lines:
                        try:
                            last_substrate = json.loads(lines[-1])
                        except Exception:
                            pass
                return self._json({
                    "simulation": "hermes_ag15_substrate",
                    "swarm": json.loads(swarm_cfg.read_text()) if swarm_cfg.is_file() else {},
                    "research": json.loads(manifest.read_text()) if manifest.is_file() else None,
                    "verification": json.loads(verification.read_text()) if verification.is_file() else None,
                    "substrate_hermes_tail": last_substrate,
                    "reachable": True,
                })
            self._json({"error": "not found", "path": self.path}, 404)
        except Exception as e:
            self._json({"error": str(e), "type": type(e).__name__}, 500)

    def log_message(self, fmt, *a):
        return  # quiet


def main():
    init()
    srv = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"[thermodynamic-daemon] listening on {HOST}:{PORT}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
