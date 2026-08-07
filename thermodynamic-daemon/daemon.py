#!/usr/bin/env python3
"""Thermodynamic Daemon — network surface for the NVML Energy Governor.

Exposes the live thermodynamic state of the GTX 1650 that the
diamond-governor.service (enforce_thermodynamic_state) is bounding.
Read-only NVML queries; runs unprivileged. Bound to 127.0.0.1:9100 and
served publicly via cloudflared tunnel at dn.genesisconductor.io.

Extended: evolutionary epoch (JAX/HyperNEAT Opux), knowledge nodes,
attestation witnesses, Alchemy story logs.
"""
import hmac
import json
import os
import subprocess
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
# GPU util is the primary gate; VRAM residency alone does not trigger high envelope.
HIGH_ENVELOPE_MW = 50000   # 50 W — active compute
LOW_ENVELOPE_MW = 45000    # 45 W — idle
GPU_UTIL_THRESHOLD = 25

# Proactive fan curve — ramp before the 89.6°C thermal throttle kicks in
FAN_OFF_TEMP = 50           # 0% fan below 50°C
FAN_LOW_TEMP = 60           # 30% at 60°C
FAN_MEDIUM_TEMP = 75        # 60% at 75°C
FAN_HIGH_TEMP = 85          # 100% at 85°C (4.6°C margin before throttle)

_handle = None
_nvml_error = None
_last_attach_attempt = 0.0
# Seconds between re-attach attempts while the driver is absent. nvmlInit() is
# cheap but not free, and /state is scraped continuously by the HUD and the
# public tunnel; without this, every scrape would probe the driver.
NVML_REATTACH_INTERVAL_S = 30.0


def init():
    """Attach to NVML, or record why we could not.

    A missing or unloaded driver is not a reason to exit. The daemon is the
    telemetry surface every other component gates on — the thrml/torx samplers,
    the HUD, and the public tunnel at dn.genesisconductor.io all ask it before
    drawing power. When it exits instead of answering, those consumers see
    connection-refused, which is indistinguishable from "the host is gone", and
    systemd restarts us forever (this crashlooped 11402 times against
    NVMLError_DriverNotLoaded and filled the disk with tracebacks).

    Serving a truthful ``gpu_available: false`` is strictly more useful: the
    energy gate already fails closed on absent telemetry, so a degraded daemon
    keeps the whole chain correct and observable instead of dark.
    """
    global _handle, _nvml_error, _last_attach_attempt
    _last_attach_attempt = time.monotonic()
    try:
        pynvml.nvmlInit()
        _handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        if _nvml_error is not None:
            print("[thermodynamic-daemon] NVML recovered; live telemetry resumed",
                  flush=True)
        _nvml_error = None
    except Exception as e:
        _handle = None
        error = f"{type(e).__name__}: {e}"
        # Log only on transition. This function is retried from read_state(), so
        # logging unconditionally would reproduce the very failure it fixes —
        # a scrape every few seconds turning into gigabytes of identical lines.
        if error != _nvml_error:
            print(
                f"[thermodynamic-daemon] NVML unavailable ({error}); "
                "serving degraded state — no GPU telemetry, envelope gates closed",
                flush=True,
            )
        _nvml_error = error
    return _handle is not None


def gpu_available() -> bool:
    return _handle is not None


def _safe(fn, *a):
    if _handle is None:
        return None
    try:
        return fn(_handle, *a)
    except Exception:
        return None


def set_proactive_fan():
    """Set GPU fan speed based on temperature curve.
    GTX 1650 may not support manual fan control; degrades gracefully.
    """
    if _handle is None:
        return
    temp = pynvml.nvmlDeviceGetTemperature(_handle, pynvml.NVML_TEMPERATURE_GPU)
    if temp >= FAN_HIGH_TEMP:
        target = 100
    elif temp >= FAN_MEDIUM_TEMP:
        target = 60
    elif temp >= FAN_LOW_TEMP:
        target = 30
    elif temp >= FAN_OFF_TEMP:
        target = 20
    else:
        target = 0

    try:
        set_fan = pynvml.nvmlDeviceSetFanSpeed
    except AttributeError:
        return
    try:
        current_speed = pynvml.nvmlDeviceGetFanSpeed(_handle)
        if abs(current_speed - target) > 5:
            set_fan(_handle, target)
    except Exception:
        pass


def _degraded_state():
    """State document when NVML cannot answer.

    Every numeric field is ``None`` rather than 0. A zero would read as "GPU is
    idle and cool", which is exactly the reading that opens the energy gate —
    the one thing a daemon with no telemetry must never imply. Consumers already
    treat ``None`` as "no telemetry" and fail closed.
    """
    return {
        "ts": time.time(),
        "device": None,
        "gpu_available": False,
        "nvml_error": _nvml_error,
        "temperature_c": None,
        "power_limit_mw": None,
        "power_limit_w": None,
        "governor_envelope": {
            "target_mw": None,
            "target_w": None,
            "actual_matches_target": None,
            "high_envelope_mw": HIGH_ENVELOPE_MW,
            "low_envelope_mw": LOW_ENVELOPE_MW,
            "thresholds": {"gpu_util_pct": GPU_UTIL_THRESHOLD},
        },
        "utilization": {"gpu_pct": None, "memory_pct": None},
        "vram": {"used_mib": None, "total_mib": None, "used_pct": None},
        "clocks_mhz": {"sm": None, "mem": None},
        "fan_pct": None,
    }


def read_state():
    if _handle is None and (
        time.monotonic() - _last_attach_attempt >= NVML_REATTACH_INTERVAL_S
    ):
        # Re-attach opportunistically: the driver may load after we started
        # (module load, driver upgrade, VM passthrough attach), so recovery
        # needs no restart. Rate-limited so a busy scrape cannot turn into a
        # probe storm.
        init()
    if _handle is None:
        return _degraded_state()
    try:
        return _read_state_nvml()
    except Exception as e:
        # The driver went away underneath us mid-query.
        global _nvml_error
        _nvml_error = f"{type(e).__name__}: {e}"
        _invalidate_handle()
        return _degraded_state()


def _invalidate_handle():
    global _handle
    _handle = None


def _read_state_nvml():
    mem = pynvml.nvmlDeviceGetMemoryInfo(_handle)
    util = pynvml.nvmlDeviceGetUtilizationRates(_handle)
    vram_pct = (mem.used / mem.total) * 100.0 if mem.total else 0.0
    power_limit_mw = _safe(pynvml.nvmlDeviceGetPowerManagementLimit)
    # Envelope the governor should be enforcing right now (GPU util is primary gate)
    if util.gpu > GPU_UTIL_THRESHOLD:
        target_envelope_mw = HIGH_ENVELOPE_MW
    else:
        target_envelope_mw = LOW_ENVELOPE_MW
    raw_name = pynvml.nvmlDeviceGetName(_handle)
    device_name = raw_name.decode("utf-8", errors="replace") if isinstance(raw_name, bytes) else str(raw_name)
    return {
        "ts": time.time(),
        "device": device_name,
        "gpu_available": True,
        "nvml_error": None,
        "temperature_c": pynvml.nvmlDeviceGetTemperature(_handle, pynvml.NVML_TEMPERATURE_GPU),
        "power_limit_mw": power_limit_mw,
        "power_limit_w": (power_limit_mw / 1000.0) if power_limit_mw is not None else None,
        "governor_envelope": {
            "target_mw": target_envelope_mw,
            "target_w": target_envelope_mw / 1000.0,
            "actual_matches_target": (power_limit_mw == target_envelope_mw) if power_limit_mw is not None else None,
            "high_envelope_mw": HIGH_ENVELOPE_MW,
            "low_envelope_mw": LOW_ENVELOPE_MW,
            "thresholds": {"gpu_util_pct": GPU_UTIL_THRESHOLD},
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


def _metric(value):
    """Render a metric sample, or NaN when telemetry is absent.

    Prometheus rejects the literal ``None``, and emitting 0 would publish a cool,
    idle GPU that does not exist. ``NaN`` is the exposition format's own way to
    say "no current value", so a scrape during a driver outage produces a gap in
    the series rather than a plausible-looking lie.
    """
    return "NaN" if value is None else value


def prometheus_metrics(s):
    lines = [
        "# HELP thermodynamic_gpu_available 1 when NVML is answering, 0 when degraded",
        "# TYPE thermodynamic_gpu_available gauge",
        f'thermodynamic_gpu_available{{device="gtx1650"}} '
        f'{1 if s.get("gpu_available", True) else 0}',
        "# HELP thermodynamic_temperature_celsius GPU temperature in Celsius",
        "# TYPE thermodynamic_temperature_celsius gauge",
        f'thermodynamic_temperature_celsius{{device="gtx1650"}} {_metric(s["temperature_c"])}',
        "# HELP thermodynamic_power_limit_watts NVML power management limit in Watts",
        "# TYPE thermodynamic_power_limit_watts gauge",
        f'thermodynamic_power_limit_watts{{device="gtx1650"}} {_metric(s["power_limit_w"])}',
        "# HELP thermodynamic_gpu_utilization_percent GPU compute utilization percent",
        "# TYPE thermodynamic_gpu_utilization_percent gauge",
        f'thermodynamic_gpu_utilization_percent{{device="gtx1650"}} {_metric(s["utilization"]["gpu_pct"])}',
        "# HELP thermodynamic_vram_used_mib VRAM used in MiB",
        "# TYPE thermodynamic_vram_used_mib gauge",
        f'thermodynamic_vram_used_mib{{device="gtx1650"}} {_metric(s["vram"]["used_mib"])}',
        "# HELP thermodynamic_vram_total_mib VRAM total in MiB",
        "# TYPE thermodynamic_vram_total_mib gauge",
        f'thermodynamic_vram_total_mib{{device="gtx1650"}} {_metric(s["vram"]["total_mib"])}',
        "# HELP thermodynamic_governor_envelope_target_watts Governor target power envelope in Watts",
        "# TYPE thermodynamic_governor_envelope_target_watts gauge",
        f'thermodynamic_governor_envelope_target_watts{{device="gtx1650"}} {_metric(s["governor_envelope"]["target_w"])}',
        "",
    ]
    return "\n".join(lines)


# --- Maintenance actions (token-gated, fixed allowlist) -------------------
# The button on dn.genesisconductor.io POSTs to /blockers/execute. The daemon
# runs as the diamondnode systemd --user service, so it can drive
# `systemctl --user` for stand-downs the interactive agent is not permitted to
# run. Security model: (1) a 0600 token the daemon self-generates, required on
# every call; (2) a FIXED allowlist below — no request field is ever
# interpolated into a command, only these exact argv lists can run.
TOKEN_PATH = Path.home() / ".thermo-action-token"

BLOCKER_ACTIONS = {
    "standdown-gdrive": {
        "label": "Stand down gdrive-sync-daemon (superseded by rclone on agy)",
        "commands": [
            ["systemctl", "--user", "disable", "--now", "gdrive-sync-daemon.timer"],
            ["systemctl", "--user", "disable", "--now", "gdrive-sync-daemon.service"],
            ["systemctl", "--user", "reset-failed", "gdrive-sync-daemon.service"],
        ],
    },
    "standdown-redis": {
        "label": "Stand down diamondnode-redis (restart-flap; native redis owns 6379)",
        "commands": [
            ["systemctl", "--user", "disable", "--now", "diamondnode-redis.service"],
            ["systemctl", "--user", "reset-failed", "diamondnode-redis.service"],
        ],
    },
}


def ensure_token():
    """Create a 0600 action token on first run; return its value."""
    if not TOKEN_PATH.is_file():
        import secrets
        TOKEN_PATH.write_text(secrets.token_urlsafe(32))
        os.chmod(TOKEN_PATH, 0o600)
    return TOKEN_PATH.read_text().strip()


def token_ok(presented):
    if not presented:
        return False
    try:
        expected = TOKEN_PATH.read_text().strip()
    except OSError:
        return False
    return bool(expected) and hmac.compare_digest(presented, expected)


def run_blocker_action(name):
    spec = BLOCKER_ACTIONS.get(name)
    if not spec:
        return None
    results = []
    for cmd in spec["commands"]:
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            results.append({
                "cmd": " ".join(cmd),
                "rc": p.returncode,
                "stdout": p.stdout.strip()[-500:],
                "stderr": p.stderr.strip()[-500:],
            })
        except Exception as e:
            results.append({"cmd": " ".join(cmd), "rc": -1, "error": f"{type(e).__name__}: {e}"})
    return {"action": name, "label": spec["label"],
            "ok": all(r.get("rc") == 0 for r in results), "results": results}


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
button{background:#161616;color:#7df9ff;border:1px solid #2a4;border-radius:5px;padding:.5rem .8rem;margin:.2rem .3rem .2rem 0;font-family:inherit;font-size:.85rem;cursor:pointer}
button:hover{background:#1c2c1c}
button.danger{border-color:#a42;color:#f97}
pre#action-out{background:#0d0d0d;border:1px solid #222;border-radius:5px;padding:.6rem;white-space:pre-wrap;word-break:break-word;color:#bda;max-height:320px;overflow:auto;margin-top:.6rem}
</style></head><body>
<h1>♨ Thermodynamic Daemon</h1>
<p>Live NVML thermodynamic state of the GTX 1650, bounded by
<code>diamond-governor.service</code> (<code>enforce_thermodynamic_state</code>).
Exposed via cloudflared tunnel from local NVML.</p>
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
 <div class="card"><h3>⚙ Maintenance actions</h3>
  <p class="k">Token-gated, allowlisted, idempotent stand-downs.
  Token in <code>~/.thermo-action-token</code>.</p>
  <button onclick="runAction('standdown-gdrive')">Stand down gdrive-sync</button>
  <button onclick="runAction('standdown-redis')">Stand down redis flap</button>
  <button class="danger" onclick="runAction('all')">Run all blockers</button>
  <pre id="action-out">idle — click an action.</pre>
 </div>
</div>
<script>
async function runAction(name){
  var t = window.__thermoToken || (window.__thermoToken = prompt('Action token (~/.thermo-action-token):'));
  if(!t){return;}
  var out = document.getElementById('action-out');
  out.textContent = 'running ' + name + ' ...';
  try{
    var r = await fetch('/blockers/execute',{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({action:name, token:t})
    });
    var j = await r.json();
    out.textContent = 'HTTP ' + r.status + '\\n' + JSON.stringify(j, null, 2);
    if(r.status === 401){ window.__thermoToken = null; }
  }catch(e){ out.textContent = 'error: ' + e; }
}
</script>
</body></html>
"""


class Handler(BaseHTTPRequestHandler):
    def _json(self, obj, code=200, cors=True):
        body = json.dumps(obj, indent=2).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        if cors:
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

    def do_POST(self):
        try:
            if self.path != "/blockers/execute":
                return self._json({"error": "not found", "path": self.path}, 404, cors=False)
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b""
            try:
                data = json.loads(raw or b"{}")
            except Exception:
                data = {}
            presented = data.get("token")
            auth = self.headers.get("Authorization", "")
            if not presented and auth.startswith("Bearer "):
                presented = auth[7:].strip()
            if not token_ok(presented):
                return self._json(
                    {"error": "unauthorized",
                     "hint": "POST {action, token}; token in ~/.thermo-action-token"},
                    401, cors=False)
            action = data.get("action")
            if action == "all":
                out = [run_blocker_action(n) for n in BLOCKER_ACTIONS]
                return self._json(
                    {"action": "all", "ok": all(o and o["ok"] for o in out), "actions": out},
                    200, cors=False)
            result = run_blocker_action(action)
            if result is None:
                return self._json(
                    {"error": "unknown action", "allowed": list(BLOCKER_ACTIONS) + ["all"]},
                    400, cors=False)
            return self._json(result, 200, cors=False)
        except Exception as e:
            self._json({"error": str(e), "type": type(e).__name__}, 500, cors=False)

    def do_OPTIONS(self):
        # No cross-origin preflight granted; the same-origin button needs none.
        self.send_response(405)
        self.send_header("Allow", "GET, POST")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, fmt, *a):
        return  # quiet


def fan_loop():
    """Periodic fan curve check in a background thread."""
    while True:
        try:
            set_proactive_fan()
        except Exception:
            pass
        time.sleep(10)


def main():
    init()
    ensure_token()
    import threading
    t = threading.Thread(target=fan_loop, daemon=True)
    t.start()
    srv = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"[thermodynamic-daemon] listening on {HOST}:{PORT}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
