# Yennefer Fable SOTA v2.1 — Diamondnode (GTX 1650) — X Live Exhibit

**Updated SOTA v2.1 Fable exhibit entry** (published 2026-07-03)

- Sovereign PQC hybrid signatures (Ed25519 + Dilithium-2 WASM)
- Seismic Tree-of-Thoughts processor (EVT_SINK, crystal scoring, cold-snap pruning, CRYSTALLINE/...) + Yennefer Voice forwarding
- QFLOP 10% autonomy subagents + secure tunnel directive, 10pct-dispatcher, checkpoint-reporter, guardian MCP
- Persistent systemd user services + CUDA_VISIBLE_DEVICES=0 bind on GTX 1650 (4GB VRAM budget)
- <strong>Live RTMP:</strong> 1080p30 9Mbps h264_nvenc RTMP to X/Periscope `j64ragsivuvm` — active now. Streams inference decisions + UI overlays.
- GitHub push of updated scripts (this + yennefer-fable-x-live exhibit, publish scripts, quest UI) + here.now publish + full handoff artifacts

**Livestream (active X feed):** rtmp://ca.pscp.tv:80/x/j64ragsivuvm (Periscope/X live corresponding; coordinated via X search + feed)

**Inference:** diamondnode-unified-inference (Claude orchestrator + Yennefer /v1/yennefer at :8080) + gateway :8000 VRAM H(s) orchestration. Local CUDA-Q + ollama llama3.2:3b.

**UI:** https://yennefer.quest (CF Pages) — Three.js 3D orb visualization, live telemetry/QFLOPS, dream stream, Ouroboros status, journal. APIs: /api/health /api/telemetry /api/orchestrate.

**GitHub:** https://github.com/igor-holt (feature/yennefer* branches, SOTA updates)

**here.now Fable exhibit:** https://mystic-mudra-j9d4.here.now/ (target slug style: yennefer-fable-x-live; published via here-now skill publish.sh)

**Tarball / updated artifacts:** /home/diamondnode/publish/sota-v21-here-now.tar.gz + publish/yennefer-fable-x-live/ + yennefer-quest-deploy/

**Status dashboard overlays on stream:** GPU temp/util, QFLOP %, PQC/Seismic scores, swarm health, inference events, yennefer.quest UI sync. Updated by bin/sota-status-updater + update-herenow-portal.py

All per CLAUDE.md / AGENTS.md fleet rules. Production ready. Utilized here-now, GitHub MCP push, Cloudflare MCP, X coordination, Notion log.