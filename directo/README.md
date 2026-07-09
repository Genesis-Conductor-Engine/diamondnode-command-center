# Directo - Directive Orchestration

Cloud-accessible command center for DiamondNode server.

## Architecture

```
Mobile App → OpenClaw Webhook → Immesaage Protocol → Local CLI Agent
                    ↓
            Notion Bridge Worker
                    ↓
            Handoff Inbox → Daemon → Materializer
```

## Components

### OpenClaw Endpoints
- **Main**: `https://notion-bridge.iholt.workers.dev/claw/openclaw`
- **KimiClaw**: `https://notion-bridge.iholt.workers.dev/claw/kimi`
- **Mobile**: `https://notion-bridge.iholt.workers.dev/claw/openclaw/mobile`

### Local Components
- **Daemon**: `~/gc-workers/diamondnode-integration/bin/diamondnode-daemon.sh`
- **Materializer**: `~/gc-workers/diamondnode-integration/lib/materializer.js`
- **Inbox**: `~/gc-workers/diamondnode-integration/handoffs/inbox/`

## Directive Flow

1. Mobile app sends directive via OpenClaw webhook
2. Notion Bridge Worker validates and forwards to local endpoint
3. Daemon picks up directive from inbox directory
4. Materializer processes directive into executable actions
5. Local CLI agent executes commands on diamondnode server
6. Results returned via reverse channel to mobile app

## Setup

### Git Remote Configuration
```bash
# Add cloud-accessible remote (GitHub)
git remote add origin git@github.com:diamondnode/diamondnode-command-center.git

# Or use Cloudflare Pages
# git remote add cloudflare https://github.com/diamondnode/diamondnode-command-center.git
```

### Enable Mobile Handoff
```bash
# Ensure daemon is running
bash ~/gc-workers/diamondnode-integration/bin/diamondnode-daemon.sh start

# Test connection
curl -X POST https://notion-bridge.iholt.workers.dev/claw/openclaw/mobile \
  -H "Content-Type: application/json" \
  -d '{"type":"ping","id":"test-001","timestamp":"2026-06-10T21:00:00Z"}'
```
