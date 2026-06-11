# OpenClaw + Immesaage Quick Start

## Overview

This DiamondNode Command Center connects mobile directives to the local CLI agent via OpenClaw endpoints and the Immesaage protocol.

## Endpoints

### OpenClaw (Primary)
```
POST https://notion-bridge.iholt.workers.dev/claw/openclaw
POST https://notion-bridge.iholt.workers.dev/claw/openclaw/mobile
```

### KimiClaw (Alternative)
```
POST https://notion-bridge.iholt.workers.dev/claw/kimi
```

## Local Daemon

The daemon monitors the handoff inbox and processes directives:

```bash
# Start the daemon
bash ~/gc-workers/diamondnode-integration/bin/diamondnode-daemon.sh start

# Check status
bash ~/gc-workers/diamondnode-integration/bin/diamondnode-daemon.sh status

# View logs
tail -f ~/gc-workers/diamondnode-integration/daemon.log
```

## Directive Format

```json
{
  "id": "unique-directive-id",
  "type": "command|query|offload|notify|execute",
  "timestamp": "2026-06-10T21:00:00Z",
  "payload": {
    "command": "string",
    "args": [],
    "env": {},
    "timeout": 300
  },
  "reply_to": "mobile-session-id"
}
```

## Mobile Integration

### Send Directive (via OpenClaw webhook)

```bash
curl -X POST https://notion-bridge.iholt.workers.dev/claw/openclaw/mobile \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${OPENCLAW_AUTH_TOKEN}" \
  -d '{
    "id": "mobile-desc-001",
    "type": "command",
    "timestamp": "2026-06-10T21:00:00Z",
    "payload": {
      "command": "ls -la /home/diamondnode",
      "timeout": 30
    }
  }'
```

### Receive Response

Responses are returned via the same webhook channel or can be polled from:
```bash
curl -X GET https://notion-bridge.iholt.workers.dev/claw/openclaw/mobile/status/${directive_id}
```

## Git Repository

This repository is the central hub for all DiamondNode agents, gateways, and assets.

```bash
# Clone from local remote (for development)
git clone /tmp/diamondnode-command-center.git diamondnode-command-center

# Or from cloud remote (once configured)
git clone git@github.com:diamondnode/diamondnode-command-center.git
```

## Inbox Directory

Directives are written to and read from:
```
~/gc-workers/diamondnode-integration/handoffs/inbox/
```

The daemon processes files with extensions:
- `.directive.json` - Pending directives
- `.processed.json` - Completed directives (archived)
- `.error.json` - Failed directives

## Architecture Flow

```
┌─────────────┐     ┌──────────────────────┐     ┌─────────────┐
│   Mobile     │────▶│   OpenClaw Webhook   │────▶│ Notion     │
│    App       │     │   (Cloudflare)        │     │  Bridge     │
└─────────────┘     └──────────────────────┘     └──────────┬──┘
                                                         ↓
┌─────────────────────────────────────────────────────────────┐
│                    DiamondNode Server                        │
│  ┌─────────────┐    ┌─────────────┐    ┌───────────────┐  │
│  │  Handoff    │    │   Daemon     │    │  Materializer │  │
│  │   Inbox     │───▶│  (Monitor)   │───▶│ (Processor)   │  │
│  └─────────────┘    └─────────────┘    └───────────────┘  │
│                           │                                  │
│                           ↓                                  │
│                  ┌───────────────┐                         │
│                  │   Local CLI   │                         │
│                  │    Agent      │                         │
│                  └───────────────┘                         │
└─────────────────────────────────────────────────────────────┘
```

## Cloud Access Point

To make this repo cloud-accessible for mobile handoff:

### Option 1: GitHub (Recommended)
```bash
# Create repository on GitHub
git remote add github git@github.com:diamondnode/diamondnode-command-center.git
git push github master
```

### Option 2: Cloudflare Pages
```bash
# Configure Cloudflare Pages with Git integration
# Set build command: echo "DiamondNode Command Center"
# Set publish directory: .
```

### Option 3: Direct SSH Access
```bash
# On mobile device or cloud server:
git clone diamondnode@diamondnode:/home/diamondnode/.git diamondnode-command-center
```

## Security Notes

- All directives are authenticated via OpenClaw auth tokens
- Handoff files in inbox directory are processed and archived
- Sensitive data should use the credential vault at `~/.credential-vault/`
- Never commit secrets to this repository

## Monitoring

```bash
# Check daemon status
bash ~/gc-workers/diamondnode-integration/bin/diamondnode-daemon.sh status

# View handoff queue
ls -la ~/gc-workers/diamondnode-integration/handoffs/inbox/

# Tail daemon logs
tail -f ~/gc-workers/diamondnode-integration/daemon.log
```
