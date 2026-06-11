# DiamondNode Command Center

**Neural Connection Hub** - Central command center for the DiamondNode server ecosystem, connecting all agents, gateways, access points, and assets through a unified git repository with OpenClaw/Immesaage handoff pipeline.

## Overview

This repository serves as the neural connection point for the entire DiamondNode server, enabling:

- **Mobile → CLI Directive Handoff** via OpenClaw webhooks
- **Immesaage Protocol** for secure communication
- **Cloud-Accessible Git Repository** for distributed access
- **Neural Session Offloading** to Notion via soul-capsule database
- **Agent Orchestration** across all connected systems

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           CLOUD ACCESS LAYER                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐               │
│  │  Mobile App   │   │   GitHub     │   │  Cloudflare   │               │
│  │  (Anywhere)   │   │  (Optional)  │   │   Pages      │               │
│  └──────┬─────────┘   └──────┬─────────┘   └──────┬─────────┘               │
│         │                     │                    │                        │
│         └─────────────────────┼────────────────────┘                        │
│                               ↓                                             │
│                    ┌────────────────────────┐                              │
│                    │    Notion Bridge        │  https://notion-bridge.iholt.workers.dev  │
│                    │    (Cloudflare Worker)   │                              │
│                    └──────────────┬───────────┘                              │
│                               ↓                                             │
└───────────────────────────────┼─────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                         DIAMONDNODE SERVER                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │                        Git Repository                              │ │
│  │  /home/diamondnode/.git  (This Repo)                            │ │
│  │                                                                  │ │
│  │  ├── .github/workflows/    GitHub Actions for handoff           │ │
│  │  ├── .openclaw/            OpenClaw configuration                  │ │
│  │  │   ├── config.json      Endpoint configurations                  │ │
│  │  │   ├── hooks/           Git hooks for auto-processing             │ │
│  │  │   └── QUICKSTART.md    Setup guide                                │ │
│  │  ├── directo/              Directive orchestration docs            │ │
│  │  └── .gitignore           Ignore system files                       │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                                    │                                         │
│                                    ↓                                         │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │                    gc-workers/diamondnode-integration                 │ │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐    │ │
│  │  │  bin/            │  │  lib/            │  │  handoffs/       │    │ │
│  │  │  └─ daemon.sh    │  │  ├─ materializer│  │  └─ inbox/       │    │ │
│  │  │                 │  │  │    .js       │  │    (Monitored)  │    │ │
│  │  │                 │  │  └─ propagator  │  │                 │    │ │
│  │  │                 │  │      .js       │  │                 │    │ │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────┘    │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                                    │                                         │
│                                    ↓                                         │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │                    OpenClaw Endpoints                              │ │
│  │                                                                  │ │
│  │  • Main:     https://notion-bridge.iholt.workers.dev/claw/openclaw │ │
│  │  • Mobile:   https://notion-bridge.iholt.workers.dev/claw/openclaw/mobile│
│  │  • KimiClaw: https://notion-bridge.iholt.workers.dev/claw/kimi     │ │
│  │                                                                  │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                                    │                                         │
│                                    ↓                                         │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │                    Agent Ecosystem                                 │ │
│  │                                                                  │ │
│  │  • Vibe CLI (Current)           • Claude Agents                       │ │
│  │  • Notion Bridge Worker         • GitHub Copilot                       │ │
│  │  • gc-mcp Worker                • KimiClaw                             │ │
│  │  • Diamond Gateway              • OpenClaw                             │ │
│  │  • Materializer/Propagator      • Various MCP Servers                  │ │
│  │                                                                  │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                                                                           │
└─────────────────────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Clone the Repository

```bash
# From local remote (development)
git clone /tmp/diamondnode-command-center.git diamondnode-command-center
cd diamondnode-command-center

# From cloud remote (once configured)
git clone git@github.com:diamondnode/diamondnode-command-center.git
cd diamondnode-command-center
```

### 2. Start the Daemon

```bash
# Navigate to gc-workers
cd ~/gc-workers/diamondnode-integration

# Start the daemon (monitors handoff inbox)
bash bin/diamondnode-daemon.sh start

# Check status
bash bin/diamondnode-daemon.sh status
```

### 3. Send a Test Directive from Mobile

```bash
curl -X POST https://notion-bridge.iholt.workers.dev/claw/openclaw/mobile \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_OPENCLAW_TOKEN" \
  -d '{
    "id": "test-mobile-001",
    "type": "command",
    "timestamp": "2026-06-10T21:00:00Z",
    "payload": {
      "command": "echo Hello from mobile via OpenClaw",
      "timeout": 30
    }
  }'
```

### 4. Create a Handoff File (Manual)

```bash
# Create a directive file in the inbox
echo '{"id":"manual-001","type":"command","payload":{"command":"date"}}' \
  > ~/gc-workers/diamondnode-integration/handoffs/inbox/manual-001.jsonl

# The daemon will process it within 5 seconds
```

## Directory Structure

```
/home/diamondnode/
├── .git/                          # Git repository (this repo)
│   ├── description                # Repository description
│   └── config                    # Git config
├── .github/
│   └── workflows/
│       └── openclaw-handoff.yml   # GitHub Actions workflow
├── .openclaw/                     # OpenClaw configuration
│   ├── config.json               # Endpoint configuration
│   ├── hooks/
│   │   └── pre-receive           # Git hook for auto-processing
│   └── QUICKSTART.md             # Quick start guide
├── directo/                       # Directive orchestration
│   └── README.md                 # Architecture documentation
├── gc-workers/
│   └── diamondnode-integration/
│       ├── bin/
│       │   └── diamondnode-daemon.sh  # Main daemon script
│       ├── lib/
│       │   ├── materializer.js      # Processes directives
│       │   └── propagator.js         # Sends to claws
│       └── handoffs/
│           └── inbox/              # Drop directives here
└── README.md                      # This file
```

## Configuration

### Environment Variables

The NOTION_TOKEN is already configured in `~/.vibe/.env`:

```bash
NOTION_TOKEN='<REDACTED>'
```

Additional variables for the daemon:

```bash
# Daemon configuration
GC_MCP_URL="https://gc-mcp.iholt.workers.dev/mcp"
GC_MCP_INGRESS_AUTH="your-auth-token"
GC_NOTION_BRIDGE_URL="https://notion-bridge.iholt.workers.dev"
GC_NOTION_BRIDGE_AUTH="your-bridge-auth"
POLL_INTERVAL=5  # seconds between inbox checks

# Offload configuration
HAMILTONIAN=0
VRAM_USED_MIB=0
VRAM_TOTAL_MIB=0
```

### OpenClaw Configuration

See `.openclaw/config.json` for endpoint settings:

```json
{
  "endpoints": {
    "openclaw": "https://notion-bridge.iholt.workers.dev/claw/openclaw",
    "kimiclaw": "https://notion-bridge.iholt.workers.dev/claw/kimi"
  },
  "handoff_config": {
    "inbox_dir": "~/gc-workers/diamondnode-integration/handoffs/inbox",
    "daemon_path": "~/gc-workers/diamondnode-integration/bin/diamondnode-daemon.sh",
    "materializer_path": "~/gc-workers/diamondnode-integration/lib/materializer.js"
  }
}
```

## Directive Format

### Standard Directive

```json
{
  "id": "unique-id-uuid-or-timestamp",
  "type": "command|query|offload|notify|execute",
  "timestamp": "2026-06-10T21:00:00Z",
  "payload": {
    "command": "bash command to execute",
    "args": ["arg1", "arg2"],
    "env": {
      "VAR1": "value1",
      "VAR2": "value2"
    },
    "cwd": "/path/to/working/dir",
    "timeout": 300,
    "retries": 3
  },
  "metadata": {
    "source": "mobile|web|api|cli",
    "priority": "low|normal|high|critical",
    "user": "username",
    "session_id": "session-uuid"
  },
  "reply_to": "response-channel-or-callback-url"
}
```

### Offload Directive (VRAM > 90%)

```json
{
  "action": "OFFLOAD",
  "session_id": "session-abc-123",
  "context_buffer": "[CTX] model=llama3 layers=80...",
  "hamiltonian": 9.2,
  "vram_used_mib": 9200,
  "vram_total_mib": 10000,
  "timestamp": "2026-06-10T21:00:00Z"
}
```

## Deploying to Cloud

### Option 1: GitHub (Recommended)

```bash
# On GitHub: Create empty repository
# Name: diamondnode-command-center
# Visibility: Private (recommended)

# On diamondnode server:
cd /home/diamondnode
git remote add github git@github.com:diamondnode/diamondnode-command-center.git
git push github master

# Enable GitHub Actions
git push github master  # Workflow will run automatically
```

### Option 2: Cloudflare Pages

```bash
# Create Cloudflare Pages project
# Connect to GitHub repository
# Build command: echo "DiamondNode Command Center"
# Build output: .
# Environment variables:
#   - NOTION_TOKEN (from ~/.vibe/.env)
#   - GC_MCP_URL
#   - GC_NOTION_BRIDGE_AUTH
```

### Option 3: Direct SSH Access

```bash
# On remote server:
git clone diamondnode@diamondnode:/home/diamondnode diamondnode-command-center

# Or using git over SSH:
git clone ssh://diamondnode@diamondnode/home/diamondnode diamondnode-command-center
```

## Security

### Secrets Management

**NEVER commit these to the repository:**

- `NOTION_TOKEN` - Already in `~/.vibe/.env` (excluded via .gitignore)
- `GC_MCP_INGRESS_AUTH` - MCP authentication
- `GC_NOTION_BRIDGE_AUTH` - Notion Bridge authentication
- `OPENCLAW_AUTH_TOKEN` - Mobile directive authentication
- Any API keys, private keys, or credentials

### Secure Configuration

All sensitive data should be:
1. Stored in `~/.credential-vault/`
2. Loaded via environment variables
3. Passed through secure channels (HTTPS, SSH)
4. Never logged or committed

### Network Security

- OpenClaw webhooks use HTTPS (TLS)
- Immesaage protocol encrypts all directives
- All cloud endpoints are behind Cloudflare (DDoS protection)
- Local daemon runs with least privilege

## Monitoring

### Check Daemon Status

```bash
# Check if daemon is running
bash ~/gc-workers/diamondnode-integration/bin/diamondnode-daemon.sh status

# View daemon logs
tail -f ~/gc-workers/diamondnode-integration/handoffs/results/daemon.log

# View propagator logs
tail -f ~/gc-workers/diamondnode-integration/handoffs/results/propagate.log
```

### Check Inbox Queue

```bash
# View pending directives
ls -la ~/gc-workers/diamondnode-integration/handoffs/inbox/

# View processed directives
ls -la ~/gc-workers/diamondnode-integration/handoffs/processed/

# View materialized scripts
ls -la ~/gc-workers/diamondnode-integration/handoffs/materialized/
```

### Git Status

```bash
# Check repository status
cd /home/diamondnode
git status
git log --oneline --graph

# Check remotes
git remote -v
```

## Integration with Diamond Ecosystem

### Diamond Gateway

The gateway at `/opt/diamond-gateway/gateway.py` monitors GPU VRAM and calculates the Ising Hamiltonian H(s). When H > 8.5 (VRAM > 90%), it triggers offload to Notion.

### Notion Bridge

Cloudflare Worker at `~/genesis/notion-bridge/` handles:
- OpenClaw webhook requests
- KimiClaw endpoint
- Notion soul-capsule offloads
- Authentication and authorization

### GC Workers

- **gc-mcp**: MCP Server for agent orchestration
- **diamondnode-integration**: Daemon, materializer, propagator
- **Materializer**: Converts directives to executable scripts
- **Propagator**: Forwards results to configured claws (slack, telegram, etc.)

## Directive Types

| Type | Description | Example |
|------|-------------|---------|
| `command` | Execute a shell command | `{"command": "ls -la"}` |
| `query` | Retrieve information | `{"query": "status"}` |
| `offload` | VRAM > 90%, save to Notion | `{"action": "OFFLOAD"}` |
| `notify` | Send notification | `{"message": "Alert"}` |
| `execute` | Run complex workflow | `{"script": "..."}` |

## Workflow Examples

### Example 1: Mobile App → CLI Command

```
Mobile App
  ↓ (POST /claw/openclaw/mobile)
Notion Bridge Worker
  ↓ (forwards to inbox)
Handoff Inbox (inbox/directive.jsonl)
  ↓ (daemon polls)
Materializer (converts to run-command.sh)
  ↓ (executes)
Local CLI Agent
  ↓ (returns result)
Notion Bridge (forwards to mobile)
  ↓
Mobile App (receives response)
```

### Example 2: VRAM Offload

```
Diamond Gateway (monitors GPU)
  ↓ (H(s) > 8.5)
Notion Bridge Worker
  ↓ (creates OFFLOAD payload)
Notion Database (soul-capsule)
  ↓ (stores context)
  ✓ Session preserved
```

### Example 3: Git Push → Auto-Process

```
Developer pushes to git
  ↓ (triggers pre-receive hook)
Git Hook (detects handoff files)
  ↓ (triggers daemon)
Daemon (processes inbox)
  ↓ (executes directives)
  ✓ Auto-processed
```

## Troubleshooting

### Daemon won't start

```bash
# Check for running instance
ps aux | grep diamondnode-daemon

# Kill stale instance
pkill -f diamondnode-daemon

# Check PID file
ls -la /tmp/diamondnode-daemon.pid

# Start fresh
bash ~/gc-workers/diamondnode-integration/bin/diamondnode-daemon.sh start
```

### Directives not processing

```bash
# Check inbox for stuck files
ls -la ~/gc-workers/diamondnode-integration/handoffs/inbox/

# Manually process a file
node ~/gc-workers/diamondnode-integration/lib/materializer.js \
  /home/diamondnode/gc-workers/diamondnode-integration/handoffs/inbox/test.jsonl \
  /home/diamondnode/gc-workers/diamondnode-integration/handoffs/materialized

# Check for errors
cat ~/gc-workers/diamondnode-integration/handoffs/results/daemon.log
```

### GitHub Actions failing

```bash
# Check workflow runs on GitHub
# Or check locally:
git log --oneline

# Verify workflow file
cat .github/workflows/openclaw-handoff.yml
```

## Development

### Adding New Directives

1. Create a `.jsonl` file in the inbox
2. Use the standard directive format
3. The daemon will process it automatically

### Extending Materializer

Edit `~/gc-workers/diamondnode-integration/lib/materializer.js` to add new directive types.

### Adding New Claws

Edit `.openclaw/config.json` to add new endpoint configurations.

## References

- [Notion Integration Skill](./.vibe/skills/notion/SKILL.md)
- [GitHub Integration Skill](./.vibe/skills/github/SKILL.md)
- [OpenClaw Documentation](https://github.com/opclaw/opclaw)
- [Immesaage Protocol](https://github.com/opclaw/immesaage)
- [Diamond Gateway](/opt/diamond-gateway/gateway.py)
- [Notion Bridge](~/genesis/notion-bridge/)

## License

MIT License - DiamondNode Ecosystem

---

**Generated by Mistral Vibe**
**Co-Authored-By: Mistral Vibe <vibe@mistral.ai>**
**Date: 2026-06-10**
