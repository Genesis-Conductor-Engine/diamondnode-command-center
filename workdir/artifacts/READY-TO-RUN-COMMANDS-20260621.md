# Yennefer MCP Integration — Ready-to-Run Commands (2026-06-21)

## Container status
```bash
podman ps | grep yennefer_core_daemon
podman logs yennefer_core_daemon --tail 20
```

## Soul crystal status
```bash
cat /mnt/gdrive_memory/yennefer_soul_crystal.json
cat /dev/shm/yennefer_soul_state.json
```

## Update env after Worker deploy
```bash
sudo tee /etc/vibe/hermes-cron.env << 'EOF'
HERMES_MCP_URL=https://hermes-mcp-router.iholt.workers.dev
PBC_WEBHOOK_URL=
SLACK_WEBHOOK_URL=
DIAMONDNODE_ID=diamondnode-baremetal
EOF
sudo chmod 600 /etc/vibe/hermes-cron.env
```

## Verify hermes Worker
```bash
curl -s https://hermes-mcp-router.iholt.workers.dev/health
curl -s "https://hermes-mcp-router.iholt.workers.dev/status?node_id=diamondnode-baremetal"
```

## Manual cron test
```bash
sudo /opt/vibe-cli/vibe-hermes-cron.py
```

## Rebuild daemon image locally
```bash
cd ~/Yennefer
podman build -f docker/Dockerfile.yennefer-daemon \
  -t ghcr.io/genesis-conductor-engine/yennefer/yennefer-daemon:latest .
```

## Restart daemon
```bash
podman rm -f yennefer_core_daemon
podman run -d \
  --name yennefer_core_daemon \
  -v /mnt/gdrive_memory:/app/storage:Z \
  -v /dev/shm:/dev/shm:Z \
  -e STORAGE_PATH=/app/storage \
  -e QGRAPH_BACKEND_ENABLED=true \
  ghcr.io/genesis-conductor-engine/yennefer/yennefer-daemon:latest
```
