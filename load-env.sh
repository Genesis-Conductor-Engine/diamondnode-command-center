#!/bin/bash
# load-env enhanced for MCP auth
if [ -f "$HOME/.env" ]; then
    set -a
    export $(grep -v "^#" "$HOME/.env" | grep -v "^$" | xargs) || true
    set +a
    echo "✅ Loaded ~/.env"
fi
if [ -f "$HOME/.env.local" ]; then
    set -a
    export $(grep -v "^#" "$HOME/.env.local" | grep -E '^(WORKOS_|NEXT_PUBLIC_WORKOS_)' | grep -v "^$" | xargs) 2>/dev/null || true
    set +a
    echo "✅ Loaded WorkOS from ~/.env.local"
fi
if [ -x "$HOME/bin/credential-vault.sh" ]; then
    for k in XAI_API_KEY X_BEARER_TOKEN TWITTER_BEARER_TOKEN NOTION_TOKEN NOTION_API_TOKEN TELEGRAM_BOT_TOKEN STRIPE_SECRET_KEY SUPABASE_URL VERCEL_TOKEN SLACK_TOKEN FIGMA_ACCESS_TOKEN SUMUP_KEY DISCORD_TOKEN HUGGINGFACE_API_KEY PEEC_AI_KEY TEMPORAL_API_KEY AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY CLOUDINARY_URL ELEGANCE_KEY GC_API_KEY AMBIENT_API_KEY; do
        val="$($HOME/bin/credential-vault.sh get "$k" 2>/dev/null || true)"
        [ -n "$val" ] && export "$k"="$val"
    done
    echo "✅ Vault merged for MCPs"
fi
echo "=== MCP env ==="
for k in GC_MCP_URL NOTION_TOKEN STRIPE_SECRET_KEY TELEGRAM_BOT_TOKEN AMBIENT_API_KEY; do v="${!k:-MISSING}"; echo "  $k: ${v:0:4}***"; done
# yennefer.quest local DNS override (stub resolver 127.0.2.2 REFUSED)
if [ -x "$HOME/bin/fix-yennefer-dns.sh" ]; then
    "$HOME/bin/fix-yennefer-dns.sh" >/dev/null 2>&1 || true
    export HOSTALIASES="${HOME}/.config/yennefer/hosts"
fi
echo "Load OK. fleet_manage mcp auth-inject for more."
