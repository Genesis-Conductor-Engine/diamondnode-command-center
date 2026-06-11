#!/bin/bash
# deploy-cloud.sh - Deploy DiamondNode Command Center to cloud

set -e

echo "=========================================="
echo "DiamondNode Command Center Cloud Deploy"
echo "=========================================="
echo

# Check if we're in the right directory
if [ ! -d ".git" ]; then
    echo "ERROR: Not in a git repository"
    exit 1
fi

# Check current remote
CURRENT_REMOTE=$(git remote get-url origin 2>/dev/null || echo "none")
echo "Current remote: $CURRENT_REMOTE"
echo

# Show options
cat << 'OPTIONS'
Choose deployment target:

1. GitHub (Recommended)
   - Creates private repository
   - Enables GitHub Actions
   - Full cloud accessibility

2. Cloudflare Pages
   - Static site hosting
   - Git integration
   - Fast edge network

3. Local Bare Repo (Current)
   - /tmp/diamondnode-command-center.git
   - Development/testing

4. SSH Server
   - Direct SSH access
   - Manual setup required

Enter choice [1-4]: 
OPTIONS

read -r choice
echo

case "$choice" in
    1)
        echo "=== GitHub Deployment ==="
        echo
        echo "Step 1: Create repository on GitHub"
        echo "  - Go to https://github.com/new"
        echo "  - Name: diamondnode-command-center"
        echo "  - Visibility: Private (recommended)"
        echo "  - Do NOT initialize with README"
        echo
        read -p "Press Enter after creating repository..." -r
        
        REPO_URL="git@github.com:$(whoami)/diamondnode-command-center.git"
        git remote add github "$REPO_URL"
        git push github master
        
        echo
        echo "✓ Pushed to GitHub"
        echo "✓ GitHub Actions workflow will run automatically"
        echo
        echo "Next steps:"
        echo "  1. Go to GitHub repository Settings"
        echo "  2. Add secrets:"
        echo "     - NOTION_TOKEN: $(grep NOTION_TOKEN ~/.vibe/.env | cut -d= -f2 | tr -d \"')"
        echo "     - GC_MCP_INGRESS_AUTH: your-mcp-auth-token"
        echo "     - GC_NOTION_BRIDGE_AUTH: your-bridge-auth-token"
        echo "     - OPENCLAW_AUTH_TOKEN: your-openclaw-token"
        echo
        ;;
    2)
        echo "=== Cloudflare Pages Deployment ==="
        echo
        echo "Step 1: Install wrangler"
        echo "  npm install -g wrangler"
        echo
        echo "Step 2: Login to Cloudflare"
        echo "  wrangler login"
        echo
        echo "Step 3: Create Pages project"
        echo "  - Go to https://dash.cloudflare.com/"
        echo "  - Create new Pages project"
        echo "  - Connect to GitHub repository"
        echo "  - Build command: echo 'DiamondNode Command Center'"
        echo "  - Build output: ."
        echo
        echo "Step 4: Add environment variables in Cloudflare dashboard:"
        echo "  - NOTION_TOKEN"
        echo "  - GC_MCP_URL"
        echo "  - GC_NOTION_BRIDGE_AUTH"
        echo
        ;;
    3)
        echo "=== Local Bare Repo (Already Deployed) ==="
        echo
        echo "Current remote: $CURRENT_REMOTE"
        echo
        echo "To clone locally:"
        echo "  git clone /tmp/diamondnode-command-center.git"
        echo
        ;;
    4)
        echo "=== SSH Server Deployment ==="
        echo
        read -p "Enter SSH server address (user@host): " ssh_server
        read -p "Enter SSH port [22]: " ssh_port
        ssh_port=${ssh_port:-22}
        
        SSH_REPO="ssh://${ssh_server}:${ssh_port}/home/diamondnode"
        git remote add ssh "$SSH_REPO"
        git push ssh master
        
        echo
        echo "✓ Pushed to SSH server"
        echo
        echo "To clone:"
        echo "  git clone ${SSH_REPO} diamondnode-command-center"
        echo
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

echo "=========================================="
echo "Deployment Complete!"
echo "=========================================="
