#!/usr/bin/env bash
# memory-offload — move cold state off local disk to gdrive:, verified before delete.
#
# Why this exists, and why it is not sync-to-gdrive.sh:
#   * workdir/artifacts/sync-to-gdrive.sh rsyncs into /mnt/gdrive_memory, which is
#     NOT a mountpoint. Every "offload" it ever did wrote to the local disk and made
#     the pressure worse. 26 files are sitting there right now believing they are on
#     Drive. This script never touches that path except to drain it.
#   * gdrive-sync.service is a *mirror* (rclone sync of three live state dirs). A
#     mirror frees nothing locally. This is the other half: copy → verify → prune.
#
# Safety model — the three rules that must not be relaxed:
#   1. NOTHING is deleted locally until `rclone check` confirms every file landed.
#   2. Secrets never leave the box. A hard exclude list is applied to every transfer
#      AND a pre-flight scanner aborts the run if a staged path still looks secret.
#      Drive is an external service; publishing a key there is unrecoverable.
#   3. Pruning is opt-in per invocation (--prune). Default is copy-only.
#
# Usage:
#   memory-offload.sh plan                 # what would move, what it would reclaim
#   memory-offload.sh offload <category>   # copy + verify (add --prune to reclaim)
#   memory-offload.sh caches               # purge regenerable caches (no Drive round-trip)
#   memory-offload.sh status               # local pressure + remote quota
#
# Categories are declared in CATEGORIES below; `all` runs every archival one.

set -uo pipefail

REMOTE="${MEMORY_OFFLOAD_REMOTE:-gdrive:}"
BASE="${MEMORY_OFFLOAD_BASE:-DiamondNode/offload/$(hostname -s)}"
LOG="${HOME}/logs/memory-offload.log"
mkdir -p "$(dirname "$LOG")"

log()  { printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$LOG"; }
die()  { log "FATAL: $*"; exit 1; }

command -v rclone >/dev/null 2>&1 || die "rclone not on PATH"
rclone listremotes 2>/dev/null | grep -qx "${REMOTE}" || die "rclone remote '${REMOTE}' not configured"

# ── Secrets exclusion ────────────────────────────────────────────────────────
# Applied to every transfer. Deny-by-pattern rather than allow-by-path because the
# home directory mixes credential vaults, per-tool .env files and session dumps in
# among the data we actually want archived — the same reason .gitignore here is
# deny-by-default. Erring toward excluding too much is free; the other error is not.
SECRET_EXCLUDES=(
  --exclude "**/.env"            --exclude "**/.env.*"        --exclude "**/*.env"
  --exclude ".vault/**"          --exclude "**/secrets*"      --exclude "**/*secret*"
  --exclude "**/ralph.env"       --exclude "**/financial-ops.env"
  --exclude "**/*.key"           --exclude "**/*.pem"         --exclude "**/*.p12"
  --exclude "**/*.pfx"           --exclude "**/*.keystore"    --exclude "**/*.jks"
  --exclude "**/id_rsa*"         --exclude "**/id_ed25519*"   --exclude "**/id_ecdsa*"
  --exclude ".ssh/**"            --exclude ".gnupg/**"        --exclude ".aws/**"
  --exclude ".config/gcloud/**"  --exclude ".config/agent-wallet/**"
  --exclude ".kube/**"           --exclude ".docker/config.json"
  --exclude "**/*mnemonic*"      --exclude "**/*seed-phrase*" --exclude "**/*privatekey*"
  --exclude "**/*private_key*"   --exclude "**/*private-key*"
  --exclude "**/credentials*"    --exclude "**/*token*.json"  --exclude "**/*.netrc"
  --exclude "**/settings.local.json"
  --exclude "**/node_modules/**" --exclude "**/__pycache__/**" --exclude "**/*.pyc"
  --exclude "**/.git/**"
)

# Belt-and-braces: even with the excludes above, refuse to transfer if a staged path
# still reads as credential-bearing. An exclude pattern that silently stops matching
# (a renamed file, a new tool's layout) would otherwise ship a key to Drive.
SECRET_PATH_RE='(^|/)(\.env|\.vault|\.ssh|\.gnupg|\.aws|agent-wallet)(/|$)|\.(key|pem|p12|pfx|jks|keystore|netrc)$|(mnemonic|seed.?phrase|private.?key|secret|credential)'

preflight_secret_scan() {
  local src="$1" hits
  hits=$(rclone lsf "$src" --recursive --files-only "${SECRET_EXCLUDES[@]}" 2>/dev/null \
          | grep -Ei "$SECRET_PATH_RE" | head -20)
  if [ -n "$hits" ]; then
    log "ABORT: staged file list still contains credential-shaped paths:"
    printf '  %s\n' $hits | tee -a "$LOG"
    die "refusing to transfer to an external service; tighten SECRET_EXCLUDES first"
  fi
}

# Path-shaped exclusion is necessary but NOT sufficient, and assuming otherwise is
# how the first run of this script shipped 40 MiB of agent session transcripts to
# Drive. A transcript is named `<uuid>.jsonl` — no pattern above matches it — but its
# body is a verbatim record of every command run and every value printed, which for
# an ops box means live API keys and private keys in tool output. Grep the CONTENT
# of what is about to leave the machine, and abort on any hit.
SECRET_CONTENT_RE='(sk-ant-[A-Za-z0-9_-]{16,}|sk-[A-Za-z0-9]{32,}|ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|AIza[A-Za-z0-9_-]{30,}|xox[baprs]-[A-Za-z0-9-]{10,}|BEGIN [A-Z ]*PRIVATE KEY|eyJ[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{20,}|\b[0-9a-fA-F]{64}\b)'

preflight_content_scan() {
  local src="$1" flagged=0 f
  # Filenames and counts only — never echo a match, or the scan itself leaks.
  while IFS= read -r f; do
    case "$f" in *.png|*.jpg|*.jpeg|*.webp|*.gif|*.pdf|*.zip|*.gz|*.woff*|*.ttf) continue;; esac
    [ -f "$src/$f" ] || continue
    if LC_ALL=C grep -qlE "$SECRET_CONTENT_RE" "$src/$f" 2>/dev/null; then
      log "  credential-shaped content in: $f"
      flagged=$((flagged + 1))
      [ "$flagged" -ge 10 ] && { log "  … (further hits suppressed)"; break; }
    fi
  done < <(rclone lsf "$src" --recursive --files-only "${SECRET_EXCLUDES[@]}" 2>/dev/null)
  if [ "$flagged" -gt 0 ]; then
    die "$flagged file(s) carry credential-shaped content; refusing to send to ${REMOTE}"
  fi
}

# ── Categories: label | local path | prune-safe ──────────────────────────────
# prune-safe=yes means the local copy is genuinely redundant once verified on Drive.
# Anything a running service reads is marked no and is archived, never pruned.
#
# DELIBERATELY ABSENT — do not re-add without a redaction pass that runs first:
#   ~/.claude/projects                        agent session transcripts
#   ~/.gemini/antigravity-cli/conversations   agent session transcripts
#   ~/logs                                    service logs; echo secrets on failure
# These are the largest cold directories on the box and therefore the most tempting
# offload targets. They are also verbatim records of credential-bearing tool output.
# Archiving them to a third-party service publishes those credentials.
CATEGORIES=(
  "stranded-mount|/mnt/gdrive_memory|yes"
  "workdir-artifacts|${HOME}/workdir/artifacts|no"
  "publish|${HOME}/publish|no"
)

cat_path()  { printf '%s\n' "${CATEGORIES[@]}" | awk -F'|' -v k="$1" '$1==k{print $2}'; }
cat_prune() { printf '%s\n' "${CATEGORIES[@]}" | awk -F'|' -v k="$1" '$1==k{print $3}'; }

human() { numfmt --to=iec --suffix=B "${1:-0}" 2>/dev/null || echo "${1}B"; }

do_plan() {
  log "=== offload plan (nothing is moved) ==="
  printf '  %-22s %10s  %-11s %s\n' CATEGORY SIZE PRUNE-SAFE PATH
  local total=0
  for entry in "${CATEGORIES[@]}"; do
    IFS='|' read -r name path prune <<<"$entry"
    [ -e "$path" ] || { printf '  %-22s %10s  %-11s %s\n' "$name" "-" "$prune" "$path (absent)"; continue; }
    local b; b=$(du -sb "$path" 2>/dev/null | cut -f1); b=${b:-0}
    [ "$prune" = yes ] && total=$((total + b))
    printf '  %-22s %10s  %-11s %s\n' "$name" "$(human "$b")" "$prune" "$path"
  done
  log "reclaimable by --prune once verified: $(human "$total")"
  echo
  log "=== NOT handled here (needs an explicit human decision) ==="
  for d in .ollama .xinference .lmstudio .litert-lm; do
    [ -d "${HOME}/$d" ] && printf '  %10s  ~/%s  (re-downloadable model weights)\n' \
      "$(du -sh "${HOME}/$d" 2>/dev/null | cut -f1)" "$d"
  done
  local nm; nm=$(du -sch $(find "$HOME" -maxdepth 4 -type d -name node_modules -prune 2>/dev/null) 2>/dev/null | tail -1 | cut -f1)
  printf '  %10s  node_modules across ~ (regenerable via npm install)\n' "${nm:-?}"
}

do_offload() {
  local name="$1" prune="${2:-}"
  local path; path=$(cat_path "$name")
  [ -n "$path" ] || die "unknown category: $name"
  [ -e "$path" ] || { log "skip $name — $path does not exist"; return 0; }
  [ "$prune" = "--prune" ] && [ "$(cat_prune "$name")" != yes ] && \
    die "$name is not prune-safe; archive it or reclassify it deliberately"

  local dest="${REMOTE}${BASE}/${name}"
  log "offload $name: $path -> $dest"
  preflight_secret_scan "$path"
  preflight_content_scan "$path"

  rclone copy "$path" "$dest" \
    "${SECRET_EXCLUDES[@]}" \
    --transfers 4 --fast-list --local-no-check-updated --stats-one-line \
    2>&1 | tee -a "$LOG"

  # Verify before any delete. --one-way: extra files already on Drive are fine.
  log "verifying $name…"
  if ! rclone check "$path" "$dest" "${SECRET_EXCLUDES[@]}" --one-way 2>&1 | tee -a "$LOG"; then
    log "VERIFY FAILED for $name — local copy left untouched"
    return 1
  fi
  log "verified $name"

  if [ "$prune" = "--prune" ]; then
    local before; before=$(du -sb "$path" 2>/dev/null | cut -f1)
    find "$path" -mindepth 1 -delete 2>/dev/null
    log "pruned $name — reclaimed $(human "${before:-0}")"
  fi
}

do_caches() {
  # Regenerable by design: these are package-manager caches, not data. Each tool's
  # own cache-clean is used where one exists so the tool's index stays consistent.
  log "=== purging regenerable caches ==="
  local before; before=$(df -B1 --output=avail "$HOME" | tail -1)
  command -v npm  >/dev/null 2>&1 && { npm cache clean --force >/dev/null 2>&1 && log "  npm cache cleaned"; }
  command -v pip  >/dev/null 2>&1 && { pip cache purge      >/dev/null 2>&1 && log "  pip cache purged"; }
  command -v uv   >/dev/null 2>&1 && { uv cache clean       >/dev/null 2>&1 && log "  uv cache cleaned"; }
  command -v go   >/dev/null 2>&1 && { go clean -modcache   >/dev/null 2>&1 && log "  go modcache cleaned"; }
  command -v bun  >/dev/null 2>&1 && { rm -rf "${HOME}/.bun/install/cache" 2>/dev/null && log "  bun cache cleaned"; }
  journalctl --user --vacuum-size=64M >/dev/null 2>&1 && log "  user journal vacuumed to 64M"
  local after; after=$(df -B1 --output=avail "$HOME" | tail -1)
  log "reclaimed on \$HOME: $(human $(( after - before )))"
}

do_status() {
  log "=== local ==="; df -h / "$HOME" | grep -v tmpfs | sed 's/^/  /'
  log "=== remote (${REMOTE}) ==="; rclone about "$REMOTE" 2>&1 | sed 's/^/  /'
  log "=== offloaded so far ==="; rclone size "${REMOTE}${BASE}" 2>&1 | sed 's/^/  /'
}

case "${1:-plan}" in
  plan)    do_plan ;;
  status)  do_status ;;
  caches)  do_caches ;;
  offload)
    [ $# -ge 2 ] || die "usage: $0 offload <category|all> [--prune]"
    if [ "$2" = all ]; then
      for entry in "${CATEGORIES[@]}"; do do_offload "${entry%%|*}" "${3:-}"; done
    else
      do_offload "$2" "${3:-}"
    fi ;;
  *) die "usage: $0 {plan|offload <category> [--prune]|caches|status}" ;;
esac
