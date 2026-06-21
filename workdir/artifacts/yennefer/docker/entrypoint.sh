#!/usr/bin/env bash
set -euo pipefail
mkdir -p "${STORAGE_PATH:-/app/storage}"
exec "$@"
