#!/bin/bash
set -euo pipefail

# ============================================================
# CONFIGURATION — Modify these before use
# ============================================================
REMOTE_HOST="swh@101.6.42.125"
REMOTE_DIR="/home/swh/AI/Experiment/CCL25_eval/"

# Sync the git worktree that contains this script. This avoids accidentally
# syncing the parent repository when worktrees live under .worktrees/.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if LOCAL_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null)"; then
    LOCAL_DIR="${LOCAL_ROOT}/"
    BRANCH="$(git -C "$LOCAL_DIR" rev-parse --abbrev-ref HEAD)"
    COMMIT="$(git -C "$LOCAL_DIR" rev-parse --short HEAD)"
else
    LOCAL_DIR="${SCRIPT_DIR}/"
    BRANCH="unknown"
    COMMIT="unknown"
fi

# Patterns to exclude from sync
EXCLUDES=(
    '.git' '.worktrees/' '.venv' '.DS_Store'
    'checkpoints/' 'logs/' 'wandb/'  '__pycache__/'
    '*.pyc'
)
# ============================================================

# Build --exclude flags (array avoids quote-in-string pitfalls)
EXCLUDE_ARGS=()
for pattern in "${EXCLUDES[@]}"; do
    EXCLUDE_ARGS+=(--exclude "$pattern")
done

echo "Syncing ${LOCAL_DIR} -> ${REMOTE_HOST}:${REMOTE_DIR}"
echo "Branch: ${BRANCH} (${COMMIT})"
echo "Excluding: ${EXCLUDES[*]}"

sync_once() {
    rsync -az --delete "${EXCLUDE_ARGS[@]}" "$LOCAL_DIR" "$REMOTE_HOST:$REMOTE_DIR"
}

# Initial sync before watching
echo "Running initial sync..."
sync_once
echo "Initial sync done. Watching for changes..."

fswatch -o "$LOCAL_DIR" | while read -r _; do
    sync_once
done
