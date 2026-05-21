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
    '.git' '.worktrees/' '.venv/' '.DS_Store'
    'checkpoints/' 'logs/' 'wandb/'  '__pycache__/'
    '*.pyc' 'uv.lock'
)

# Files that affect execution should be strict mirrors: removed locally means
# removed remotely. Everything else is upload-only so remote run artifacts are
# never deleted just because they do not exist locally yet.
STRICT_ROOT_FILES=(
    '*.py'
    'pyproject.toml'
    'sync.sh'
)
STRICT_SOURCE_DIRS=(
    'src/'
    'tests/'
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
echo "Upload-only sync: whole worktree"
echo "Strict mirror delete: root files (${STRICT_ROOT_FILES[*]}), dirs (${STRICT_SOURCE_DIRS[*]})"

sync_once() {
    rsync -az "${EXCLUDE_ARGS[@]}" "$LOCAL_DIR" "$REMOTE_HOST:$REMOTE_DIR"

    ROOT_FILTER_ARGS=()
    for pattern in "${STRICT_ROOT_FILES[@]}"; do
        ROOT_FILTER_ARGS+=(--include "/$pattern")
    done
    ROOT_FILTER_ARGS+=(--exclude "/*")
    rsync -az --delete "${ROOT_FILTER_ARGS[@]}" "$LOCAL_DIR" "$REMOTE_HOST:$REMOTE_DIR"

    for dir in "${STRICT_SOURCE_DIRS[@]}"; do
        rsync -az --delete \
            --include "*/" \
            --include "*.py" \
            --exclude "*" \
            "$LOCAL_DIR$dir" "$REMOTE_HOST:$REMOTE_DIR$dir"
    done
}

# Initial sync before watching
echo "Running initial sync..."
sync_once
echo "Initial sync done. Watching for changes..."

fswatch -o "$LOCAL_DIR" | while read -r _; do
    sync_once
done
