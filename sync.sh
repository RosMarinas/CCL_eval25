#!/bin/bash
# ============================================================
# CONFIGURATION — Modify these before use
# ============================================================
LOCAL_DIR="/Users/polaris/Documents/Graduate/Course/High-Level_MachineLearning/CCL25-Eval/"
REMOTE_HOST="swh@101.6.42.125"
REMOTE_DIR="/home/swh/AI/Experiment/CCL25_eval/"

# Patterns to exclude from sync
EXCLUDES=(
    '.git' '.venv' '.DS_Store'
    'checkpoints/' 'logs/' '__pycache__/'
    '*.pyc' 'uv.lock' '.claude/' '.venv/' '.antigravitycli/' '.python-version'
)
# ============================================================

# Build --exclude flags (array avoids quote-in-string pitfalls)
EXCLUDE_ARGS=()
for pattern in "${EXCLUDES[@]}"; do
    EXCLUDE_ARGS+=(--exclude "$pattern")
done

echo "Syncing ${LOCAL_DIR} -> ${REMOTE_HOST}:${REMOTE_DIR}"
echo "Excluding: ${EXCLUDES[*]}"

# Initial sync before watching
echo "Running initial sync..."
rsync -avz "${EXCLUDE_ARGS[@]}" "$LOCAL_DIR" "$REMOTE_HOST:$REMOTE_DIR"
echo "Initial sync done. Watching for changes..."

fswatch -o "$LOCAL_DIR" | xargs -n1 -I{} rsync -avz \
    "${EXCLUDE_ARGS[@]}" \
    "$LOCAL_DIR" \
    "$REMOTE_HOST:$REMOTE_DIR"
