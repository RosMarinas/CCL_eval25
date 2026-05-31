#!/bin/bash
# Run B8-v2 and BC8-v2 full evals in parallel on two GPUs, output to terminal.
# Usage: bash run_full_evals.sh
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ---- cleanup ----
echo "[cleanup] Removing stale output files ..."
rm -f checkpoints/B8-v2-lr5e5-ep2/full_eval.json
rm -f checkpoints/BC8-v2-lr5e5-ep2/full_eval.json
pkill -f "python.*full_eval" 2>/dev/null || true
sleep 1

export PYTHONPATH=.
export HF_ENDPOINT=https://hf-mirror.com

echo ""
echo "=== Launching B8-v2 on cuda:0 + BC8-v2 on cuda:1 ==="
echo ""

# Run both in background, tag each output line, wait for both
uv run python tests/full_eval.py \
  --checkpoint checkpoints/B8-v2-lr5e5-ep2/adapter \
  --device cuda:0 --name B8-v2 \
  --output checkpoints/B8-v2-lr5e5-ep2/full_eval.json \
  2>&1 | sed 's/^/[B8]  /' &
PID_B8=$!

uv run python tests/full_eval.py \
  --checkpoint checkpoints/BC8-v2-lr5e5-ep2/adapter \
  --device cuda:1 --name BC8-v2 \
  --output checkpoints/BC8-v2-lr5e5-ep2/full_eval.json \
  2>&1 | sed 's/^/[BC8] /' &
PID_BC8=$!

echo "PIDs: B8=$PID_B8  BC8=$PID_BC8"
echo ""

wait $PID_B8 $PID_BC8
echo ""
echo "=== Both evals complete ==="
