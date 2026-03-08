#!/bin/bash
# Weekly retrain script for PelagicGPT
# Rebuilds corpus with latest data and resumes training for 50 more iterations.
#
# Usage: ./scripts/retrain.sh
# Can be added to cron: 0 2 * * 0 cd /Users/Bala_1/dev/pelagic-gpt && ./scripts/retrain.sh

set -euo pipefail

cd "$(dirname "$0")/.."

echo "=== PelagicGPT Retrain $(date) ==="

# 1. Rebuild corpus with latest Supabase data + research
echo "Step 1: Preparing data..."
python data/pelagic/prepare.py

# 2. Resume training for 50 more iterations
echo "Step 2: Training (resume, 50 iters)..."
python train.py config/finetune_pelagic.py \
    --init_from=resume \
    --max_iters=50

# 3. Restart serve.py if running
echo "Step 3: Restarting server..."
SERVE_PID=$(pgrep -f "uvicorn serve:app" || true)
if [ -n "$SERVE_PID" ]; then
    kill "$SERVE_PID"
    sleep 2
    nohup uvicorn serve:app --port 8000 > /dev/null 2>&1 &
    echo "Server restarted (PID: $!)"
else
    echo "Server not running. Start with: uvicorn serve:app --port 8000"
fi

echo "=== Retrain complete ==="
