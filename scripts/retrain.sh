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
# Read current iter from checkpoint to compute absolute max_iters
CURRENT_ITER=$(python -c "
import torch, os
ckpt = os.path.join('out-pelagic', 'ckpt.pt')
if os.path.exists(ckpt):
    d = torch.load(ckpt, map_location='cpu', weights_only=False)
    print(d.get('iter_num', 0))
else:
    print(0)
" 2>/dev/null || echo 0)
TARGET_ITERS=$((CURRENT_ITER + 50))
echo "Step 2: Training (resume from iter $CURRENT_ITER, target $TARGET_ITERS)..."
python train.py config/finetune_pelagic.py \
    --init_from=resume \
    --max_iters=$TARGET_ITERS

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
