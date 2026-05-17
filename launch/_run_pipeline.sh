#!/bin/bash
# ============================================================
#  _run_pipeline.sh — Inner runner. Called by submit.pbs and
#  interactive.sh. Don't run directly unless you've already
#  set up your environment.
# ============================================================
set -euo pipefail

PROJECT_DIR="${MELD_PROJECT:-$HOME/meld_emotion}"
cd "$PROJECT_DIR"

# venv
if [ ! -d venv ]; then
    echo "ERROR: venv/ missing. Run from login node:"
    echo "  python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
    exit 2
fi
source venv/bin/activate

# offline mode
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0
export TORCH_HOME="$PROJECT_DIR/.torch_cache"
mkdir -p "$TORCH_HOME/hub/checkpoints" logs
if [ -f "$PROJECT_DIR/models/r3d_18-b3b3357e.pth" ]; then
    cp -u "$PROJECT_DIR/models/r3d_18-b3b3357e.pth" \
          "$TORCH_HOME/hub/checkpoints/" 2>/dev/null || true
fi

ts=$(date +%Y%m%d_%H%M%S)
LOG="logs/pipeline_${ts}.log"
exec > >(tee -a "$LOG") 2>&1

echo "============================================================"
echo "  $(date)  host=$(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv | head -2 || echo "no GPU"
echo "  log: $LOG"
echo "============================================================"

bash pipeline/run_all.sh

echo "============================================================"
echo "  COMPLETE: $(date)"
echo "============================================================"
