#!/bin/bash
# ============================================================
#  run_all.sh — Run the full pipeline end-to-end.
#  Each stage is independent and saves its state, so if any stage
#  crashes you can rerun just that stage with its --run-dir flag.
# ============================================================
set -euo pipefail

PROJECT_DIR="${MELD_PROJECT:-$HOME/meld_emotion}"
cd "$PROJECT_DIR"
source venv/bin/activate

# --- offline (compute nodes have no internet) ---
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0

# torch hub cache (for r3d_18 weights backup)
export TORCH_HOME="$PROJECT_DIR/.torch_cache"
mkdir -p "$TORCH_HOME/hub/checkpoints"
[ -f "$PROJECT_DIR/models/r3d_18-b3b3357e.pth" ] && \
  cp -u "$PROJECT_DIR/models/r3d_18-b3b3357e.pth" \
        "$TORCH_HOME/hub/checkpoints/" 2>/dev/null || true

mkdir -p logs
ts=$(date +%Y%m%d_%H%M%S)
LOG="logs/pipeline_${ts}.log"
exec > >(tee -a "$LOG") 2>&1

echo "============================================================"
echo "  MELD emotion pipeline"
echo "  $(date)  $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv | head -2
echo "============================================================"

# ---------- stage 0: pre-flight ----------
echo
echo "============ STAGE 0: PRE-FLIGHT ============"
RUN_DIR=$(python pipeline/00_preflight.py --timeout 5 | tail -1)
echo "RUN_DIR=$RUN_DIR"

# ---------- stage 1: diagnostics ----------
echo
echo "============ STAGE 1: DIAGNOSTICS ============"
if ! python pipeline/01_diagnostics.py --run-dir "$RUN_DIR"; then
    echo "DIAGNOSTICS FAILED — see $RUN_DIR/diagnostics.log"
    echo "  Fix the reported issues and re-run from this stage:"
    echo "    python pipeline/01_diagnostics.py --run-dir $RUN_DIR"
    exit 1
fi

# ---------- stage 2: tuning ----------
echo
echo "============ STAGE 2: TUNING ============"
python pipeline/02_tune.py --run-dir "$RUN_DIR" --n-trials 25 \
       --study-name "meld_$(basename $RUN_DIR)"

# ---------- stage 3: full training ----------
echo
echo "============ STAGE 3: FULL TRAINING ============"
python pipeline/03_train_full.py --run-dir "$RUN_DIR"

# ---------- stage 4: export ----------
echo
echo "============ STAGE 4: EXPORT ============"
python pipeline/04_export.py --run-dir "$RUN_DIR"

echo
echo "============================================================"
echo "  DONE: $(date)"
echo "  artifacts: $RUN_DIR"
echo "  deploy bundle: $RUN_DIR/deploy/"
echo "============================================================"
