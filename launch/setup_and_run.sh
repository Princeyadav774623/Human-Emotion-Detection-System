#!/bin/bash
# ============================================================
#  setup_and_run.sh — One-shot first-time setup + launch.
#
#  For when you've just SSH'd in to a fresh machine and want
#  to install + verify + launch in a single command.
#
#  This is the SIMPLE option. It does NOT survive SSH disconnect.
#  Use submit.pbs (background) or interactive.sh (tmux) for long
#  unattended runs.
#
#  Usage:
#      ssh user@cluster
#      cd ~/meld_emotion       # (after git clone or rsync)
#      bash launch/setup_and_run.sh
# ============================================================
set -euo pipefail

PROJECT_DIR="${MELD_PROJECT:-$HOME/meld_emotion}"
cd "$PROJECT_DIR"

echo "============================================================"
echo "  MELD pipeline — setup + run"
echo "  project: $PROJECT_DIR"
echo "============================================================"

# ---------- 1. python venv ----------
if [ ! -d venv ]; then
    echo
    echo "[1/5] creating venv..."
    python3 -m venv venv
    source venv/bin/activate
    pip install -U pip wheel
    pip install -r requirements.txt
else
    echo
    echo "[1/5] venv exists; activating"
    source venv/bin/activate
fi

# ---------- 2. check models are downloaded ----------
echo
echo "[2/5] checking models..."
need_dl=0
if [ ! -f "models/llama3_2_3b/config.json" ]; then
    echo "  MISSING: Llama-3.2-3B at models/llama3_2_3b/"
    need_dl=1
fi
if [ ! -f "models/r3d_18-b3b3357e.pth" ]; then
    echo "  MISSING: R3D-18 weights at models/r3d_18-b3b3357e.pth"
    need_dl=1
fi
if [ "$need_dl" -eq 1 ]; then
    echo
    if [ -z "${HF_TOKEN:-}" ]; then
        echo "  Set HF_TOKEN and run:  bash scripts/download_models.sh"
        echo "  (must be done on a login node with internet)"
        exit 2
    fi
    echo "  downloading now (this needs internet)..."
    bash scripts/download_models.sh
fi

# ---------- 3. check manifests ----------
echo
echo "[3/5] checking data manifests..."
if [ ! -f "data/train.json" ] || [ ! -f "data/val.json" ]; then
    echo "  MISSING: data/train.json or data/val.json"
    echo "  Generate them before running. Each must be a JSON list of"
    echo "  {video_path, utterance, emotion} objects with MELD-native"
    echo "  labels (neutral/joy/anger/surprise/sadness/fear/disgust)."
    exit 2
fi
echo "  train: $(python3 -c "import json; print(len(json.load(open('data/train.json'))))") items"
echo "  val:   $(python3 -c "import json; print(len(json.load(open('data/val.json'))))") items"

# ---------- 4. GPU check ----------
echo
echo "[4/5] GPU check..."
if ! nvidia-smi >/dev/null 2>&1; then
    echo "  no GPU on this machine."
    echo "  options:"
    echo "    qsub launch/submit.pbs        # background, recommended"
    echo "    bash launch/interactive.sh    # request a GPU node first"
    exit 2
fi
nvidia-smi --query-gpu=name,memory.total --format=csv | head -2

# ---------- 5. launch ----------
echo
echo "[5/5] launching pipeline..."
echo "  This will take ~24-30 hours total."
echo "  WARNING: if your SSH drops, the run dies. Use submit.pbs or"
echo "           interactive.sh for unattended runs."
echo
read -p "  continue? [y/N] " ans
if [[ "$ans" != "y" && "$ans" != "Y" ]]; then
    echo "  aborted by user"; exit 0
fi

bash launch/_run_pipeline.sh
