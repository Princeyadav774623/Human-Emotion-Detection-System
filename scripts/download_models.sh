#!/bin/bash
# ============================================================
#  download_models.sh — Run on LOGIN NODE (has internet)
#  Downloads Llama-3.2-3B + r3d_18 weights for offline training
#
#  Prerequisites:
#    1. Accept license at https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct
#    2. Create HF token at https://huggingface.co/settings/tokens
#    3. export HF_TOKEN=hf_xxxxxxxxxxxxx   (before running this)
#
#  Usage: bash scripts/download_models.sh
# ============================================================
set -euo pipefail

PROJECT_DIR="$HOME/meld_emotion"
LLAMA_DIR="$PROJECT_DIR/models/llama3_2_3b"
R3D_LOCAL="$PROJECT_DIR/models/r3d_18-b3b3357e.pth"
R3D_CACHE="$HOME/.cache/torch/hub/checkpoints/r3d_18-b3b3357e.pth"

cd "$PROJECT_DIR"
source venv/bin/activate

if [ -z "${HF_TOKEN:-}" ]; then
    echo "ERROR: HF_TOKEN not set."
    echo "  1. Accept license: https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct"
    echo "  2. Create token:   https://huggingface.co/settings/tokens"
    echo "  3. export HF_TOKEN=hf_xxx"
    exit 1
fi

echo "================================================"
echo "  Downloading models for offline use"
echo "================================================"

# ---------- 1. Llama-3.2-3B-Instruct ----------
echo ""
echo "[1/2] Downloading Llama-3.2-3B-Instruct to $LLAMA_DIR ..."
mkdir -p "$LLAMA_DIR"

python3 - <<PY
from transformers import AutoTokenizer, AutoModelForCausalLM
import os, sys

model_name = "meta-llama/Llama-3.2-3B-Instruct"
save_dir   = "$LLAMA_DIR"
token      = os.environ["HF_TOKEN"]

print("  Tokenizer...")
tok = AutoTokenizer.from_pretrained(model_name, token=token)
tok.save_pretrained(save_dir)

print("  Model weights (~6 GB, takes a few minutes)...")
model = AutoModelForCausalLM.from_pretrained(model_name, token=token)
model.save_pretrained(save_dir, safe_serialization=True)

# sanity check
required = ["config.json", "tokenizer_config.json"]
missing  = [f for f in required if not os.path.exists(os.path.join(save_dir, f))]
if missing:
    print(f"  ERROR: missing files: {missing}", file=sys.stderr); sys.exit(1)

has_weights = any(f.endswith((".safetensors", ".bin"))
                  for f in os.listdir(save_dir))
if not has_weights:
    print("  ERROR: no weight file saved", file=sys.stderr); sys.exit(1)

print(f"  OK: {save_dir}")
PY

# ---------- 2. r3d_18 ----------
echo ""
echo "[2/2] Downloading r3d_18 (Kinetics-400) ..."
mkdir -p "$(dirname "$R3D_LOCAL")"

python3 - <<PY
from torchvision.models.video import r3d_18, R3D_18_Weights
_ = r3d_18(weights=R3D_18_Weights.KINETICS400_V1)
print("  downloaded to torch hub cache")
PY

if [ -f "$R3D_CACHE" ]; then
    cp -u "$R3D_CACHE" "$R3D_LOCAL"
    size=$(du -h "$R3D_LOCAL" | cut -f1)
    echo "  OK: $R3D_LOCAL ($size)"
else
    echo "  ERROR: $R3D_CACHE not found" >&2
    exit 1
fi

echo ""
echo "================================================"
echo "  Done."
echo "  Llama-3.2-3B : $LLAMA_DIR"
echo "  r3d_18       : $R3D_LOCAL"
echo ""
echo "  Now: qsub scripts/train.pbs"
echo "================================================"
