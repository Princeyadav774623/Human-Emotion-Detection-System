#!/bin/bash
# ============================================================
#  interactive.sh — Run the pipeline on a GPU node with live output.
#
#  This script:
#    1. Requests an interactive compute node (if you're not on one)
#    2. Starts a tmux session — survives SSH disconnects
#    3. Runs the pipeline, streaming logs to your terminal
#
#  Usage:
#      ssh user@cluster
#      cd ~/meld_emotion
#      bash launch/interactive.sh
#
#  If your SSH drops, reconnect and reattach:
#      ssh user@cluster
#      tmux attach -t meld
#
#  Detach without killing: Ctrl-b then d
#  Kill the run:           Ctrl-c (twice if needed)
# ============================================================
set -euo pipefail

PROJECT_DIR="${MELD_PROJECT:-$HOME/meld_emotion}"
SESSION="meld"
WALLTIME="${WALLTIME:-12:00:00}"   # change if you want longer

# --- check we're on a GPU node ---
if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi >/dev/null 2>&1; then
    echo "No GPU detected. Requesting an interactive compute node..."
    echo "  walltime=$WALLTIME  (override with: WALLTIME=24:00:00 bash $0)"
    # PBS interactive job — change qsub flags if your cluster differs
    exec qsub -I \
              -l select=1:ncpus=16:ngpus=1:mem=96gb \
              -l walltime=$WALLTIME \
              -- bash -lc "cd $PROJECT_DIR && bash launch/interactive.sh"
fi

# --- we have a GPU; set up tmux ---
if ! command -v tmux >/dev/null 2>&1; then
    echo "tmux not installed. Falling back to nohup (won't survive SSH drop reliably)."
    cd "$PROJECT_DIR"
    nohup bash launch/_run_pipeline.sh > "logs/interactive_$(date +%Y%m%d_%H%M%S).log" 2>&1 &
    echo "PID=$! — tail logs/interactive_*.log to watch"
    exit 0
fi

cd "$PROJECT_DIR"

# tmux session already running?
if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "Existing tmux session '$SESSION' found. Attaching..."
    exec tmux attach -t "$SESSION"
fi

# fresh tmux session
echo "Starting tmux session '$SESSION'..."
tmux new-session -d -s "$SESSION" \
    "bash $PROJECT_DIR/launch/_run_pipeline.sh"

echo
echo "Pipeline started in tmux session '$SESSION'."
echo "  Attach with:    tmux attach -t $SESSION"
echo "  Detach:         Ctrl-b then d"
echo "  Kill:           tmux kill-session -t $SESSION"
echo
echo "Attaching now (Ctrl-b d to detach without killing)..."
sleep 1
exec tmux attach -t "$SESSION"
