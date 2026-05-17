"""
common.py — Shared config, paths, and utilities for the pipeline.

Every stage imports from here so paths and class definitions stay in sync.
"""
import os
import json
import time
import logging
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List

# ============================================================
# Offline mode (set BEFORE importing transformers anywhere)
# ============================================================
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ============================================================
# Paths — change PROJECT_DIR if your install lives elsewhere
# ============================================================
PROJECT_DIR = Path(os.environ.get("MELD_PROJECT",
                                  os.path.expanduser("~/meld_emotion")))
DATA_DIR    = PROJECT_DIR / "data"
MODELS_DIR  = PROJECT_DIR / "models"
RUNS_DIR    = PROJECT_DIR / "runs"
RUNS_DIR.mkdir(parents=True, exist_ok=True)

LLAMA_DIR   = MODELS_DIR / "llama3_2_3b"
R3D_WEIGHTS = MODELS_DIR / "r3d_18-b3b3357e.pth"

# ============================================================
# Labels — MELD native 7-class
# ============================================================
CLASSES  = ["neutral", "joy", "anger", "surprise", "sadness", "fear", "disgust"]
LABEL2ID = {c: i for i, c in enumerate(CLASSES)}
ID2LABEL = {i: c for c, i in LABEL2ID.items()}
NUM_CLASSES = len(CLASSES)

# Common aliases that might show up in messy manifests.
LABEL_ALIASES = {
    "happy":     "joy",
    "happiness": "joy",
    "angry":     "anger",
    "sad":       "sadness",
    "scared":    "fear",
    "disgusted": "disgust",
    "surprised": "surprise",
    "neu":       "neutral",
}

def normalize_label(s: str) -> str:
    s = (s or "").strip().lower()
    return LABEL_ALIASES.get(s, s)

# ============================================================
# Default training config — tuner overrides these
# ============================================================
@dataclass
class Config:
    # paths
    train_manifest: str = str(DATA_DIR / "train.json")
    val_manifest:   str = str(DATA_DIR / "val.json")
    text_model:     str = str(LLAMA_DIR)
    r3d_weights:    str = str(R3D_WEIGHTS)

    # data
    classes: List[str] = field(default_factory=lambda: CLASSES.copy())
    num_frames:  int   = 16
    frame_size:  int   = 112
    max_text_len: int  = 64
    video_read_timeout: int = 10  # seconds per video read

    # training
    epochs:      int   = 25
    batch_size:  int   = 16
    workers:     int   = 8
    grad_accum:  int   = 2
    warmup_epochs: int = 1
    patience:    int   = 5

    # optimization
    lr_lora:     float = 2e-4
    lr_video:    float = 1e-5
    lr_head:     float = 3e-4
    weight_decay: float = 0.05
    grad_clip:   float = 1.0
    label_smoothing: float = 0.05
    focal_gamma: float = 2.0

    # LoRA
    lora_r:      int   = 16
    lora_alpha:  int   = 32
    lora_dropout: float = 0.05

    # fusion
    fusion_dim:  int   = 512
    n_heads:     int   = 8
    dropout:     float = 0.3

    # misc
    seed:        int   = 42
    compile_submodules: bool = True

    # diagnostics — these auto-trip and stop bad runs
    collapse_entropy_threshold: float = 0.30  # below this = collapsed
    collapse_check_epoch: int = 2             # check after this epoch

    def update(self, **kw):
        for k, v in kw.items():
            if hasattr(self, k):
                setattr(self, k, v)
        return self

    def to_dict(self):
        return {k: v for k, v in asdict(self).items()}

# ============================================================
# Run directory + logging
# ============================================================
def make_run_dir(name: str = None) -> Path:
    ts  = time.strftime("%Y%m%d_%H%M%S")
    rd  = RUNS_DIR / (f"{name}_{ts}" if name else ts)
    for sub in ("manifests", "tune", "checkpoints", "deploy", "diagnostics"):
        (rd / sub).mkdir(parents=True, exist_ok=True)
    return rd

def setup_logging(run_dir: Path, name: str = "pipeline"):
    log = logging.getLogger(name)
    log.handlers.clear()
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                            datefmt="%H:%M:%S")
    fh = logging.FileHandler(run_dir / f"{name}.log")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    log.addHandler(fh); log.addHandler(sh)
    log.propagate = False
    return log

def latest_run_dir() -> Path:
    runs = sorted([d for d in RUNS_DIR.iterdir() if d.is_dir()],
                  key=lambda p: p.stat().st_mtime, reverse=True)
    if not runs:
        raise FileNotFoundError(f"no runs found under {RUNS_DIR}")
    return runs[0]

def save_json(obj, path: Path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)

def load_json(path: Path):
    with open(path) as f:
        return json.load(f)
