#!/usr/bin/env python3
"""
train_h100.py — Multimodal emotion recognition (MELD-style, 7 classes)

Fixes over the previous run:
  - Offline model loading (HF_HUB_OFFLINE, local paths)
  - Llama-3.2-3B-Instruct (4-bit) + LoRA instead of TinyLlama
  - WeightedRandomSampler to fix class imbalance collapse
  - Per-class focal loss with inverse-frequency alpha + label smoothing
  - Clean warmup + cosine LR schedule (per-step, not per-epoch)
  - Differential LRs for LoRA / backbone / head
  - Macro-F1 early stopping (not accuracy / not loss)
  - bf16 mixed precision on H100 (no GradScaler needed)
  - torch.compile on non-quantized submodules

ADAPTATION POINTS are marked with:  # >>> ADAPT
"""
# -------------------- offline env (must be before transformers import) --------
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import json
import math
import time
import random
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import LambdaLR
from torch.amp import autocast

from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from transformers import (
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torchvision.models.video import r3d_18
from torchvision import transforms

# decord is much faster than torchvision.io for video reading
try:
    from decord import VideoReader, cpu
    HAS_DECORD = True
except ImportError:
    HAS_DECORD = False
    import torchvision.io as tvio

# ============================================================
# Config
# ============================================================
@dataclass
class Config:
    # paths                                                          # >>> ADAPT
    project_dir:    str = os.path.expanduser("~/meld_emotion")
    train_manifest: str = os.path.expanduser("~/meld_emotion/data/train.json")
    val_manifest:   str = os.path.expanduser("~/meld_emotion/data/val.json")
    text_model:     str = os.path.expanduser("~/meld_emotion/models/llama3_2_3b")
    r3d_weights:    str = os.path.expanduser("~/meld_emotion/models/r3d_18-b3b3357e.pth")
    output_dir:     str = os.path.expanduser("~/meld_emotion/runs")

    # data
    classes: List[str] = field(default_factory=lambda: [
        "happy", "sad", "angry", "anxiety", "stress", "surprise", "neutral"
    ])
    num_frames:  int   = 16
    frame_size:  int   = 112
    max_text_len: int  = 64

    # training
    epochs:      int   = 20
    batch_size:  int   = 16          # reduced from 32 — 3B model is bigger
    workers:     int   = 8
    grad_accum:  int   = 2           # effective batch = 32
    warmup_epochs: int = 1
    patience:    int   = 4

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
    lora_targets: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj"
    ])

    # fusion
    fusion_dim:  int   = 512
    n_heads:     int   = 8
    dropout:     float = 0.3

    # reproducibility
    seed:        int   = 42
    compile_submodules: bool = True

cfg = Config()
NUM_CLASSES = len(cfg.classes)
LABEL2ID = {c: i for i, c in enumerate(cfg.classes)}
ID2LABEL = {i: c for c, i in LABEL2ID.items()}

# ============================================================
# Setup
# ============================================================
torch.manual_seed(cfg.seed); np.random.seed(cfg.seed); random.seed(cfg.seed)
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
run_id   = time.strftime("%Y%m%d_%H%M%S")
run_dir  = Path(cfg.output_dir) / run_id
run_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(run_dir / "train.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    log.info(f"[GPU] {p.name} | {p.total_memory/1e9:.1f}GB")
else:
    log.warning("CUDA unavailable — running on CPU will be infeasibly slow")

# ============================================================
# Dataset
# ============================================================
# >>> ADAPT: manifest format is assumed to be a JSON list of dicts:
#     [{"video_path": "...", "utterance": "...", "emotion": "happy"}, ...]
# If yours is a CSV, swap json.load for pandas.read_csv and adjust accessors.
class EmotionDataset(Dataset):
    def __init__(self, manifest_path: str, tokenizer, is_train: bool):
        with open(manifest_path) as f:
            self.items = json.load(f)
        # filter out samples with unknown labels and nonexistent videos
        keep = []
        for it in self.items:
            if it["emotion"] not in LABEL2ID:
                continue
            if not Path(it["video_path"]).exists():
                continue
            keep.append(it)
        self.items = keep

        self.tokenizer = tokenizer
        self.is_train  = is_train
        self.labels    = np.array([LABEL2ID[it["emotion"]] for it in self.items],
                                  dtype=np.int64)

        # video transforms
        mean = [0.43216, 0.394666, 0.37645]
        std  = [0.22803, 0.22145, 0.216989]
        if is_train:
            self.vid_tf = transforms.Compose([
                transforms.Resize(128, antialias=True),
                transforms.RandomCrop(cfg.frame_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
                transforms.Normalize(mean=mean, std=std),
            ])
        else:
            self.vid_tf = transforms.Compose([
                transforms.Resize(128, antialias=True),
                transforms.CenterCrop(cfg.frame_size),
                transforms.Normalize(mean=mean, std=std),
            ])

    def __len__(self):
        return len(self.items)

    def _read_video(self, path: str) -> torch.Tensor:
        """Return (T, 3, H, W) uint8 tensor with T = cfg.num_frames."""
        T = cfg.num_frames
        if HAS_DECORD:
            vr = VideoReader(path, ctx=cpu(0))
            n  = len(vr)
            if n == 0:
                return torch.zeros(T, 3, 128, 128, dtype=torch.uint8)
            if self.is_train:
                # random segment-based sampling
                seg = np.linspace(0, n, T + 1).astype(int)
                idx = [random.randint(seg[i], max(seg[i], seg[i+1]-1))
                       for i in range(T)]
            else:
                idx = np.linspace(0, n - 1, T).astype(int).tolist()
            frames = vr.get_batch(idx).asnumpy()  # (T, H, W, 3) uint8
            frames = torch.from_numpy(frames).permute(0, 3, 1, 2).contiguous()
        else:
            frames, _, _ = tvio.read_video(path, pts_unit="sec", output_format="TCHW")
            n = frames.shape[0]
            if n == 0:
                return torch.zeros(T, 3, 128, 128, dtype=torch.uint8)
            idx = np.linspace(0, n - 1, T).astype(int).tolist()
            frames = frames[idx]
        return frames  # (T, 3, H, W) uint8

    def __getitem__(self, i):
        it     = self.items[i]
        label  = int(self.labels[i])

        # -- video --
        try:
            frames = self._read_video(it["video_path"])
        except Exception as e:
            log.warning(f"video read failed {it['video_path']}: {e}")
            frames = torch.zeros(cfg.num_frames, 3, 128, 128, dtype=torch.uint8)
        frames = frames.float() / 255.0
        frames = self.vid_tf(frames)               # (T, 3, H, W)
        frames = frames.permute(1, 0, 2, 3)        # (3, T, H, W) for r3d_18

        # -- text --
        tok = self.tokenizer(
            it["utterance"],
            truncation=True,
            max_length=cfg.max_text_len,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "video":          frames,
            "input_ids":      tok["input_ids"].squeeze(0),
            "attention_mask": tok["attention_mask"].squeeze(0),
            "label":          label,
        }

# ============================================================
# Model
# ============================================================
class VideoEncoder(nn.Module):
    def __init__(self, weights_path: str, out_dim: int):
        super().__init__()
        bb = r3d_18(weights=None)
        if weights_path and Path(weights_path).exists():
            state = torch.load(weights_path, map_location="cpu")
            bb.load_state_dict(state)
            log.info(f"  r3d_18 loaded from {weights_path}")
        else:
            log.warning(f"  r3d_18 weights not found at {weights_path} — random init")
        in_feats = bb.fc.in_features
        bb.fc = nn.Identity()
        self.backbone = bb
        self.proj     = nn.Linear(in_feats, out_dim)

    def forward(self, x):                              # x: (B, 3, T, H, W)
        h = self.backbone(x)                           # (B, 512)
        return self.proj(h)                            # (B, out_dim)


class TAAVN(nn.Module):
    """Text-attended audio-visual fusion (cross-attention variant).
    Here: text tokens attend to video features + residual, video pools."""
    def __init__(self, text_dim: int, vid_dim: int, hidden: int,
                 n_heads: int, dropout: float):
        super().__init__()
        self.text_proj = nn.Linear(text_dim, hidden)
        self.vid_proj  = nn.Linear(vid_dim,  hidden)
        self.attn = nn.MultiheadAttention(
            hidden, n_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, hidden * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden),
        )
        self.norm2 = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text_feats, text_mask, vid_feat):
        # text_feats: (B, L, D_t); text_mask: (B, L); vid_feat: (B, D_v)
        t = self.text_proj(text_feats)                 # (B, L, H)
        v = self.vid_proj(vid_feat).unsqueeze(1)       # (B, 1, H)
        kv = torch.cat([t, v], dim=1)                  # (B, L+1, H)
        # attention mask: True = ignore
        key_pad = torch.cat([
            text_mask == 0,
            torch.zeros(vid_feat.size(0), 1, dtype=torch.bool, device=vid_feat.device)
        ], dim=1)
        q = self.norm1(t)
        a, _ = self.attn(q, kv, kv, key_padding_mask=key_pad, need_weights=False)
        x = t + self.dropout(a)
        x = x + self.dropout(self.ffn(self.norm2(x)))

        # masked mean pool over text tokens
        mask = text_mask.unsqueeze(-1).float()
        pooled = (x * mask).sum(1) / mask.sum(1).clamp(min=1)
        return pooled                                   # (B, H)


class EmotionModel(nn.Module):
    def __init__(self):
        super().__init__()
        log.info("Building model...")

        # --- text: Llama-3.2-3B in 4-bit with LoRA ---
        log.info(f"  Loading {Path(cfg.text_model).name} (4-bit)...")
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        llm = AutoModelForCausalLM.from_pretrained(
            cfg.text_model,
            quantization_config=bnb,
            device_map={"": 0},
            attn_implementation="sdpa",
        )
        llm = prepare_model_for_kbit_training(llm, use_gradient_checkpointing=True)
        llm.config.use_cache = False

        lora_cfg = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=cfg.lora_targets,
            bias="none",
            task_type="FEATURE_EXTRACTION",
        )
        self.llm = get_peft_model(llm, lora_cfg)
        n_train = sum(p.numel() for p in self.llm.parameters() if p.requires_grad)
        log.info(f"  [LoRA] {n_train:,} trainable params")

        text_hidden = self.llm.config.hidden_size        # 3072 for Llama-3.2-3B

        # --- video ---
        self.video_enc = VideoEncoder(cfg.r3d_weights, out_dim=cfg.fusion_dim)

        # --- fusion ---
        self.taavn = TAAVN(
            text_dim=text_hidden, vid_dim=cfg.fusion_dim,
            hidden=cfg.fusion_dim, n_heads=cfg.n_heads, dropout=cfg.dropout,
        )

        # --- head ---
        self.head = nn.Sequential(
            nn.LayerNorm(cfg.fusion_dim),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.fusion_dim, cfg.fusion_dim // 2),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.fusion_dim // 2, NUM_CLASSES),
        )

    def forward(self, video, input_ids, attention_mask):
        # text hidden states
        out = self.llm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        text_h = out.hidden_states[-1]                    # (B, L, 3072)

        # video
        v = self.video_enc(video)                         # (B, fusion_dim)

        # fuse
        z = self.taavn(text_h, attention_mask, v)         # (B, fusion_dim)
        return self.head(z)                               # (B, num_classes)

# ============================================================
# Loss
# ============================================================
class FocalLoss(nn.Module):
    def __init__(self, alpha: torch.Tensor, gamma: float = 2.0,
                 label_smoothing: float = 0.05):
        super().__init__()
        self.register_buffer("alpha", alpha.float())
        self.gamma = gamma
        self.ls = label_smoothing

    def forward(self, logits, target):
        n = logits.size(-1)
        logp = F.log_softmax(logits, dim=-1)
        p    = logp.exp()
        with torch.no_grad():
            true = torch.full_like(logp, self.ls / max(n - 1, 1))
            true.scatter_(1, target.unsqueeze(1), 1.0 - self.ls)
        focal = (1 - p).pow(self.gamma)
        loss  = -(self.alpha.view(1, -1) * focal * true * logp).sum(-1)
        return loss.mean()

# ============================================================
# Schedulers
# ============================================================
def warmup_cosine(optimizer, warmup_steps, total_steps, min_ratio=0.05):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * prog))
    return LambdaLR(optimizer, lr_lambda)

# ============================================================
# Train / eval
# ============================================================
def build_optimizer(model):
    lora_params, vid_params, head_params = [], [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "lora_" in n:
            lora_params.append(p)
        elif n.startswith("video_enc."):
            vid_params.append(p)
        else:
            head_params.append(p)
    groups = []
    if lora_params: groups.append({"params": lora_params, "lr": cfg.lr_lora})
    if vid_params:  groups.append({"params": vid_params,  "lr": cfg.lr_video})
    if head_params: groups.append({"params": head_params, "lr": cfg.lr_head})
    log.info(f"  optimizer groups: lora={len(lora_params)} "
             f"video={len(vid_params)} head={len(head_params)}")
    return torch.optim.AdamW(groups, weight_decay=cfg.weight_decay, betas=(0.9, 0.95))


def evaluate(model, loader, criterion):
    model.eval()
    all_logits, all_labels, tot_loss = [], [], 0.0
    with torch.no_grad():
        for batch in loader:
            video = batch["video"].to(DEVICE, non_blocking=True)
            ids   = batch["input_ids"].to(DEVICE, non_blocking=True)
            mask  = batch["attention_mask"].to(DEVICE, non_blocking=True)
            lab   = batch["label"].to(DEVICE, non_blocking=True)
            with autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(video, ids, mask)
                loss   = criterion(logits, lab)
            tot_loss += loss.item() * lab.size(0)
            all_logits.append(logits.float().cpu())
            all_labels.append(lab.cpu())
    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy()
    preds  = logits.argmax(-1)
    probs  = torch.softmax(torch.from_numpy(logits), dim=-1).numpy()

    acc   = accuracy_score(labels, preds)
    f1    = f1_score(labels, preds, average="macro", zero_division=0)
    try:
        auc = roc_auc_score(labels, probs, multi_class="ovr", average="macro")
    except ValueError:
        auc = float("nan")
    loss_avg = tot_loss / max(len(labels), 1)
    return dict(loss=loss_avg, acc=acc, f1=f1, auc=auc,
                preds=preds, labels=labels)


def plot_dashboard(hist, path):
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    ep = list(range(len(hist["train_loss"])))

    axes[0, 0].plot(ep, hist["train_loss"], label="Train", color="tab:green")
    axes[0, 0].plot(ep, hist["val_loss"],   label="Val",   color="tab:red")
    axes[0, 0].set_title("Loss"); axes[0, 0].legend(); axes[0, 0].grid(alpha=.3)

    axes[0, 1].plot(ep, [a*100 for a in hist["train_acc"]], label="Train", color="tab:green")
    axes[0, 1].plot(ep, [a*100 for a in hist["val_acc"]],   label="Val",   color="tab:red")
    axes[0, 1].set_title("Accuracy %"); axes[0, 1].legend(); axes[0, 1].grid(alpha=.3)

    axes[0, 2].plot(ep, hist["val_f1"],  label="F1",  color="tab:purple")
    axes[0, 2].plot(ep, hist["val_auc"], label="AUC", color="tab:blue")
    axes[0, 2].set_title("Val F1 & AUC"); axes[0, 2].legend(); axes[0, 2].grid(alpha=.3)

    axes[1, 0].plot(hist["lr_hist"], color="tab:orange")
    axes[1, 0].set_title("LR (per step)"); axes[1, 0].grid(alpha=.3)

    axes[1, 1].plot(ep, hist["grad_norm"], color="tab:brown")
    axes[1, 1].set_title("Grad norm"); axes[1, 1].grid(alpha=.3)

    axes[1, 2].axis("off")
    plt.suptitle("Training Dashboard (H100)", fontweight="bold")
    plt.tight_layout(); plt.savefig(path, dpi=120, bbox_inches="tight"); plt.close()


def plot_confusion(preds, labels, acc, path):
    cm = confusion_matrix(labels, preds, labels=list(range(NUM_CLASSES)))
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="YlOrRd",
                xticklabels=cfg.classes, yticklabels=cfg.classes)
    plt.title(f"Confusion Matrix — acc={acc*100:.1f}%")
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.tight_layout(); plt.savefig(path, dpi=120, bbox_inches="tight"); plt.close()


def main():
    log.info(f"[CFG] run_dir={run_dir}")

    # --- tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(cfg.text_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    log.info(f"[TOK] {Path(cfg.text_model).name} (vocab={tokenizer.vocab_size})")

    # --- data ---
    log.info("Building datasets...")
    train_ds = EmotionDataset(cfg.train_manifest, tokenizer, is_train=True)
    val_ds   = EmotionDataset(cfg.val_manifest,   tokenizer, is_train=False)

    counts = np.bincount(train_ds.labels, minlength=NUM_CLASSES)
    log.info(f"  [train] {len(train_ds)} samples")
    for i, c in enumerate(cfg.classes):
        log.info(f"    {c:>10s} {counts[i]:>5d} {counts[i]/len(train_ds)*100:>5.1f}%")
    log.info(f"  [val]   {len(val_ds)} samples")

    # weighted sampler for imbalance
    cls_w = 1.0 / np.maximum(counts, 1)
    samp_w = cls_w[train_ds.labels]
    sampler = WeightedRandomSampler(
        torch.as_tensor(samp_w, dtype=torch.double),
        num_samples=len(train_ds), replacement=True,
    )

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, sampler=sampler,
        num_workers=cfg.workers, pin_memory=True, persistent_workers=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.workers, pin_memory=True, persistent_workers=True,
    )
    log.info(f"[DATA] train: {len(train_loader)} batches | val: {len(val_loader)} batches")

    # --- model ---
    model = EmotionModel().to(DEVICE)

    if cfg.compile_submodules:
        try:
            model.video_enc = torch.compile(model.video_enc, mode="reduce-overhead")
            model.taavn     = torch.compile(model.taavn,     mode="reduce-overhead")
            model.head      = torch.compile(model.head,      mode="reduce-overhead")
            log.info("  torch.compile: video_enc + taavn + head")
        except Exception as e:
            log.warning(f"  torch.compile failed: {e}")

    # --- loss ---
    alpha = (1.0 / np.maximum(counts, 1))
    alpha = alpha / alpha.sum() * NUM_CLASSES           # mean ~1
    criterion = FocalLoss(
        torch.as_tensor(alpha), gamma=cfg.focal_gamma,
        label_smoothing=cfg.label_smoothing,
    ).to(DEVICE)
    log.info(f"  alpha: {np.round(alpha, 3).tolist()}")

    # --- optimizer + scheduler ---
    optimizer = build_optimizer(model)
    total_steps  = len(train_loader) * cfg.epochs // cfg.grad_accum
    warmup_steps = len(train_loader) * cfg.warmup_epochs // cfg.grad_accum
    scheduler = warmup_cosine(optimizer, warmup_steps, total_steps)

    hist = dict(train_loss=[], val_loss=[], train_acc=[], val_acc=[],
                val_f1=[], val_auc=[], lr_hist=[], grad_norm=[])
    best_f1, bad = -1.0, 0
    step = 0

    for epoch in range(cfg.epochs):
        # ------- TRAIN -------
        model.train()
        t0 = time.time()
        tot_loss, tot_correct, tot_n = 0.0, 0, 0
        optimizer.zero_grad(set_to_none=True)
        gnorm_last = 0.0

        for bidx, batch in enumerate(train_loader):
            video = batch["video"].to(DEVICE, non_blocking=True)
            ids   = batch["input_ids"].to(DEVICE, non_blocking=True)
            mask  = batch["attention_mask"].to(DEVICE, non_blocking=True)
            lab   = batch["label"].to(DEVICE, non_blocking=True)

            with autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(video, ids, mask)
                loss   = criterion(logits, lab) / cfg.grad_accum

            loss.backward()

            if (bidx + 1) % cfg.grad_accum == 0:
                gnorm_last = torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    cfg.grad_clip,
                ).item()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                hist["lr_hist"].append(optimizer.param_groups[0]["lr"])

            with torch.no_grad():
                tot_loss    += loss.item() * cfg.grad_accum * lab.size(0)
                tot_correct += (logits.argmax(-1) == lab).sum().item()
                tot_n       += lab.size(0)

            if bidx % 50 == 0:
                log.info(f"  ep{epoch} b{bidx}/{len(train_loader)} "
                         f"loss={loss.item()*cfg.grad_accum:.4f} "
                         f"lr={optimizer.param_groups[0]['lr']:.2e}")

        train_loss = tot_loss / max(tot_n, 1)
        train_acc  = tot_correct / max(tot_n, 1)
        dt = time.time() - t0

        # ------- EVAL -------
        val = evaluate(model, val_loader, criterion)

        hist["train_loss"].append(train_loss)
        hist["val_loss"].append(val["loss"])
        hist["train_acc"].append(train_acc)
        hist["val_acc"].append(val["acc"])
        hist["val_f1"].append(val["f1"])
        hist["val_auc"].append(val["auc"])
        hist["grad_norm"].append(gnorm_last)

        log.info(f"[EPOCH {epoch}] {dt:.0f}s | "
                 f"train_loss={train_loss:.4f} acc={train_acc*100:.1f}% | "
                 f"val_loss={val['loss']:.4f} acc={val['acc']*100:.1f}% "
                 f"f1={val['f1']:.4f} auc={val['auc']:.4f}")

        plot_dashboard(hist, run_dir / "curves.png")
        plot_confusion(val["preds"], val["labels"], val["acc"], run_dir / "cm.png")

        # early stopping on macro-F1
        if val["f1"] > best_f1:
            best_f1, bad = val["f1"], 0
            # save only trainable params (LoRA + head + video + fusion)
            save_state = {k: v for k, v in model.state_dict().items()
                          if any(sub in k for sub in
                                 ["lora_", "video_enc", "taavn", "head"])}
            torch.save(save_state, run_dir / "best.pt")
            log.info(f"  [BEST] macro-F1={best_f1:.4f} saved")
        else:
            bad += 1
            log.info(f"  no improvement ({bad}/{cfg.patience})")
            if bad >= cfg.patience:
                log.info("  early stopping"); break

    log.info(f"DONE. best macro-F1 = {best_f1:.4f}")
    log.info(f"Artifacts: {run_dir}")


if __name__ == "__main__":
    main()
