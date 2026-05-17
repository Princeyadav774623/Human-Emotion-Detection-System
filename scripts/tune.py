#!/usr/bin/env python3
"""
tune.py — Automated hyperparameter search for the MELD emotion model.

Uses Optuna with the ASHA pruner: bad trials get killed at epoch 1
based on val macro-F1, so we don't waste 2 hours on collapses.

PROXY SETUP (default, ~30-50min/trial):
  - 3 epochs max per trial (instead of 25)
  - 30% of training data (set TUNE_FRACTION below)
  - smaller batch via grad accumulation
  - early-pruning at epoch 1 if F1 < 0.10

After this finds good hyperparameters, run a full training with them.

Usage:
    python tune.py --n-trials 30 --study-name meld_v1
    python tune.py --resume                       # continue existing study
    python tune.py --report                       # show best config so far
"""
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import json
import time
import logging
from pathlib import Path
from copy import deepcopy

import numpy as np
import torch
import optuna
from optuna.pruners import HyperbandPruner
from optuna.samplers import TPESampler

# import your training script's pieces
import train_h100 as T  # the script I sent you

# ============================================================
# Tuning config
# ============================================================
TUNE_EPOCHS    = 3       # max epochs per trial
TUNE_FRACTION  = 0.3     # use 30% of training data per trial
PRUNE_AT_EPOCH = 1       # prune trials with F1 < PRUNE_F1 after this epoch
PRUNE_F1       = 0.10
STUDY_DB       = "sqlite:///optuna_meld.db"

LOG = logging.getLogger("tune")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")

# ============================================================
# Search space — picked for YOUR specific failure modes
# ============================================================
def suggest_config(trial: optuna.Trial) -> dict:
    """Sample a hyperparameter configuration."""
    return {
        # learning rates — biggest lever, biggest collapse risk
        "lr_lora":   trial.suggest_float("lr_lora",   5e-5, 5e-4, log=True),
        "lr_video":  trial.suggest_float("lr_video",  1e-6, 5e-5, log=True),
        "lr_head":   trial.suggest_float("lr_head",   1e-4, 1e-3, log=True),

        # regularization
        "weight_decay": trial.suggest_float("weight_decay", 0.001, 0.1, log=True),
        "dropout":      trial.suggest_float("dropout",      0.1,   0.5),
        "label_smoothing": trial.suggest_float("label_smoothing", 0.0, 0.15),

        # focal loss — directly addresses class imbalance behavior
        "focal_gamma":  trial.suggest_float("focal_gamma",  1.0, 4.0),

        # LoRA capacity
        "lora_r":       trial.suggest_categorical("lora_r",     [8, 16, 32]),
        "lora_alpha":   trial.suggest_categorical("lora_alpha", [16, 32, 64]),
        "lora_dropout": trial.suggest_float("lora_dropout", 0.0, 0.2),

        # fusion
        "fusion_dim":   trial.suggest_categorical("fusion_dim", [256, 512, 768]),
        "n_heads":      trial.suggest_categorical("n_heads",    [4, 8]),

        # warmup
        "warmup_epochs": trial.suggest_int("warmup_epochs", 0, 2),
    }

# ============================================================
# Trial runner
# ============================================================
def run_trial(trial: optuna.Trial) -> float:
    """Train with sampled hyperparameters, return best val macro-F1."""
    sampled = suggest_config(trial)
    LOG.info(f"trial {trial.number} config: {sampled}")

    # patch the global config in train_h100
    cfg = T.cfg
    for k, v in sampled.items():
        setattr(cfg, k, v)

    # shrink for tuning
    cfg.epochs       = TUNE_EPOCHS
    cfg.batch_size   = 16            # keep stable so memory doesn't blow up
    cfg.compile_submodules = False   # compile is slow per-trial; disable

    # build datasets — subsample train for speed
    tokenizer = T.AutoTokenizer.from_pretrained(cfg.text_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    full_train = T.EmotionDataset(cfg.train_manifest, tokenizer, is_train=True)
    val_ds     = T.EmotionDataset(cfg.val_manifest,   tokenizer, is_train=False)

    # stratified subsample
    rng = np.random.default_rng(42)
    keep_idx = []
    for c in range(T.NUM_CLASSES):
        cls_idx = np.where(full_train.labels == c)[0]
        n = max(1, int(len(cls_idx) * TUNE_FRACTION))
        keep_idx.extend(rng.choice(cls_idx, n, replace=False))
    keep_idx = np.array(sorted(keep_idx))

    train_ds = torch.utils.data.Subset(full_train, keep_idx)
    train_labels = full_train.labels[keep_idx]
    LOG.info(f"  using {len(train_ds)}/{len(full_train)} train samples")

    # weighted sampler on the subset
    counts = np.bincount(train_labels, minlength=T.NUM_CLASSES)
    cls_w  = 1.0 / np.maximum(counts, 1)
    samp_w = cls_w[train_labels]
    sampler = torch.utils.data.WeightedRandomSampler(
        torch.as_tensor(samp_w, dtype=torch.double),
        num_samples=len(train_ds), replacement=True,
    )

    train_loader = T.DataLoader(
        train_ds, batch_size=cfg.batch_size, sampler=sampler,
        num_workers=cfg.workers, pin_memory=True,
        persistent_workers=True, drop_last=True,
    )
    val_loader = T.DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.workers, pin_memory=True, persistent_workers=True,
    )

    # build model
    try:
        model = T.EmotionModel().to(T.DEVICE)
    except Exception as e:
        LOG.error(f"model build failed: {e}")
        raise optuna.TrialPruned()

    # focal loss with sampled gamma
    alpha = cls_w / cls_w.sum() * T.NUM_CLASSES
    criterion = T.FocalLoss(
        torch.as_tensor(alpha),
        gamma=cfg.focal_gamma,
        label_smoothing=cfg.label_smoothing,
    ).to(T.DEVICE)

    optimizer = T.build_optimizer(model)
    total_steps  = len(train_loader) * cfg.epochs
    warmup_steps = max(1, len(train_loader) * cfg.warmup_epochs)
    scheduler    = T.warmup_cosine(optimizer, warmup_steps, total_steps)

    best_f1 = 0.0
    for epoch in range(cfg.epochs):
        # ----- train one epoch -----
        model.train()
        t0 = time.time()
        for batch in train_loader:
            video = batch["video"].to(T.DEVICE, non_blocking=True)
            ids   = batch["input_ids"].to(T.DEVICE, non_blocking=True)
            mask  = batch["attention_mask"].to(T.DEVICE, non_blocking=True)
            lab   = batch["label"].to(T.DEVICE, non_blocking=True)
            with T.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(video, ids, mask)
                loss   = criterion(logits, lab)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step(); scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        # ----- eval -----
        val = T.evaluate(model, val_loader, criterion)
        f1  = val["f1"]
        dt  = time.time() - t0
        LOG.info(f"  trial {trial.number} ep {epoch}: F1={f1:.4f} "
                 f"acc={val['acc']*100:.1f}% ({dt/60:.1f}m)")

        # report to Optuna for pruning
        trial.report(f1, step=epoch)

        # prune if collapsed
        if epoch >= PRUNE_AT_EPOCH and f1 < PRUNE_F1:
            LOG.info(f"  trial {trial.number} pruned: F1={f1:.4f} < {PRUNE_F1}")
            del model; torch.cuda.empty_cache()
            raise optuna.TrialPruned()

        if trial.should_prune():
            del model; torch.cuda.empty_cache()
            raise optuna.TrialPruned()

        best_f1 = max(best_f1, f1)

    del model; torch.cuda.empty_cache()
    return best_f1

# ============================================================
# Main
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-trials", type=int, default=30)
    ap.add_argument("--study-name", default="meld_emotion_v1")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()

    sampler = TPESampler(seed=42, n_startup_trials=5)
    pruner  = HyperbandPruner(min_resource=1, max_resource=TUNE_EPOCHS,
                              reduction_factor=3)

    study = optuna.create_study(
        study_name=args.study_name,
        storage=STUDY_DB,
        load_if_exists=args.resume,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
    )

    if args.report:
        df = study.trials_dataframe(attrs=("number", "value", "state", "params"))
        print(df.to_string())
        if study.best_trial:
            print("\nbest trial:")
            print(f"  F1 = {study.best_value:.4f}")
            print(f"  params = {json.dumps(study.best_params, indent=2)}")
        return

    LOG.info(f"starting {args.n_trials} trials  study={args.study_name}")
    study.optimize(run_trial, n_trials=args.n_trials,
                   gc_after_trial=True, show_progress_bar=False)

    LOG.info(f"\n=== best ===")
    LOG.info(f"  F1 = {study.best_value:.4f}")
    LOG.info(f"  params = {json.dumps(study.best_params, indent=2)}")

    # save for use in full training run
    Path("best_params.json").write_text(json.dumps(study.best_params, indent=2))
    LOG.info("saved best_params.json")


if __name__ == "__main__":
    main()
