"""
02_tune.py — Stage 2 of the pipeline.

Optuna hyperparameter tuning with Hyperband pruning + entropy-based
collapse detection. Trials that collapse to a single class get killed
by epoch 2 instead of running for 25 epochs.

Standard config: 25 trials, 30% data subset, 3 epochs/trial, ~12hr total.
"""
import sys
import time
import json
import argparse
import logging
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import optuna
from optuna.pruners import HyperbandPruner
from optuna.samplers import TPESampler
from torch.amp import autocast
from torch.utils.data import Subset

sys.path.insert(0, str(Path(__file__).parent))
from common import (Config, NUM_CLASSES, CLASSES, setup_logging,
                    latest_run_dir, save_json, load_json)
from model import (EmotionDataset, EmotionModel, make_loss,
                   warmup_cosine, build_optimizer, get_tokenizer,
                   build_loaders)

# ---------- tuning constants (Standard) ----------
TUNE_EPOCHS    = 3
TUNE_FRACTION  = 0.30
PRUNE_AT_EPOCH = 1
PRUNE_F1       = 0.10
PRUNE_ENTROPY  = 0.30
N_TRIALS_DEFAULT = 25


# ============================================================
def evaluate_quick(model, loader, criterion, device):
    from sklearn.metrics import f1_score, accuracy_score
    model.eval()
    all_preds, all_labels, tot_loss, n = [], [], 0.0, 0
    with torch.no_grad():
        for batch in loader:
            v = batch["video"].to(device, non_blocking=True)
            a = batch["audio"].to(device, non_blocking=True)
            i = batch["input_ids"].to(device, non_blocking=True)
            m = batch["attention_mask"].to(device, non_blocking=True)
            l = batch["label"].to(device, non_blocking=True)
            with autocast(device_type="cuda", dtype=torch.bfloat16):
                logits, aux = model(v, a, i, m, return_aux=True)
                loss = criterion(logits, l, aux=aux)
            tot_loss += loss.item() * l.size(0)
            n        += l.size(0)
            all_preds.append(logits.float().argmax(-1).cpu())
            all_labels.append(l.cpu())
    preds  = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()

    # prediction-distribution entropy (collapse detector)
    pred_counts = np.array([np.sum(preds == i) for i in range(NUM_CLASSES)],
                           dtype=np.float64)
    p = pred_counts / max(pred_counts.sum(), 1)
    p = np.clip(p, 1e-12, 1.0)
    entropy = -(p * np.log(p)).sum() / np.log(NUM_CLASSES)

    return {
        "loss":    tot_loss / max(n, 1),
        "acc":     float(accuracy_score(labels, preds)),
        "f1":      float(f1_score(labels, preds, average="macro",
                                  zero_division=0)),
        "entropy": float(entropy),
        "preds":   preds,
        "labels":  labels,
    }


def stratified_subset(labels: np.ndarray, fraction: float, seed: int = 42):
    rng = np.random.default_rng(seed)
    idx = []
    for c in range(NUM_CLASSES):
        cls_idx = np.where(labels == c)[0]
        n = max(1, int(round(len(cls_idx) * fraction)))
        idx.extend(rng.choice(cls_idx, n, replace=False).tolist())
    return np.array(sorted(idx))


# ============================================================
def make_objective(run_dir, log):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    paths = load_json(run_dir / "manifests" / "manifest_paths.json")

    def objective(trial: optuna.Trial) -> float:
        cfg = Config()
        cfg.train_manifest = paths["train"]
        cfg.val_manifest   = paths["val"]
        cfg.epochs         = TUNE_EPOCHS
        cfg.compile_submodules = False  # too slow per trial

        # ---- search space ----
        cfg.lr_lora        = trial.suggest_float("lr_lora",   5e-5, 5e-4, log=True)
        cfg.lr_video       = trial.suggest_float("lr_video",  1e-6, 5e-5, log=True)
        cfg.lr_head        = trial.suggest_float("lr_head",   1e-4, 1e-3, log=True)
        cfg.weight_decay   = trial.suggest_float("weight_decay", 1e-3, 1e-1, log=True)
        cfg.dropout        = trial.suggest_float("dropout", 0.1, 0.5)
        cfg.label_smoothing = trial.suggest_float("label_smoothing", 0.0, 0.15)
        cfg.focal_gamma    = trial.suggest_float("focal_gamma", 1.0, 4.0)
        cfg.lora_r         = trial.suggest_categorical("lora_r",     [8, 16, 32])
        cfg.lora_alpha     = trial.suggest_categorical("lora_alpha", [16, 32, 64])
        cfg.lora_dropout   = trial.suggest_float("lora_dropout", 0.0, 0.2)
        cfg.fusion_dim     = trial.suggest_categorical("fusion_dim", [256, 512, 768])
        cfg.n_heads        = trial.suggest_categorical("n_heads",    [4, 8])
        cfg.warmup_epochs  = trial.suggest_int("warmup_epochs", 0, 2)

        log.info(f"\n=== trial {trial.number} ===")
        for k in ("lr_lora", "lr_video", "lr_head", "focal_gamma",
                  "lora_r", "fusion_dim", "warmup_epochs"):
            log.info(f"  {k} = {getattr(cfg, k)}")

        # ---- data ----
        tokenizer = get_tokenizer(cfg)
        full_train = EmotionDataset(cfg.train_manifest, tokenizer, cfg, is_train=True)
        val_ds     = EmotionDataset(cfg.val_manifest,   tokenizer, cfg, is_train=False)
        keep = stratified_subset(full_train.labels, TUNE_FRACTION)
        train_ds = SubsetWithLabels(full_train, keep)
        log.info(f"  train: {len(train_ds)}/{len(full_train)} (subsampled)")

        train_loader, val_loader, counts, cls_w = build_loaders(train_ds, val_ds, cfg)

        # ---- model ----
        try:
            model = EmotionModel(cfg).to(DEVICE)
        except torch.cuda.OutOfMemoryError:
            log.error(f"  trial {trial.number} OOM — pruning")
            torch.cuda.empty_cache()
            raise optuna.TrialPruned()

        criterion = make_loss(cls_w, cfg, DEVICE)
        optimizer = build_optimizer(model, cfg)
        total_steps  = len(train_loader) * cfg.epochs
        warmup_steps = max(1, len(train_loader) * cfg.warmup_epochs)
        scheduler    = warmup_cosine(optimizer, warmup_steps, total_steps)

        best_f1 = 0.0
        try:
            for epoch in range(cfg.epochs):
                t0 = time.time()
                model.train()
                for batch in train_loader:
                    v = batch["video"].to(DEVICE, non_blocking=True)
                    a = batch["audio"].to(DEVICE, non_blocking=True)
                    i = batch["input_ids"].to(DEVICE, non_blocking=True)
                    m = batch["attention_mask"].to(DEVICE, non_blocking=True)
                    l = batch["label"].to(DEVICE, non_blocking=True)
                    with autocast(device_type="cuda", dtype=torch.bfloat16):
                        logits, aux = model(v, a, i, m, return_aux=True)
                        loss = criterion(logits, l, aux=aux)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad],
                        cfg.grad_clip,
                    )
                    optimizer.step(); scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                val = evaluate_quick(model, val_loader, criterion, DEVICE)
                dt  = time.time() - t0
                log.info(f"  ep {epoch}: F1={val['f1']:.4f} acc={val['acc']*100:.1f}% "
                         f"H={val['entropy']:.3f} ({dt/60:.1f}m)")

                trial.report(val["f1"], step=epoch)
                # collapse detection
                if epoch >= PRUNE_AT_EPOCH:
                    if val["f1"] < PRUNE_F1:
                        log.info(f"  pruned: F1={val['f1']:.4f} < {PRUNE_F1}")
                        raise optuna.TrialPruned()
                    if val["entropy"] < PRUNE_ENTROPY:
                        log.info(f"  pruned: collapsed (entropy={val['entropy']:.3f})")
                        raise optuna.TrialPruned()
                if trial.should_prune():
                    raise optuna.TrialPruned()

                best_f1 = max(best_f1, val["f1"])
        finally:
            del model; torch.cuda.empty_cache()

        return best_f1

    return objective


class SubsetWithLabels(Subset):
    """Subset that exposes .labels attribute for the WeightedRandomSampler."""
    def __init__(self, dataset, indices):
        super().__init__(dataset, indices)
        self.labels = dataset.labels[np.asarray(indices)]


# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--n-trials", type=int, default=N_TRIALS_DEFAULT)
    ap.add_argument("--study-name", default="meld_v1")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    run_dir = Path(args.run_dir) if args.run_dir else latest_run_dir()
    log = setup_logging(run_dir, "tune")
    log.info(f"run_dir: {run_dir}")
    log.info(f"trials: {args.n_trials}  fraction: {TUNE_FRACTION}  "
             f"epochs/trial: {TUNE_EPOCHS}")

    db_path = run_dir / "tune" / "optuna.db"
    storage = f"sqlite:///{db_path}"

    sampler = TPESampler(seed=42, n_startup_trials=5)
    pruner  = HyperbandPruner(min_resource=1, max_resource=TUNE_EPOCHS,
                              reduction_factor=3)

    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        load_if_exists=args.resume,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
    )

    objective = make_objective(run_dir, log)
    study.optimize(objective, n_trials=args.n_trials,
                   gc_after_trial=True, show_progress_bar=False)

    log.info("\n=== best ===")
    log.info(f"  F1 = {study.best_value:.4f}")
    log.info(f"  params = {json.dumps(study.best_params, indent=2)}")

    save_json(study.best_params, run_dir / "tune" / "best_params.json")
    save_json({"best_f1": float(study.best_value),
               "n_trials": len(study.trials),
               "n_pruned": sum(1 for t in study.trials
                               if t.state == optuna.trial.TrialState.PRUNED),
               "n_complete": sum(1 for t in study.trials
                                 if t.state == optuna.trial.TrialState.COMPLETE)},
              run_dir / "tune" / "tune_summary.json")
    log.info(f"saved {run_dir/'tune'/'best_params.json'}")


if __name__ == "__main__":
    main()
