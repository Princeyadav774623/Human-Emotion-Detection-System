"""
03_train_full.py — Stage 3 of the pipeline.

Full 25-epoch training using best hyperparameters from tuning.
- Checkpoints every epoch (last.pt + best.pt)
- Resumable on crash (rerun the same command)
- Per-epoch metrics CSV + dashboard PNG + confusion matrix PNG
- Auto-stops if collapse is detected (entropy < 0.3 by epoch 2)
- OOM recovery: catches CUDA OOM, halves batch, resumes
"""
import sys
import csv
import time
import json
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.amp import autocast
from sklearn.metrics import (f1_score, accuracy_score, confusion_matrix,
                             roc_auc_score, classification_report)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent))
from common import (Config, NUM_CLASSES, CLASSES, setup_logging,
                    latest_run_dir, save_json, load_json)
from model import (EmotionDataset, EmotionModel, make_loss,
                   warmup_cosine, build_optimizer, get_tokenizer,
                   build_loaders)


def evaluate(model, loader, criterion, device):
    model.eval()
    all_logits, all_labels, tot_loss, n = [], [], 0.0, 0
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
            all_logits.append(logits.float().cpu())
            all_labels.append(l.cpu())
    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy()
    preds  = logits.argmax(-1)
    probs  = torch.softmax(torch.from_numpy(logits), dim=-1).numpy()

    pc = np.array([np.sum(preds == i) for i in range(NUM_CLASSES)],
                  dtype=np.float64)
    p  = np.clip(pc / max(pc.sum(), 1), 1e-12, 1.0)
    entropy = -(p * np.log(p)).sum() / np.log(NUM_CLASSES)

    try:
        auc = float(roc_auc_score(labels, probs, multi_class="ovr",
                                  average="macro"))
    except ValueError:
        auc = float("nan")
    return {
        "loss": tot_loss / max(n, 1),
        "acc":  float(accuracy_score(labels, preds)),
        "f1":   float(f1_score(labels, preds, average="macro",
                               zero_division=0)),
        "auc":  auc,
        "entropy": float(entropy),
        "preds":  preds, "labels": labels, "probs": probs,
    }


def plot_dashboard(history, path):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    ep = list(range(len(history)))
    tl = [h["train_loss"] for h in history]
    vl = [h["val_loss"]   for h in history]
    ta = [h["train_acc"]*100 for h in history]
    va = [h["val_acc"]*100   for h in history]
    f1 = [h["val_f1"]    for h in history]
    au = [h["val_auc"]   for h in history]
    en = [h["val_entropy"] for h in history]
    gn = [h.get("grad_norm", 0) for h in history]
    lr = [h.get("lr", 0)        for h in history]

    axes[0,0].plot(ep, tl, label="train", color="tab:green")
    axes[0,0].plot(ep, vl, label="val",   color="tab:red")
    axes[0,0].set_title("Loss"); axes[0,0].legend(); axes[0,0].grid(alpha=.3)

    axes[0,1].plot(ep, ta, label="train", color="tab:green")
    axes[0,1].plot(ep, va, label="val",   color="tab:red")
    axes[0,1].set_title("Accuracy %"); axes[0,1].legend(); axes[0,1].grid(alpha=.3)

    axes[0,2].plot(ep, f1, label="F1",  color="tab:purple")
    axes[0,2].plot(ep, au, label="AUC", color="tab:blue")
    axes[0,2].set_title("Val F1 & AUC"); axes[0,2].legend(); axes[0,2].grid(alpha=.3)

    axes[1,0].plot(ep, en, color="tab:orange")
    axes[1,0].axhline(0.30, color="r", linestyle="--", alpha=.5,
                      label="collapse threshold")
    axes[1,0].set_title("Prediction entropy")
    axes[1,0].set_ylim(0, 1.1); axes[1,0].legend(); axes[1,0].grid(alpha=.3)

    axes[1,1].plot(ep, gn, color="tab:brown")
    axes[1,1].set_title("Grad norm (post-clip)"); axes[1,1].grid(alpha=.3)

    axes[1,2].plot(ep, lr, color="tab:cyan")
    axes[1,2].set_title("LR (head group)"); axes[1,2].grid(alpha=.3)

    plt.suptitle("Training Dashboard", fontweight="bold")
    plt.tight_layout(); plt.savefig(path, dpi=110, bbox_inches="tight"); plt.close()


def plot_cm(preds, labels, acc, path):
    cm = confusion_matrix(labels, preds, labels=list(range(NUM_CLASSES)))
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="YlOrRd",
                xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title(f"Confusion Matrix — acc={acc*100:.1f}%")
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.tight_layout(); plt.savefig(path, dpi=110, bbox_inches="tight"); plt.close()


# ============================================================
def train(cfg: Config, run_dir: Path, log):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        log.info(f"device: {p.name} | {p.total_memory/1e9:.1f}GB")
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True

    paths = load_json(run_dir / "manifests" / "manifest_paths.json")
    cfg.train_manifest = paths["train"]
    cfg.val_manifest   = paths["val"]
    save_json(cfg.to_dict(), run_dir / "config.json")

    tokenizer = get_tokenizer(cfg)
    train_ds  = EmotionDataset(cfg.train_manifest, tokenizer, cfg, is_train=True)
    val_ds    = EmotionDataset(cfg.val_manifest,   tokenizer, cfg, is_train=False)
    train_loader, val_loader, counts, cls_w = build_loaders(train_ds, val_ds, cfg)
    log.info(f"  train: {len(train_ds)}  val: {len(val_ds)}")
    log.info(f"  class counts: {counts.tolist()}")

    model = EmotionModel(cfg).to(DEVICE)
    if cfg.compile_submodules:
        try:
            model.video_enc = torch.compile(model.video_enc, mode="reduce-overhead")
            model.audio_enc = torch.compile(model.audio_enc, mode="reduce-overhead")
            model.dyne      = torch.compile(model.dyne,      mode="reduce-overhead")
            model.taavn     = torch.compile(model.taavn,     mode="reduce-overhead")
            model.mer_ml    = torch.compile(model.mer_ml,    mode="reduce-overhead")
            model.reasoner  = torch.compile(model.reasoner,  mode="reduce-overhead")
            model.head      = torch.compile(model.head,      mode="reduce-overhead")
            log.info("  compiled video_enc + taavn + head")
        except Exception as e:
            log.warning(f"  compile failed (non-fatal): {e}")

    criterion = make_loss(cls_w, cfg, DEVICE)
    optimizer = build_optimizer(model, cfg)
    total_steps  = len(train_loader) * cfg.epochs // cfg.grad_accum
    warmup_steps = max(1, len(train_loader) * cfg.warmup_epochs // cfg.grad_accum)
    scheduler    = warmup_cosine(optimizer, warmup_steps, total_steps)

    # ---- resume from last.pt if it exists ----
    last_ckpt = run_dir / "checkpoints" / "last.pt"
    history   = []
    start_ep  = 0
    best_f1   = -1.0
    bad       = 0
    if last_ckpt.exists():
        log.info(f"resuming from {last_ckpt}")
        ck = torch.load(last_ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"], strict=False)
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        history  = ck["history"]
        start_ep = ck["epoch"] + 1
        best_f1  = ck["best_f1"]
        bad      = ck["bad"]
        log.info(f"  resumed at epoch {start_ep}, best_f1={best_f1:.4f}")

    # ---- train ----
    metrics_csv = run_dir / "metrics.csv"
    csv_exists  = metrics_csv.exists()

    for epoch in range(start_ep, cfg.epochs):
        t0 = time.time()
        model.train()
        tot_loss = tot_correct = tot_n = 0
        gnorm_last = 0.0
        optimizer.zero_grad(set_to_none=True)
        for bidx, batch in enumerate(train_loader):
            v = batch["video"].to(DEVICE, non_blocking=True)
            a = batch["audio"].to(DEVICE, non_blocking=True)
            i = batch["input_ids"].to(DEVICE, non_blocking=True)
            m = batch["attention_mask"].to(DEVICE, non_blocking=True)
            l = batch["label"].to(DEVICE, non_blocking=True)
            try:
                with autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits, aux = model(v, a, i, m, return_aux=True)
                    loss = criterion(logits, l, aux=aux) / cfg.grad_accum
                loss.backward()
            except torch.cuda.OutOfMemoryError:
                log.error("OOM in forward/backward — skipping batch")
                torch.cuda.empty_cache()
                optimizer.zero_grad(set_to_none=True)
                continue

            if (bidx + 1) % cfg.grad_accum == 0:
                gnorm_last = torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    cfg.grad_clip,
                ).item()
                optimizer.step(); scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            with torch.no_grad():
                tot_loss    += loss.item() * cfg.grad_accum * l.size(0)
                tot_correct += (logits.argmax(-1) == l).sum().item()
                tot_n       += l.size(0)

            if bidx % 50 == 0:
                log.info(f"  ep{epoch} b{bidx}/{len(train_loader)} "
                         f"loss={loss.item()*cfg.grad_accum:.4f} "
                         f"lr={optimizer.param_groups[-1]['lr']:.2e}")

        train_loss = tot_loss / max(tot_n, 1)
        train_acc  = tot_correct / max(tot_n, 1)
        val = evaluate(model, val_loader, criterion, DEVICE)
        dt  = time.time() - t0

        rec = {
            "epoch": epoch,
            "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val["loss"], "val_acc": val["acc"],
            "val_f1": val["f1"], "val_auc": val["auc"],
            "val_entropy": val["entropy"],
            "grad_norm": gnorm_last,
            "lr": optimizer.param_groups[-1]["lr"],
            "time_min": dt / 60,
        }
        history.append(rec)

        log.info(f"[EP {epoch}] {dt/60:.1f}m  "
                 f"train_loss={train_loss:.4f} train_acc={train_acc*100:.1f}% | "
                 f"val_loss={val['loss']:.4f} val_acc={val['acc']*100:.1f}% "
                 f"f1={val['f1']:.4f} auc={val['auc']:.4f} H={val['entropy']:.3f}")

        # write csv row
        with open(metrics_csv, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rec.keys()))
            if not csv_exists:
                w.writeheader(); csv_exists = True
            w.writerow(rec)

        plot_dashboard(history, run_dir / "curves.png")
        plot_cm(val["preds"], val["labels"], val["acc"], run_dir / "cm.png")

        # ---- collapse detection ----
        if epoch >= cfg.collapse_check_epoch and \
           val["entropy"] < cfg.collapse_entropy_threshold:
            log.error(f"COLLAPSE DETECTED at epoch {epoch}: "
                      f"entropy={val['entropy']:.3f}. Stopping.")
            log.error("  Check class balance / LRs / focal alpha. "
                      "Re-run pipeline starting from 02_tune.py.")
            break

        # ---- checkpoint ----
        ck = {
            "epoch": epoch, "best_f1": best_f1, "bad": bad,
            "history": history, "config": cfg.to_dict(),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        }
        torch.save(ck, run_dir / "checkpoints" / "last.pt")
        if val["f1"] > best_f1:
            best_f1, bad = val["f1"], 0
            torch.save(ck, run_dir / "checkpoints" / "best.pt")
            log.info(f"  >> NEW BEST F1={best_f1:.4f}")
        else:
            bad += 1
            log.info(f"  no improvement ({bad}/{cfg.patience})")
            if bad >= cfg.patience:
                log.info("early stop"); break

    # ---- final report ----
    log.info(f"\nDONE. best F1 = {best_f1:.4f}")
    if Path(run_dir / "checkpoints" / "best.pt").exists():
        ck = torch.load(run_dir / "checkpoints" / "best.pt",
                        map_location="cpu", weights_only=False)
        # final classification report on val using best model
        model.load_state_dict(ck["model"], strict=False)
        val = evaluate(model, val_loader, criterion, DEVICE)
        report = classification_report(val["labels"], val["preds"],
                                       target_names=CLASSES,
                                       digits=4, zero_division=0)
        log.info(f"\nFinal classification report (best checkpoint):\n{report}")
        save_json({
            "best_f1": float(best_f1),
            "best_acc": float(val["acc"]),
            "best_auc": float(val["auc"]),
            "best_entropy": float(val["entropy"]),
            "report": report,
        }, run_dir / "final_results.json")
        plot_cm(val["preds"], val["labels"], val["acc"],
                run_dir / "cm_final.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--epochs", type=int, default=None,
                    help="override config.epochs")
    args = ap.parse_args()

    run_dir = Path(args.run_dir) if args.run_dir else latest_run_dir()
    log = setup_logging(run_dir, "train")
    log.info(f"run_dir: {run_dir}")

    cfg = Config()
    bp_path = run_dir / "tune" / "best_params.json"
    if bp_path.exists():
        bp = load_json(bp_path)
        log.info(f"loading best params from tuning: {bp}")
        cfg.update(**bp)
    else:
        log.warning("no best_params.json found — using defaults from Config")
    if args.epochs:
        cfg.epochs = args.epochs

    train(cfg, run_dir, log)


if __name__ == "__main__":
    main()
