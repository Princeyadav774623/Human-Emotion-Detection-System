"""
01_diagnostics.py — Stage 1 of the pipeline.

Verifies the model + data pipeline work end-to-end before spending hours on
tuning. Catches: sampler not balancing, NaN loss, frozen modules, no gradient
flow into TAAVN/head (the bug that caused your last collapse).

Run after pre-flight, before tuning. Takes ~3-5 minutes.
"""
import sys
import argparse
from pathlib import Path
from collections import Counter

import numpy as np
import torch
from torch.amp import autocast

sys.path.insert(0, str(Path(__file__).parent))
from common import (Config, NUM_CLASSES, CLASSES, setup_logging,
                    latest_run_dir, load_json, save_json)
from model import (EmotionDataset, EmotionModel, build_loaders, make_loss,
                   build_optimizer, get_tokenizer)


def diagnose(cfg: Config, run_dir: Path, log):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"device: {DEVICE}")
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        log.info(f"  {p.name} | {p.total_memory/1e9:.1f}GB")

    # --- load clean manifests ---
    paths = load_json(run_dir / "manifests" / "manifest_paths.json")
    cfg.train_manifest = paths["train"]
    cfg.val_manifest   = paths["val"]
    log.info(f"using clean manifests:")
    log.info(f"  train: {cfg.train_manifest}")
    log.info(f"  val:   {cfg.val_manifest}")

    tokenizer = get_tokenizer(cfg)
    log.info(f"  tokenizer vocab: {tokenizer.vocab_size}")

    train_ds = EmotionDataset(cfg.train_manifest, tokenizer, cfg, is_train=True)
    val_ds   = EmotionDataset(cfg.val_manifest,   tokenizer, cfg, is_train=False)
    log.info(f"  train: {len(train_ds)}  val: {len(val_ds)}")

    train_loader, val_loader, counts, cls_w = build_loaders(train_ds, val_ds, cfg)
    log.info(f"  class counts: {dict(zip(CLASSES, counts.tolist()))}")

    # ---------- 1. sampler balance check ----------
    log.info("\n[1/4] sampler balance check (10 batches)...")
    seen = Counter()
    it = iter(train_loader)
    for _ in range(10):
        try:
            batch = next(it)
        except StopIteration:
            break
        seen.update(batch["label"].tolist())
    total = sum(seen.values())
    log.info(f"  seen distribution over 10 batches:")
    for cid, c in CLASSES_with_counts(seen):
        log.info(f"    {c:>10s}: {seen.get(cid, 0):>4d}  ({seen.get(cid, 0)/max(total,1)*100:>5.1f}%)")
    # check uniformity
    pcts = np.array([seen.get(i, 0) / max(total, 1) for i in range(NUM_CLASSES)])
    expected = 1.0 / NUM_CLASSES
    deviation = np.abs(pcts - expected).max()
    if deviation < 0.10:
        log.info(f"  PASS: max class deviation {deviation:.3f} < 0.10")
    else:
        log.warning(f"  WARN: max class deviation {deviation:.3f} > 0.10 — "
                    f"sampler may not be balancing correctly")

    # ---------- 2. batch contents check ----------
    log.info("\n[2/4] batch contents check...")
    batch = next(iter(train_loader))
    v = batch["video"]; ids = batch["input_ids"]; lab = batch["label"]
    log.info(f"  video: {tuple(v.shape)}  dtype={v.dtype}  "
             f"min={v.min().item():.3f} max={v.max().item():.3f}")
    log.info(f"  input_ids: {tuple(ids.shape)}  dtype={ids.dtype}")
    log.info(f"  labels: {lab[:8].tolist()} ... range=[{lab.min().item()},{lab.max().item()}]")
    # check for all-zero videos (silent failure)
    n_zero = (v.view(v.size(0), -1).abs().sum(1) < 1e-6).sum().item()
    if n_zero > 0:
        log.warning(f"  WARN: {n_zero}/{v.size(0)} videos in batch are all-zero "
                    f"(read failures). Pre-flight may have missed some.")
    else:
        log.info(f"  PASS: all videos non-zero")
    # decode a sample
    sample_text = tokenizer.decode(ids[0], skip_special_tokens=True)[:120]
    log.info(f"  sample text: {sample_text!r}")

    # ---------- 3. forward + backward dry run ----------
    log.info("\n[3/4] forward + backward dry run (this loads the LLM)...")
    model = EmotionModel(cfg).to(DEVICE)

    if cfg.compile_submodules:
        try:
            model.video_enc = torch.compile(model.video_enc, mode="reduce-overhead")
            log.info("  compiled video_enc")
        except Exception as e:
            log.warning(f"  compile failed (non-fatal): {e}")

    criterion = make_loss(cls_w, cfg, DEVICE)
    optimizer = build_optimizer(model, cfg)

    batch = {k: v.to(DEVICE, non_blocking=True) for k, v in batch.items()}
    model.train()
    with autocast(device_type="cuda", dtype=torch.bfloat16):
        logits, aux = model(batch["video"], batch["audio"],
                            batch["input_ids"], batch["attention_mask"],
                            return_aux=True)
        loss = criterion(logits, batch["label"], aux=aux)
    log.info(f"  logits: {tuple(logits.shape)}  "
             f"min={logits.min().item():.3f} max={logits.max().item():.3f}")
    log.info(f"  per-class logit std: "
             f"{[round(s, 3) for s in logits.float().std(0).tolist()]}")
    log.info(f"  loss: {loss.item():.4f}")
    if torch.isnan(loss) or torch.isinf(loss):
        log.error(f"  FAIL: loss is NaN/Inf — check focal alpha or logit scale")
        return False
    if logits.float().std(0).max().item() < 1e-4:
        log.warning(f"  WARN: logits have ~zero variance across classes")

    loss.backward()

    # ---------- 4. gradient flow check ----------
    log.info("\n[4/4] gradient flow check (the bug that caused your collapse)...")
    groups = {
        "lora_":     [p for n, p in model.named_parameters()
                      if "lora_" in n and p.grad is not None],
        "video_enc": [p for n, p in model.named_parameters()
                      if n.startswith("video_enc") and p.grad is not None],
        "audio_enc": [p for n, p in model.named_parameters()
                      if n.startswith("audio_enc") and p.grad is not None],
        "dyne":      [p for n, p in model.named_parameters()
                      if n.startswith("dyne") and p.grad is not None],
        "taavn":     [p for n, p in model.named_parameters()
                      if n.startswith("taavn") and p.grad is not None],
        "mer_ml":    [p for n, p in model.named_parameters()
                      if n.startswith("mer_ml") and p.grad is not None],
        "reasoner":  [p for n, p in model.named_parameters()
                      if n.startswith("reasoner") and p.grad is not None],
        "head":      [p for n, p in model.named_parameters()
                      if n.startswith("head") and p.grad is not None],
    }
    norms = {}
    failed = False
    for name, ps in groups.items():
        if not ps:
            log.error(f"  FAIL: {name} has NO gradient-bearing params — frozen?")
            failed = True
            continue
        norm = sum(p.grad.norm().item()**2 for p in ps) ** 0.5
        norms[name] = norm
        flag = ""
        if norm < 1e-6:
            flag = "  << VANISHING"; failed = True
        elif norm > 1e3:
            flag = "  << EXPLODING"
        log.info(f"  {name:<10s} grad_norm = {norm:.4e}{flag}")

    # critical: if head/taavn dominate and video is dead, you'll collapse
    if "video_enc" in norms and "head" in norms:
        ratio = norms["video_enc"] / max(norms["head"], 1e-12)
        if ratio < 0.001:
            log.warning(f"  WARN: video grad / head grad = {ratio:.2e} — "
                        f"head may ignore video features")

    save_json({"sampler_pcts": pcts.tolist(),
               "loss": float(loss.item()),
               "grad_norms": norms,
               "n_zero_videos": int(n_zero)},
              run_dir / "diagnostics" / "preflight_checks.json")

    log.info(f"\nDIAGNOSTICS: {'PASS' if not failed else 'FAIL'}")
    return not failed


def CLASSES_with_counts(seen):
    """yield (idx, name) for each class — helper for nicer logging."""
    return list(enumerate(CLASSES))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=None,
                    help="run dir from preflight; default = latest")
    args = ap.parse_args()

    run_dir = Path(args.run_dir) if args.run_dir else latest_run_dir()
    log = setup_logging(run_dir, "diagnostics")
    log.info(f"run_dir: {run_dir}")

    cfg = Config()
    ok  = diagnose(cfg, run_dir, log)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
