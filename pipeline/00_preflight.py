"""
00_preflight.py — Stage 0 of the pipeline.

Scans all videos in the train/val manifests with a hard timeout, drops any
that are corrupted or unreadable, validates labels, and writes cleaned
manifests into runs/<ts>/manifests/.

Run this once before tuning. Fast (~1-2 min/1000 videos with decord).
"""
import sys
import json
import argparse
from pathlib import Path
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import signal
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from common import (CLASSES, LABEL2ID, normalize_label, save_json,
                    make_run_dir, setup_logging, load_json, Config)

try:
    from decord import VideoReader, cpu
    HAS_DECORD = True
except ImportError:
    HAS_DECORD = False


class _Timeout(Exception): pass
def _alarm(s, f): raise _Timeout()


def probe_video(path: str, timeout: int = 5) -> dict:
    """Try to open + read 1 frame. Returns dict with status + frame count."""
    out = {"path": path, "ok": False, "frames": 0, "reason": None}
    if not Path(path).exists():
        out["reason"] = "file not found"
        return out
    if Path(path).stat().st_size < 1024:
        out["reason"] = f"file too small ({Path(path).stat().st_size}B)"
        return out
    if HAS_DECORD:
        signal.signal(signal.SIGALRM, _alarm)
        signal.alarm(timeout)
        try:
            vr = VideoReader(path, ctx=cpu(0))
            n = len(vr)
            if n < 8:
                out["reason"] = f"only {n} frames"
            else:
                _ = vr.get_batch([0]).asnumpy()  # actually read a frame
                out["ok"] = True
                out["frames"] = n
        except _Timeout:
            out["reason"] = f"timeout after {timeout}s"
        except Exception as e:
            out["reason"] = f"{type(e).__name__}: {str(e)[:80]}"
        finally:
            signal.alarm(0)
    else:
        # fallback: just check existence + size
        out["ok"] = True
        out["frames"] = -1  # unknown
    return out


def scan_manifest(manifest_path: Path, log, timeout: int = 5) -> dict:
    """Scan a manifest, return {clean: [...], dropped: [...], stats: {...}}."""
    log.info(f"scanning {manifest_path}")
    items = load_json(manifest_path)
    log.info(f"  {len(items)} items in manifest")

    # 1. label normalization + filtering
    label_counts = Counter()
    label_dropped = []
    label_normed = []
    for it in items:
        raw = it.get("emotion", "")
        norm = normalize_label(raw)
        if norm not in LABEL2ID:
            label_dropped.append({"path": it.get("video_path"),
                                  "reason": f"unknown label '{raw}'"})
            continue
        if not it.get("video_path"):
            label_dropped.append({"path": "<missing>",
                                  "reason": "no video_path field"})
            continue
        # rewrite normalized label
        it["emotion"] = norm
        label_normed.append(it)
        label_counts[norm] += 1

    log.info(f"  after label filter: {len(label_normed)} items, "
             f"{len(label_dropped)} dropped")
    log.info(f"  label distribution: {dict(label_counts)}")

    # 2. video probing (parallel)
    log.info(f"  probing videos (timeout={timeout}s, parallel)...")
    results = []

    # NOTE: decord + signal-based timeout is not threadsafe across workers.
    # Run serially here — a single thread is fine because decord is fast on
    # readable files; corruption shows up either as instant errors or as
    # the exact timeouts we want to catch.
    for it in tqdm(label_normed, desc="    probe", unit="vid"):
        results.append((it, probe_video(it["video_path"], timeout=timeout)))

    clean, dropped = [], list(label_dropped)
    for it, r in results:
        if r["ok"]:
            clean.append(it)
        else:
            dropped.append({"path": r["path"], "reason": r["reason"]})

    # summary
    drop_reasons = Counter(d["reason"].split(":")[0] for d in dropped)
    final_counts = Counter(it["emotion"] for it in clean)

    return {
        "clean": clean,
        "dropped": dropped,
        "stats": {
            "input": len(items),
            "kept":  len(clean),
            "dropped": len(dropped),
            "drop_reasons": dict(drop_reasons),
            "final_label_counts": dict(final_counts),
        }
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default=str(Config().train_manifest))
    ap.add_argument("--val",   default=str(Config().val_manifest))
    ap.add_argument("--timeout", type=int, default=5)
    ap.add_argument("--run-name", default="full")
    args = ap.parse_args()

    run_dir = make_run_dir(args.run_name)
    log     = setup_logging(run_dir, "preflight")
    log.info(f"run_dir: {run_dir}")

    summary = {}
    for split, path in [("train", args.train), ("val", args.val)]:
        path = Path(path)
        if not path.exists():
            log.error(f"  MANIFEST MISSING: {path}")
            log.error(f"  Generate it before running pre-flight.")
            sys.exit(2)
        result = scan_manifest(path, log, timeout=args.timeout)
        # write clean manifest
        out = run_dir / "manifests" / f"{split}_clean.json"
        save_json(result["clean"], out)
        # write drop log for inspection
        save_json(result["dropped"], run_dir / "manifests" / f"{split}_dropped.json")
        log.info(f"  -> {out} ({result['stats']['kept']} items)")
        log.info(f"     dropped: {result['stats']['drop_reasons']}")
        summary[split] = result["stats"]

    # imbalance ratio (training only, for the user's awareness)
    counts = summary["train"]["final_label_counts"]
    if counts:
        imb = max(counts.values()) / max(min(counts.values()), 1)
        log.info(f"\nTRAIN imbalance ratio: {imb:.1f}x")
        if imb > 10:
            log.info(f"  >> WeightedRandomSampler + focal loss WILL be used")

    # write summary
    save_json(summary, run_dir / "preflight_summary.json")
    # write a marker pointing to clean manifests for downstream stages
    save_json({
        "run_dir":  str(run_dir),
        "train":    str(run_dir / "manifests" / "train_clean.json"),
        "val":      str(run_dir / "manifests" / "val_clean.json"),
    }, run_dir / "manifests" / "manifest_paths.json")

    log.info(f"\npre-flight complete: {run_dir}")
    print(run_dir)  # for shell capture


if __name__ == "__main__":
    main()
