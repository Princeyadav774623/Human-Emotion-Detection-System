"""
04_export.py — Stage 4 of the pipeline.

Builds a self-contained deployment bundle from the best checkpoint.
The inline inference.py contains the full model definition (audio
encoder, video encoder, DynE, TA-AVN, MER-ML(SE), Reasoner) so the
deploy host doesn't need the pipeline/ directory.
"""
import sys
import json
import argparse
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from common import (CLASSES, NUM_CLASSES, setup_logging,
                    latest_run_dir, save_json)


INFERENCE_TEMPLATE = '''"""
inference.py — Self-contained predictor for FastAPI / any Python backend.

Architecture: text (Llama-3.2-3B + LoRA) + audio (CNN-BiLSTM)
            + video (r3d_18 + DynE) > TA-AVN > MER-ML(SE) > Reasoner(4L)

Usage:
    from inference import EmotionPredictor
    p = EmotionPredictor("path/to/deploy")
    result = p.predict("clip.mp4", "What is going on here?!")
    # -> {"label": "surprise", "confidence": 0.71, "scores": {...}}

The video file is expected to contain audio. If you have separate audio
files, pass `audio_path=...` to predict().
"""
import os
import json
import math
import signal
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.video import r3d_18
from torchvision import transforms
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

try:
    from decord import VideoReader, cpu
    HAS_DECORD = True
except ImportError:
    HAS_DECORD = False
    import torchvision.io as tvio

try:
    import torchaudio
    import torchaudio.transforms as tat
    HAS_TORCHAUDIO = True
except ImportError:
    HAS_TORCHAUDIO = False


SAMPLE_RATE = 16000
N_FFT       = 400
HOP_LENGTH  = 160
N_MELS      = 64
N_MFCC      = 20
AUDIO_SECS  = 4
AUDIO_FEAT_DIM   = N_MELS + N_MFCC
AUDIO_TIME_STEPS = AUDIO_SECS * SAMPLE_RATE // HOP_LENGTH


class _Timeout(Exception): pass
def _alarm(s, f): raise _Timeout()


# ============================================================
# Inline modules — keep in lockstep with pipeline/model.py
# ============================================================
class _AudioEncoder(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, dropout):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_dim, 128, 3, padding=1), nn.BatchNorm1d(128), nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 128, 3, padding=1), nn.BatchNorm1d(128), nn.GELU(),
            nn.MaxPool1d(2), nn.Dropout(dropout),
        )
        self.bilstm = nn.LSTM(128, hidden, num_layers=2, batch_first=True,
                              bidirectional=True, dropout=dropout)
        self.proj = nn.Linear(hidden * 2, out_dim)
        self.norm = nn.LayerNorm(out_dim)
    def forward(self, x):
        h = self.cnn(x).transpose(1, 2)
        out, _ = self.bilstm(h)
        return self.norm(self.proj(out.mean(dim=1)))


class _VideoEnc(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        bb = r3d_18(weights=None)
        in_feats = bb.fc.in_features
        bb.fc = nn.Identity()
        self.backbone = bb
        self.proj = nn.Linear(in_feats, out_dim)
    def forward(self, x):
        return self.proj(self.backbone(x))


class _DynE(nn.Module):
    def __init__(self, dim, n_tokens=8, n_heads=4, dropout=0.1):
        super().__init__()
        self.tokens = nn.Parameter(torch.randn(1, n_tokens, dim) * 0.02)
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim*2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(dim*2, dim))
        self.norm2 = nn.LayerNorm(dim)
    def forward(self, vid_feat):
        B = vid_feat.size(0)
        seq = torch.cat([vid_feat.unsqueeze(1),
                         self.tokens.expand(B, -1, -1)], dim=1)
        q = self.norm1(seq)
        a, _ = self.attn(q, q, q, need_weights=False)
        x = seq + a
        x = x + self.ffn(self.norm2(x))
        return x


class _TAAVN(nn.Module):
    def __init__(self, text_dim, audio_dim, vid_dim, hidden, n_heads, dropout):
        super().__init__()
        self.text_proj  = nn.Linear(text_dim,  hidden)
        self.audio_proj = nn.Linear(audio_dim, hidden)
        self.vid_proj   = nn.Linear(vid_dim,   hidden)
        self.attn = nn.MultiheadAttention(hidden, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, hidden*2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden*2, hidden))
        self.norm2 = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)
    def forward(self, t_h, t_mask, a_feat, v_seq):
        t = self.text_proj(t_h)
        a = self.audio_proj(a_feat).unsqueeze(1)
        v = self.vid_proj(v_seq)
        kv = torch.cat([t, a, v], dim=1)
        B, L = t_mask.shape
        kp = torch.cat([
            t_mask == 0,
            torch.zeros(B, 1 + v.size(1), dtype=torch.bool, device=t_h.device),
        ], dim=1)
        q = self.norm1(t)
        a_out, _ = self.attn(q, kv, kv, key_padding_mask=kp, need_weights=False)
        x = t + self.dropout(a_out)
        x = x + self.dropout(self.ffn(self.norm2(x)))
        m = t_mask.unsqueeze(-1).float()
        return (x * m).sum(1) / m.sum(1).clamp(min=1)


class _SE(nn.Module):
    def __init__(self, dim, ratio=8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim, dim // ratio), nn.GELU(),
            nn.Linear(dim // ratio, dim), nn.Sigmoid())
    def forward(self, x):
        return x * self.fc(x)


class _MERMLSE(nn.Module):
    def __init__(self, dim, dropout=0.2):
        super().__init__()
        self.se = _SE(dim, 8)
        self.refine = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim),
            nn.GELU(), nn.Dropout(dropout))
        self.aux = nn.Sequential(
            nn.Linear(dim, dim // 2), nn.GELU(),
            nn.Linear(dim // 2, 2))
    def forward(self, x):
        gated = self.se(x)
        refined = self.refine(gated)
        return refined, self.aux(refined)


class _Reasoner(nn.Module):
    def __init__(self, dim, n_layers=4, n_heads=8, n_tokens=4, dropout=0.1):
        super().__init__()
        self.cls = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.tokens = nn.Parameter(torch.randn(1, n_tokens, dim) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=n_heads, dim_feedforward=dim*4,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(dim)
    def forward(self, x):
        B = x.size(0)
        seq = torch.cat([self.cls.expand(B, -1, -1),
                         x.unsqueeze(1),
                         self.tokens.expand(B, -1, -1)], dim=1)
        return self.norm(self.encoder(seq)[:, 0])


class _EmotionModel(nn.Module):
    def __init__(self, cfg, num_classes, llama_dir):
        super().__init__()
        bnb = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True)
        llm = AutoModelForCausalLM.from_pretrained(
            llama_dir, quantization_config=bnb, device_map={"": 0},
            attn_implementation="sdpa")
        llm = prepare_model_for_kbit_training(llm)
        llm.config.use_cache = False
        self.llm = get_peft_model(llm, LoraConfig(
            r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"],
            lora_dropout=cfg["lora_dropout"], bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            task_type="FEATURE_EXTRACTION"))
        text_hidden = self.llm.config.hidden_size

        self.audio_enc = _AudioEncoder(AUDIO_FEAT_DIM, 128,
                                       cfg["fusion_dim"],
                                       cfg["dropout"] * 0.5)
        self.video_enc = _VideoEnc(cfg["fusion_dim"])
        self.dyne = _DynE(cfg["fusion_dim"], n_tokens=8,
                          n_heads=max(2, cfg["n_heads"] // 2),
                          dropout=cfg["dropout"])
        self.taavn = _TAAVN(text_hidden, cfg["fusion_dim"], cfg["fusion_dim"],
                            cfg["fusion_dim"], cfg["n_heads"], cfg["dropout"])
        self.mer_ml = _MERMLSE(cfg["fusion_dim"], cfg["dropout"])
        self.reasoner = _Reasoner(cfg["fusion_dim"], n_layers=4,
                                  n_heads=cfg["n_heads"], n_tokens=4,
                                  dropout=cfg["dropout"])
        self.head = nn.Sequential(
            nn.LayerNorm(cfg["fusion_dim"]), nn.Dropout(cfg["dropout"]),
            nn.Linear(cfg["fusion_dim"], cfg["fusion_dim"]//2), nn.GELU(),
            nn.Dropout(cfg["dropout"]),
            nn.Linear(cfg["fusion_dim"]//2, num_classes))

    def forward(self, video, audio, ids, mask):
        out = self.llm(input_ids=ids, attention_mask=mask,
                       output_hidden_states=True, return_dict=True)
        text_h = out.hidden_states[-1]
        a = self.audio_enc(audio)
        v = self.video_enc(video)
        v_seq = self.dyne(v)
        fused = self.taavn(text_h, mask, a, v_seq)
        refined, _ = self.mer_ml(fused)
        reasoned = self.reasoner(refined)
        return self.head(reasoned)


# ============================================================
# The thing your backend imports
# ============================================================
class EmotionPredictor:
    def __init__(self, deploy_dir, llama_dir=None, device=None,
                 video_timeout=10):
        self.deploy_dir = Path(deploy_dir)
        self.cfg = json.load(open(self.deploy_dir / "config.json"))
        self.id2label = {int(k): v for k, v in
                         json.load(open(self.deploy_dir / "label_map.json")).items()}
        self.classes = [self.id2label[i] for i in range(len(self.id2label))]
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.timeout = video_timeout
        if llama_dir is None:
            llama_dir = self.cfg.get("text_model", os.path.expanduser(
                "~/meld_emotion/models/llama3_2_3b"))
        self.llama_dir = llama_dir

        self.tokenizer = AutoTokenizer.from_pretrained(llama_dir)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = _EmotionModel(self.cfg, num_classes=len(self.classes),
                                   llama_dir=llama_dir).to(self.device)
        sd = torch.load(self.deploy_dir / "model.pt",
                        map_location=self.device, weights_only=False)
        self.model.load_state_dict(sd, strict=False)
        self.model.eval()

        # video transform (matches val transform during training)
        mean = [0.43216, 0.394666, 0.37645]
        std  = [0.22803, 0.22145, 0.216989]
        self.tf = transforms.Compose([
            transforms.Resize(128, antialias=True),
            transforms.CenterCrop(self.cfg["frame_size"]),
            transforms.Normalize(mean=mean, std=std),
        ])

        # audio feature extractor
        if HAS_TORCHAUDIO:
            self._mel = tat.MelSpectrogram(sample_rate=SAMPLE_RATE, n_fft=N_FFT,
                                           hop_length=HOP_LENGTH, n_mels=N_MELS)
            self._db = tat.AmplitudeToDB()
            self._mfcc = tat.MFCC(sample_rate=SAMPLE_RATE, n_mfcc=N_MFCC,
                                  melkwargs={"n_fft": N_FFT, "hop_length": HOP_LENGTH,
                                             "n_mels": N_MELS})

    # --------------------------
    def _read_video(self, path):
        T = self.cfg["num_frames"]
        signal.signal(signal.SIGALRM, _alarm)
        signal.alarm(self.timeout)
        try:
            if HAS_DECORD:
                vr = VideoReader(path, ctx=cpu(0))
                n = len(vr)
                if n == 0: raise RuntimeError("empty video")
                idx = np.linspace(0, n - 1, T).astype(int).tolist()
                frames = vr.get_batch(idx).asnumpy()
                return torch.from_numpy(frames).permute(0, 3, 1, 2).contiguous()
            else:
                f, _, _ = tvio.read_video(path, pts_unit="sec",
                                          output_format="TCHW")
                idx = np.linspace(0, f.shape[0] - 1, T).astype(int).tolist()
                return f[idx]
        finally:
            signal.alarm(0)

    def _read_audio(self, path):
        target_len = AUDIO_SECS * SAMPLE_RATE
        if not HAS_TORCHAUDIO:
            return torch.zeros(1, target_len)
        signal.signal(signal.SIGALRM, _alarm)
        signal.alarm(self.timeout)
        try:
            wav, sr = torchaudio.load(path)
            if wav.size(0) > 1:
                wav = wav.mean(0, keepdim=True)
            if sr != SAMPLE_RATE:
                wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
            if wav.size(1) >= target_len:
                wav = wav[:, :target_len]
            else:
                wav = F.pad(wav, (0, target_len - wav.size(1)))
            return wav
        finally:
            signal.alarm(0)

    def _audio_feat(self, wav):
        if not HAS_TORCHAUDIO:
            return torch.zeros(AUDIO_FEAT_DIM, AUDIO_TIME_STEPS)
        with torch.no_grad():
            m = self._db(self._mel(wav))
            c = self._mfcc(wav)
        feat = torch.cat([m, c], dim=1).squeeze(0)
        feat = (feat - feat.mean(-1, keepdim=True)) / (feat.std(-1, keepdim=True) + 1e-5)
        if feat.size(-1) < AUDIO_TIME_STEPS:
            feat = F.pad(feat, (0, AUDIO_TIME_STEPS - feat.size(-1)))
        else:
            feat = feat[..., :AUDIO_TIME_STEPS]
        return feat

    def _prep(self, video_path, text, audio_path=None):
        frames = self._read_video(video_path).float() / 255.0
        frames = self.tf(frames).permute(1, 0, 2, 3).unsqueeze(0)
        wav = self._read_audio(audio_path or video_path)
        afeat = self._audio_feat(wav).unsqueeze(0)
        tok = self.tokenizer(text or " ", truncation=True,
                             max_length=self.cfg["max_text_len"],
                             padding="max_length", return_tensors="pt")
        return (frames.to(self.device),
                afeat.to(self.device),
                tok["input_ids"].to(self.device),
                tok["attention_mask"].to(self.device))

    @torch.no_grad()
    def predict(self, video_path, text, audio_path=None):
        v, a, i, m = self._prep(video_path, text, audio_path)
        amp_device = self.device.split(":")[0] if ":" in self.device else self.device
        with torch.amp.autocast(device_type=amp_device,
                                dtype=torch.bfloat16,
                                enabled=self.device.startswith("cuda")):
            logits = self.model(v, a, i, m)
        probs = F.softmax(logits.float(), dim=-1)[0].cpu().numpy()
        idx = int(probs.argmax())
        return {
            "label":      self.id2label[idx],
            "confidence": float(probs[idx]),
            "scores":     {self.id2label[i]: float(probs[i])
                           for i in range(len(probs))},
        }

    @torch.no_grad()
    def predict_batch(self, items):
        return [self.predict(*it) if isinstance(it, tuple) else self.predict(**it)
                for it in items]
'''


README_TEMPLATE = '''# Emotion classifier — deploy bundle

Trained {date}. Best macro-F1 on val: **{best_f1:.4f}**.

Architecture: Llama-3.2-3B (4-bit + LoRA) text encoder + CNN-BiLSTM
audio encoder + r3d_18/DynE video encoder, fused via TA-AVN, refined
with MER-ML(SE), reasoned with a 4-layer transformer.

## Files
- `model.pt` — trained weights (state_dict)
- `config.json` — model config
- `label_map.json` — class id → emotion name
- `inference.py` — `EmotionPredictor` class

## Requirements (deploy machine)
```
pip install torch torchvision torchaudio transformers peft bitsandbytes accelerate decord
```
Llama-3.2-3B base weights must be at `~/meld_emotion/models/llama3_2_3b/`
(or pass `llama_dir=` to `EmotionPredictor`).

## FastAPI usage
```python
from fastapi import FastAPI, UploadFile, Form
from inference import EmotionPredictor
import shutil, tempfile

predictor = EmotionPredictor("./deploy")
app = FastAPI()

@app.post("/predict")
async def predict(video: UploadFile, text: str = Form(...)):
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        shutil.copyfileobj(video.file, f); path = f.name
    return predictor.predict(path, text)
```

## Notes
- The video file must contain audio. If your deployment has separate
  audio files, pass `audio_path=` to `predict()`.
- First inference is slow (~5s) while the LLM loads in 4-bit.
  Subsequent calls take ~300-500ms on GPU.
- Hard {timeout}s timeout on video/audio reads.
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=None)
    args = ap.parse_args()

    run_dir = Path(args.run_dir) if args.run_dir else latest_run_dir()
    log = setup_logging(run_dir, "export")
    log.info(f"run_dir: {run_dir}")

    best_path = run_dir / "checkpoints" / "best.pt"
    if not best_path.exists():
        log.error(f"no best.pt found at {best_path}")
        sys.exit(2)

    deploy = run_dir / "deploy"
    deploy.mkdir(exist_ok=True)

    log.info(f"loading {best_path}")
    ck = torch.load(best_path, map_location="cpu", weights_only=False)
    cfg = ck["config"]

    log.info("saving model.pt (state_dict only)")
    torch.save(ck["model"], deploy / "model.pt")

    deploy_cfg = {
        "lora_r":         cfg["lora_r"],
        "lora_alpha":     cfg["lora_alpha"],
        "lora_dropout":   cfg["lora_dropout"],
        "fusion_dim":     cfg["fusion_dim"],
        "n_heads":        cfg["n_heads"],
        "dropout":        cfg["dropout"],
        "num_frames":     cfg["num_frames"],
        "frame_size":     cfg["frame_size"],
        "max_text_len":   cfg["max_text_len"],
        "text_model":     cfg["text_model"],
    }
    save_json(deploy_cfg, deploy / "config.json")
    save_json({i: c for i, c in enumerate(CLASSES)}, deploy / "label_map.json")

    (deploy / "inference.py").write_text(INFERENCE_TEMPLATE)

    final = run_dir / "final_results.json"
    best_f1 = json.loads(final.read_text())["best_f1"] if final.exists() else 0.0
    (deploy / "README.md").write_text(
        README_TEMPLATE.format(
            date=__import__("time").strftime("%Y-%m-%d"),
            best_f1=best_f1, timeout=10,
        )
    )

    import os as _os
    log.info("\nDeploy bundle:")
    for p in sorted(deploy.iterdir()):
        log.info(f"  {p.name}: {_os.path.getsize(p)/1e6:.1f} MB")
    log.info(f"deploy bundle: {deploy}")


if __name__ == "__main__":
    main()
