import sys as _sys, os as _os; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__))); from audio_fix import load_audio as _load_audio
"""
model.py — Dataset, model architecture, loss, scheduler.

Architecture reconstruction (from summary.json: "TinyLlama(LoRA) +
CNN-BiLSTM + r3d_18/DynE > TA-AVN > MER-ML(SE) > Reasoner(4L)").

I (Claude) DID NOT see your original code. The non-standard pieces below
are best-effort guesses based on the names. Each guessed module is
marked with "# >>> GUESS:" comments so you can swap them for your real
implementations when you find them. The data flow and parameter counts
roughly match your reported 29.3M trainable params.

What's standard:
  - Llama-3.2-3B + LoRA  (was TinyLlama, swapped per your request)
  - r3d_18 backbone
  - mel/MFCC audio features

What's guessed:
  - DynE       — temporal attention over r3d_18 frame features
  - TA-AVN     — text-attended audio-visual cross-attention
  - MER-ML(SE) — multi-loss head with squeeze-excitation gating
  - Reasoner   — 4-layer transformer over fused features
"""
import math
import json
import signal
import random
import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import LambdaLR
from torchvision.models.video import r3d_18
from torchvision import transforms
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from common import Config, CLASSES, LABEL2ID, NUM_CLASSES, normalize_label

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

log = logging.getLogger("pipeline")

# ============================================================
# Audio constants
# ============================================================
SAMPLE_RATE = 16000
N_FFT       = 400
HOP_LENGTH  = 160     # 10ms @ 16kHz
N_MELS      = 64
N_MFCC      = 20
AUDIO_SECS  = 4
AUDIO_FEAT_DIM   = N_MELS + N_MFCC          # 84
AUDIO_TIME_STEPS = AUDIO_SECS * SAMPLE_RATE // HOP_LENGTH  # 400

# ============================================================
# Robust I/O (with hard timeout)
# ============================================================
class _Timeout(Exception): pass
def _alarm(s, f): raise _Timeout()


def read_video_safe(path: str, num_frames: int, is_train: bool,
                    timeout: int = 10) -> torch.Tensor:
    fb = torch.zeros(num_frames, 3, 128, 128, dtype=torch.uint8)
    if not Path(path).exists():
        return fb
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(timeout)
    try:
        if HAS_DECORD:
            vr = VideoReader(path, ctx=cpu(0))
            n = len(vr)
            if n == 0: return fb
            if is_train:
                seg = np.linspace(0, n, num_frames + 1).astype(int)
                idx = [random.randint(seg[i], max(seg[i], seg[i+1]-1))
                       for i in range(num_frames)]
            else:
                idx = np.linspace(0, n - 1, num_frames).astype(int).tolist()
            frames = vr.get_batch(idx).asnumpy()
            return torch.from_numpy(frames).permute(0, 3, 1, 2).contiguous()
        else:
            f, _, _ = tvio.read_video(path, pts_unit="sec",
                                      output_format="TCHW")
            n = f.shape[0]
            if n == 0: return fb
            idx = np.linspace(0, n - 1, num_frames).astype(int).tolist()
            return f[idx]
    except (_Timeout, Exception) as e:
        log.warning(f"  video read failed [{type(e).__name__}]: {Path(path).name}")
        return fb
    finally:
        signal.alarm(0)


def read_audio_safe(path: str, timeout: int = 10) -> torch.Tensor:
    """(1, samples) at SAMPLE_RATE, padded/clipped to AUDIO_SECS."""
    target_len = AUDIO_SECS * SAMPLE_RATE
    fb = torch.zeros(1, target_len)
    if not Path(path).exists() or not HAS_TORCHAUDIO:
        return fb
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(timeout)
    try:
        wav, sr = _load_audio(path)
        if wav.size(0) > 1:
            wav = wav.mean(0, keepdim=True)
        if sr != SAMPLE_RATE:
            wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
        if wav.size(1) >= target_len:
            wav = wav[:, :target_len]
        else:
            wav = F.pad(wav, (0, target_len - wav.size(1)))
        return wav
    except (_Timeout, Exception) as e:
        log.warning(f"  audio read failed [{type(e).__name__}]: {Path(path).name}")
        return fb
    finally:
        signal.alarm(0)


def make_audio_feat_extractor():
    if not HAS_TORCHAUDIO:
        return None
    mel = tat.MelSpectrogram(sample_rate=SAMPLE_RATE, n_fft=N_FFT,
                             hop_length=HOP_LENGTH, n_mels=N_MELS)
    db  = tat.AmplitudeToDB()
    mfcc = tat.MFCC(sample_rate=SAMPLE_RATE, n_mfcc=N_MFCC,
                    melkwargs={"n_fft": N_FFT, "hop_length": HOP_LENGTH,
                               "n_mels": N_MELS})
    def extract(wav: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            m = db(mel(wav))                             # (1, n_mels, T)
            c = mfcc(wav)                                # (1, n_mfcc, T)
        feat = torch.cat([m, c], dim=1).squeeze(0)       # (84, T)
        feat = (feat - feat.mean(-1, keepdim=True)) / \
               (feat.std(-1, keepdim=True) + 1e-5)
        if feat.size(-1) < AUDIO_TIME_STEPS:
            feat = F.pad(feat, (0, AUDIO_TIME_STEPS - feat.size(-1)))
        else:
            feat = feat[..., :AUDIO_TIME_STEPS]
        return feat                                       # (84, 400)
    return extract


# ============================================================
# Dataset
# ============================================================
class EmotionDataset(Dataset):
    """Loads video frames, audio features, and tokenized text per item.

    Audio is extracted from the video file unless an 'audio_path' field is
    given in the manifest item.
    """
    def __init__(self, manifest_path: str, tokenizer, cfg: Config,
                 is_train: bool, indices: Optional[np.ndarray] = None):
        with open(manifest_path) as f:
            items = list(json.load(f))
        keep = []
        for it in items:
            lab = normalize_label(it.get("emotion", ""))
            if lab not in LABEL2ID:
                continue
            it["_label_id"] = LABEL2ID[lab]
            keep.append(it)
        self.items = keep if indices is None else [keep[i] for i in indices]

        self.tokenizer = tokenizer
        self.cfg       = cfg
        self.is_train  = is_train
        self.labels    = np.array([it["_label_id"] for it in self.items],
                                  dtype=np.int64)

        mean = [0.43216, 0.394666, 0.37645]
        std  = [0.22803, 0.22145, 0.216989]
        if is_train:
            self.tf = transforms.Compose([
                transforms.Resize(128, antialias=True),
                transforms.RandomCrop(cfg.frame_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
                transforms.Normalize(mean=mean, std=std),
            ])
        else:
            self.tf = transforms.Compose([
                transforms.Resize(128, antialias=True),
                transforms.CenterCrop(cfg.frame_size),
                transforms.Normalize(mean=mean, std=std),
            ])
        self._audio_extract = None  # lazy init per worker

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it    = self.items[i]
        label = int(it["_label_id"])

        # --- video ---
        frames = read_video_safe(
            it["video_path"], self.cfg.num_frames,
            is_train=self.is_train, timeout=self.cfg.video_read_timeout,
        ).float() / 255.0
        frames = self.tf(frames).permute(1, 0, 2, 3)

        # --- audio ---
        if self._audio_extract is None:
            self._audio_extract = make_audio_feat_extractor()
        audio_path = it.get("audio_path", it["video_path"])
        wav = read_audio_safe(audio_path, timeout=self.cfg.video_read_timeout)
        if self._audio_extract is not None:
            audio_feat = self._audio_extract(wav)
        else:
            audio_feat = torch.zeros(AUDIO_FEAT_DIM, AUDIO_TIME_STEPS)

        # --- text ---
        tok = self.tokenizer(
            it.get("utterance", "") or " ",
            truncation=True, max_length=self.cfg.max_text_len,
            padding="max_length", return_tensors="pt",
        )
        return {
            "video":          frames,
            "audio":          audio_feat,
            "input_ids":      tok["input_ids"].squeeze(0),
            "attention_mask": tok["attention_mask"].squeeze(0),
            "label":          label,
        }


# ============================================================
# Audio encoder: CNN-BiLSTM
# ============================================================
class AudioEncoder(nn.Module):
    """1D CNN over (mel+MFCC) features → 2-layer BiLSTM → projection.
    Input:  (B, 84, T)   Output: (B, out_dim)
    """
    def __init__(self, in_dim: int = AUDIO_FEAT_DIM, hidden: int = 128,
                 out_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_dim, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128), nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128), nn.GELU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
        )
        self.bilstm = nn.LSTM(128, hidden, num_layers=2,
                              batch_first=True, bidirectional=True,
                              dropout=dropout)
        self.proj = nn.Linear(hidden * 2, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x):
        h = self.cnn(x)                # (B, 128, T/4)
        h = h.transpose(1, 2)          # (B, T/4, 128)
        out, _ = self.bilstm(h)        # (B, T/4, 256)
        pooled = out.mean(dim=1)       # (B, 256)
        return self.norm(self.proj(pooled))


# ============================================================
# Video encoder + DynE
# ============================================================
class VideoEncoder(nn.Module):
    def __init__(self, weights_path: str, out_dim: int):
        super().__init__()
        bb = r3d_18(weights=None)
        if weights_path and Path(weights_path).exists():
            bb.load_state_dict(torch.load(weights_path, map_location="cpu"))
            log.info(f"  r3d_18 loaded from {Path(weights_path).name}")
        in_feats = bb.fc.in_features
        bb.fc = nn.Identity()
        self.backbone = bb
        self.proj     = nn.Linear(in_feats, out_dim)

    def forward(self, x):
        return self.proj(self.backbone(x))   # (B, out_dim)


class DynE(nn.Module):
    """>>> GUESS: 'Dynamic Encoder' interpreted as temporal self-attention.
    r3d_18 outputs a single clip-level vector — we expand it with a
    learned token bank and self-attend so downstream attention has more
    than one position to attend to. If your real DynE operates on raw
    per-frame backbone features, swap this module."""
    def __init__(self, dim: int, n_tokens: int = 8, n_heads: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        self.tokens = nn.Parameter(torch.randn(1, n_tokens, dim) * 0.02)
        self.attn = nn.MultiheadAttention(dim, n_heads,
                                          dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(dim * 2, dim),
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, vid_feat):
        B = vid_feat.size(0)
        seq = torch.cat([vid_feat.unsqueeze(1),
                         self.tokens.expand(B, -1, -1)], dim=1)
        q = self.norm1(seq)
        a, _ = self.attn(q, q, q, need_weights=False)
        x = seq + a
        x = x + self.ffn(self.norm2(x))
        return x                                   # (B, 1+N, dim)


# ============================================================
# TA-AVN: Text-Attended Audio-Visual Network
# ============================================================
class TAAVN(nn.Module):
    """>>> GUESS: text tokens (Q) attend to concatenated [audio, video_dyne]
    (K, V). Returns pooled fused representation per sample."""
    def __init__(self, text_dim: int, audio_dim: int, vid_dim: int,
                 hidden: int, n_heads: int, dropout: float):
        super().__init__()
        self.text_proj  = nn.Linear(text_dim,  hidden)
        self.audio_proj = nn.Linear(audio_dim, hidden)
        self.vid_proj   = nn.Linear(vid_dim,   hidden)

        self.attn = nn.MultiheadAttention(hidden, n_heads,
                                          dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, hidden * 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden * 2, hidden),
        )
        self.norm2 = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text_h, text_mask, audio_feat, vid_seq):
        t = self.text_proj(text_h)                     # (B, L, H)
        a = self.audio_proj(audio_feat).unsqueeze(1)   # (B, 1, H)
        v = self.vid_proj(vid_seq)                     # (B, N_v, H)
        kv = torch.cat([t, a, v], dim=1)               # (B, L+1+N_v, H)
        B, L = text_mask.shape
        Nv = v.size(1)
        kp = torch.cat([
            text_mask == 0,
            torch.zeros(B, 1 + Nv, dtype=torch.bool,
                        device=text_h.device),
        ], dim=1)
        q = self.norm1(t)
        a_out, _ = self.attn(q, kv, kv, key_padding_mask=kp,
                             need_weights=False)
        x = t + self.dropout(a_out)
        x = x + self.dropout(self.ffn(self.norm2(x)))
        m = text_mask.unsqueeze(-1).float()
        return (x * m).sum(1) / m.sum(1).clamp(min=1)   # (B, H)


# ============================================================
# MER-ML(SE)
# ============================================================
class SEBlock(nn.Module):
    """Channel-wise squeeze-excitation."""
    def __init__(self, dim: int, ratio: int = 8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim, dim // ratio), nn.GELU(),
            nn.Linear(dim // ratio, dim), nn.Sigmoid(),
        )
    def forward(self, x):
        return x * self.fc(x)


class MERMLSE(nn.Module):
    """>>> GUESS: 'Multimodal Emotion Recognition - Multi-Loss with
    Squeeze-Excitation'. Refines fused features through SE gating and
    emits an auxiliary 2-d output used as a structural regularizer."""
    def __init__(self, dim: int, dropout: float = 0.2):
        super().__init__()
        self.se = SEBlock(dim, ratio=8)
        self.refine = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim),
            nn.GELU(), nn.Dropout(dropout),
        )
        self.aux = nn.Sequential(
            nn.Linear(dim, dim // 2), nn.GELU(),
            nn.Linear(dim // 2, 2),
        )

    def forward(self, x):
        gated = self.se(x)
        refined = self.refine(gated)
        aux_out = self.aux(refined)
        return refined, aux_out


# ============================================================
# Reasoner(4L)
# ============================================================
class Reasoner(nn.Module):
    """>>> GUESS: 4-layer transformer encoder over fused features. The
    fused vector is treated as a 1-token sequence, augmented with [CLS]
    + learned reasoning tokens; we read out the [CLS] position."""
    def __init__(self, dim: int, n_layers: int = 4, n_heads: int = 8,
                 n_tokens: int = 4, dropout: float = 0.1):
        super().__init__()
        self.cls    = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.tokens = nn.Parameter(torch.randn(1, n_tokens, dim) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=n_heads, dim_feedforward=dim * 4,
            dropout=dropout, activation="gelu", batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm    = nn.LayerNorm(dim)

    def forward(self, x):
        B = x.size(0)
        seq = torch.cat([self.cls.expand(B, -1, -1),
                         x.unsqueeze(1),
                         self.tokens.expand(B, -1, -1)], dim=1)
        out = self.encoder(seq)
        return self.norm(out[:, 0])      # (B, dim) — CLS readout


# ============================================================
# Full model
# ============================================================
class EmotionModel(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        log.info("Building model...")

        # ---- text: Llama-3.2-3B in 4-bit + LoRA ----
        bnb = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        log.info(f"  loading {Path(cfg.text_model).name} (4-bit)...")
        llm = AutoModelForCausalLM.from_pretrained(
            cfg.text_model, quantization_config=bnb,
            device_map={"": 0}, attn_implementation="sdpa",
        )
        llm = prepare_model_for_kbit_training(llm,
                                              use_gradient_checkpointing=True)
        llm.config.use_cache = False
        self.llm = get_peft_model(llm, LoraConfig(
            r=cfg.lora_r, lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout, bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            task_type="FEATURE_EXTRACTION",
        ))
        text_hidden = self.llm.config.hidden_size

        # ---- audio: CNN-BiLSTM ----
        self.audio_enc = AudioEncoder(
            in_dim=AUDIO_FEAT_DIM, hidden=128,
            out_dim=cfg.fusion_dim, dropout=cfg.dropout * 0.5,
        )

        # ---- video: r3d_18 + DynE ----
        self.video_enc = VideoEncoder(cfg.r3d_weights, out_dim=cfg.fusion_dim)
        self.dyne      = DynE(cfg.fusion_dim, n_tokens=8,
                              n_heads=max(2, cfg.n_heads // 2),
                              dropout=cfg.dropout)

        # ---- fusion: TA-AVN ----
        self.taavn = TAAVN(
            text_dim=text_hidden, audio_dim=cfg.fusion_dim,
            vid_dim=cfg.fusion_dim, hidden=cfg.fusion_dim,
            n_heads=cfg.n_heads, dropout=cfg.dropout,
        )

        # ---- MER-ML(SE) ----
        self.mer_ml = MERMLSE(cfg.fusion_dim, dropout=cfg.dropout)

        # ---- Reasoner(4L) ----
        self.reasoner = Reasoner(
            cfg.fusion_dim, n_layers=4, n_heads=cfg.n_heads,
            n_tokens=4, dropout=cfg.dropout,
        )

        # ---- classification head ----
        self.head = nn.Sequential(
            nn.LayerNorm(cfg.fusion_dim), nn.Dropout(cfg.dropout),
            nn.Linear(cfg.fusion_dim, cfg.fusion_dim // 2), nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.fusion_dim // 2, NUM_CLASSES),
        )

        total = sum(p.numel() for p in self.parameters())
        train_n = sum(p.numel() for p in self.parameters() if p.requires_grad)
        log.info(f"  total params: {total:,}  trainable: {train_n:,} "
                 f"({train_n/total*100:.1f}%)")

    def forward(self, video, audio, input_ids, attention_mask,
                return_aux: bool = False):
        out = self.llm(input_ids=input_ids, attention_mask=attention_mask,
                       output_hidden_states=True, return_dict=True)
        text_h = out.hidden_states[-1]

        a_feat = self.audio_enc(audio)
        v_feat = self.video_enc(video)
        v_seq  = self.dyne(v_feat)

        fused          = self.taavn(text_h, attention_mask, a_feat, v_seq)
        refined, aux   = self.mer_ml(fused)
        reasoned       = self.reasoner(refined)
        logits         = self.head(reasoned)

        if return_aux:
            return logits, aux
        return logits


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
        return -(self.alpha.view(1, -1) * focal * true * logp).sum(-1).mean()


class CombinedLoss(nn.Module):
    """Focal CE + small L2 regularizer on the MER-ML aux output."""
    def __init__(self, focal: FocalLoss, aux_weight: float = 0.05):
        super().__init__()
        self.focal = focal
        self.aux_weight = aux_weight

    def forward(self, logits, target, aux=None):
        loss = self.focal(logits, target)
        if aux is not None and self.aux_weight > 0:
            loss = loss + self.aux_weight * aux.pow(2).mean()
        return loss


# ============================================================
# Schedulers + builders
# ============================================================
def warmup_cosine(optimizer, warmup_steps, total_steps, min_ratio=0.05):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * prog))
    return LambdaLR(optimizer, lr_lambda)


def build_optimizer(model, cfg: Config):
    """Differential LRs: LoRA, video, audio, fusion+head."""
    lora_p, vid_p, audio_p, head_p = [], [], [], []
    for n, p in model.named_parameters():
        if not p.requires_grad: continue
        if "lora_" in n:                lora_p.append(p)
        elif n.startswith("video_enc"): vid_p.append(p)
        elif n.startswith("audio_enc"): audio_p.append(p)
        else:                           head_p.append(p)
    groups = []
    if lora_p:  groups.append({"params": lora_p,  "lr": cfg.lr_lora})
    if vid_p:   groups.append({"params": vid_p,   "lr": cfg.lr_video})
    if audio_p: groups.append({"params": audio_p, "lr": cfg.lr_video * 5})
    if head_p:  groups.append({"params": head_p,  "lr": cfg.lr_head})
    return torch.optim.AdamW(groups, weight_decay=cfg.weight_decay,
                             betas=(0.9, 0.95))


def build_loaders(train_ds, val_ds, cfg: Config):
    counts = np.bincount(train_ds.labels, minlength=NUM_CLASSES)
    cls_w  = 1.0 / np.maximum(counts, 1)
    samp_w = cls_w[train_ds.labels]
    sampler = WeightedRandomSampler(
        torch.as_tensor(samp_w, dtype=torch.double),
        num_samples=len(train_ds), replacement=True,
    )
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, sampler=sampler,
        num_workers=cfg.workers, pin_memory=True,
        persistent_workers=cfg.workers > 0, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.workers, pin_memory=True,
        persistent_workers=cfg.workers > 0,
    )
    return train_loader, val_loader, counts, cls_w


def make_focal(cls_w, cfg: Config, device):
    alpha = cls_w / cls_w.sum() * NUM_CLASSES
    return FocalLoss(torch.as_tensor(alpha), gamma=cfg.focal_gamma,
                     label_smoothing=cfg.label_smoothing).to(device)


def make_loss(cls_w, cfg: Config, device):
    return CombinedLoss(make_focal(cls_w, cfg, device),
                        aux_weight=0.05).to(device)


def get_tokenizer(cfg: Config):
    tok = AutoTokenizer.from_pretrained(cfg.text_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok
