"""
Drop-in audio loader that uses ffmpeg subprocess when torchaudio fails.
Import this and call load_audio(path) instead of torchaudio.load(path).
"""
import subprocess, numpy as np, torch, os

FFMPEG = os.path.expanduser("~/meld_emotion/bin/ffmpeg")

def load_audio(path, sr=16000):
    """Load audio via ffmpeg, return (waveform [1,T], sr)."""
    try:
        import torchaudio
        wav, orig_sr = torchaudio.load(path)
        if orig_sr != sr:
            import torchaudio.functional as F
            wav = F.resample(wav, orig_sr, sr)
        return wav, sr
    except Exception:
        pass
    # fallback: ffmpeg → raw pcm
    try:
        cmd = [FFMPEG, "-v", "quiet", "-i", path,
               "-f", "f32le", "-ac", "1", "-ar", str(sr), "pipe:1"]
        raw = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
        if len(raw) == 0:
            raise ValueError("empty")
        wav = torch.from_numpy(np.frombuffer(raw, dtype=np.float32).copy()).unsqueeze(0)
        return wav, sr
    except Exception:
        # silent fallback — return zeros (4 sec)
        return torch.zeros(1, sr * 4), sr
