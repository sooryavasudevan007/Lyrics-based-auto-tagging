"""MFCC and spectral summaries from the original Music4All clips.

FFmpeg decodes at most 30 seconds to mono 22,050 Hz. A Hann STFT (2048/512),
40 HTK mel bands and a type-II orthonormal DCT yield 20 MFCCs. We report
means/stds over frames, plus six spectral/time-domain descriptors and duration.
No Spotify metadata is used as a substitute for waveform features.
"""
from functools import lru_cache
import hashlib
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pandas as pd
from scipy.fft import dct
from scipy.signal import stft

SAMPLE_RATE = 22050
N_FFT = 2048
HOP = 512
AUDIO_FEATURE_NAMES = [f"mfcc_{i:02d}_{stat}" for i in range(20) for stat in ("mean", "std")]
AUDIO_FEATURE_NAMES += [f"{name}_{stat}" for name in
                        ("centroid", "bandwidth", "rolloff85", "flatness", "rms", "zcr")
                        for stat in ("mean", "std")]
AUDIO_FEATURE_NAMES += ["duration_seconds"]


def fixed_subset(splits, n=10000, seed=575):
    """Freeze a song-ID-only sample with proportional train/val/test quotas."""
    if n <= 0 or n > len(splits) or splits.id.duplicated().any():
        raise ValueError("Subset size must be positive, fit the dataset, and use unique song IDs")
    if not set(splits.split).issubset({"train", "val", "test"}):
        raise ValueError("Unexpected split name")
    leakage = splits.assign(_artist=splits.artist.str.strip().str.casefold()).groupby("_artist").split.nunique()
    if (leakage > 1).any():
        raise ValueError("Artist overlap in frozen splits")
    counts = splits.split.value_counts().reindex(["train", "val", "test"], fill_value=0)
    raw = n * counts / len(splits)
    quotas = np.floor(raw).astype(int)
    for split in (raw - quotas).sort_values(ascending=False, kind="stable").index[:n - quotas.sum()]:
        quotas[split] += 1
    parts = []
    for split, quota in quotas.items():
        group = splits.loc[splits.split == split, ["id", "artist", "split"]].copy()
        group["selection_hash"] = group.id.map(lambda sid: hashlib.sha256(f"{seed}:{sid}".encode()).hexdigest())
        parts.append(group.sort_values(["selection_hash", "id"]).head(quota))
    return pd.concat(parts, ignore_index=True).sort_values(["split", "id"]).reset_index(drop=True)


def decode_audio(path):
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("FFmpeg is required to decode Music4All MP3 clips")
    result = subprocess.run([ffmpeg, "-v", "error", "-i", str(Path(path)), "-t", "30",
                             "-ar", str(SAMPLE_RATE), "-ac", "1", "-f", "f32le", "pipe:1"],
                            capture_output=True, check=False, timeout=60)
    if result.returncode:
        raise ValueError(f"Audio decode failed for {Path(path).name}: {result.stderr.decode(errors='replace')[:200]}")
    samples = np.frombuffer(result.stdout, dtype="<f4").copy()
    if len(samples) < SAMPLE_RATE or not np.isfinite(samples).all():
        raise ValueError(f"Invalid or shorter-than-one-second audio: {Path(path).name}")
    return samples


@lru_cache(maxsize=1)
def mel_filterbank():
    mel = np.linspace(0, 2595 * np.log10(1 + (SAMPLE_RATE / 2) / 700), 42)
    hz = 700 * (10 ** (mel / 2595) - 1)
    freq = np.fft.rfftfreq(N_FFT, 1 / SAMPLE_RATE)
    bank = np.zeros((40, len(freq)))
    for i in range(40):
        bank[i] = np.maximum(0, np.minimum((freq - hz[i]) / (hz[i + 1] - hz[i]),
                                          (hz[i + 2] - freq) / (hz[i + 2] - hz[i + 1])))
    return bank


def summarize_waveform(samples):
    samples = np.asarray(samples, dtype=np.float64)
    if samples.ndim != 1 or len(samples) < SAMPLE_RATE or not np.isfinite(samples).all():
        raise ValueError("Expected a finite mono waveform of at least one second")
    frequencies, _, spectrum = stft(samples, fs=SAMPLE_RATE, nperseg=N_FFT,
                                    noverlap=N_FFT - HOP, boundary=None, padded=False)
    magnitude = np.abs(spectrum)
    power = magnitude ** 2
    eps = 1e-12
    mfcc = dct(np.log(np.maximum(mel_filterbank() @ power, eps)), type=2, axis=0, norm="ortho")[:20]
    mass = magnitude.sum(axis=0)
    centroid = (frequencies[:, None] * magnitude).sum(axis=0) / np.maximum(mass, eps)
    bandwidth = np.sqrt(((frequencies[:, None] - centroid) ** 2 * magnitude).sum(axis=0) / np.maximum(mass, eps))
    crossing = np.argmax(np.cumsum(magnitude, axis=0) >= .85 * mass, axis=0)
    rolloff = frequencies[crossing]
    flatness = np.exp(np.log(np.maximum(power, eps)).mean(axis=0)) / np.maximum(power.mean(axis=0), eps)
    frames = np.lib.stride_tricks.sliding_window_view(samples, N_FFT)[::HOP]
    rms = np.sqrt((frames ** 2).mean(axis=1))
    zcr = (np.signbit(frames[:, 1:]) != np.signbit(frames[:, :-1])).mean(axis=1)
    arrays = [*mfcc, centroid, bandwidth, rolloff, flatness, rms, zcr]
    values = np.array([v for a in arrays for v in (np.mean(a), np.std(a))] + [len(samples) / SAMPLE_RATE], np.float32)
    if values.shape != (len(AUDIO_FEATURE_NAMES),) or not np.isfinite(values).all():
        raise ValueError("Invalid audio feature vector")
    return values


def find_clip(audio_dir, sid):
    for suffix in (".mp3", ".wav", ".flac", ".ogg"):
        path = Path(audio_dir) / f"{sid}{suffix}"
        if path.is_file():
            return path
    return None
