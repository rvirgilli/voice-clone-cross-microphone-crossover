#!/usr/bin/env python3
"""Produce the three manipulated prompt sets of EXP-211 (see PREREG.md), on CPU.

For every speaker and arm, both microphone captures are processed so that the
attribution can be run against unmodified candidates and against candidates that
received the same transform. Outputs are 24 kHz mono WAV under
<run>/prompts/<condition>/<speaker>_<arm>_<mic>.wav, written atomically; existing
files are skipped.

Conditions:
  flat     WORLD resynthesis with F0 set to the speaker's A/B pair-mean voiced F0 (unvoiced
           frames unchanged) and the frame RMS envelope flattened to the utterance mean.
  stretch  uniform time-stretch of A and B to their pair-mean duration (rate = dur/mean).
  ltas     long-term spectral envelope equalised toward the A/B pair-mean envelope
           (smoothed log-magnitude LTAS difference applied as a static STFT gain).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import librosa
import numpy as np
import pyworld as pw
import soundfile as sf

import os

# Historical run layout: these scripts document what ran. Point the two roots at copies of
# the private run directories and the experiment tree; nothing here reads the release itself.
RUNS = Path(os.environ.get("ICASSP_RUNS", "/runs"))
EXPERIMENTS = Path(os.environ.get("ICASSP_EXPERIMENTS", "/experiments"))
HF = Path(os.environ.get("HF_HUB_CACHE", "~/.cache/huggingface/hub")).expanduser()
EXP205 = EXPERIMENTS / "EXP-205-f2-crossmic-crossover"
SR = 24000
CONDITIONS = ("flat", "stretch", "ltas")
N_FFT = 1024
HOP = 256


def load(path: str) -> np.ndarray:
    audio, sr = sf.read(path, dtype="float64", always_2d=True)
    audio = audio.mean(axis=1)
    if sr != SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SR)
    return audio


def rms_envelope(x: np.ndarray) -> np.ndarray:
    frames = librosa.util.frame(np.pad(x, (HOP, HOP)), frame_length=N_FFT, hop_length=HOP)
    return np.sqrt((frames ** 2).mean(axis=0) + 1e-12)


def flatten_energy(x: np.ndarray) -> np.ndarray:
    env = rms_envelope(x)
    target = float(np.sqrt((x ** 2).mean() + 1e-12))
    gain = np.clip(target / env, 0.25, 4.0)
    frame_times = np.arange(env.size) * HOP
    return x * np.interp(np.arange(x.size), frame_times, gain)


def world_flat(x: np.ndarray, target_f0: float) -> np.ndarray:
    f0, t = pw.harvest(x, SR)
    sp = pw.cheaptrick(x, f0, t, SR)
    ap = pw.d4c(x, f0, t, SR)
    f0_flat = np.where(f0 > 0, target_f0, 0.0)
    y = pw.synthesize(f0_flat, sp, ap, SR)
    return flatten_energy(y[: x.size] if y.size >= x.size else np.pad(y, (0, x.size - y.size)))


def voiced_mean_f0(x: np.ndarray) -> float:
    f0, _ = pw.harvest(x, SR)
    voiced = f0[f0 > 0]
    return float(voiced.mean()) if voiced.size else 0.0


def ltas_db(x: np.ndarray) -> np.ndarray:
    mag = np.abs(librosa.stft(x, n_fft=N_FFT, hop_length=HOP))
    return 20 * np.log10(mag.mean(axis=1) + 1e-9)


def smooth(v: np.ndarray, width: int = 9) -> np.ndarray:
    kernel = np.ones(width) / width
    return np.convolve(np.pad(v, (width // 2, width // 2), mode="edge"), kernel, mode="valid")


def equalise(x: np.ndarray, target_db: np.ndarray) -> np.ndarray:
    spec = librosa.stft(x, n_fft=N_FFT, hop_length=HOP)
    gain_db = np.clip(smooth(target_db) - smooth(ltas_db(x)), -20.0, 20.0)
    y = librosa.istft(spec * (10 ** (gain_db / 20))[:, None], hop_length=HOP, length=x.size)
    return y


def write(path: Path, y: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    peak = float(np.abs(y).max()) or 1.0
    if peak > 0.99:
        y = y / peak * 0.99
    tmp = path.with_suffix(".partial.wav")
    sf.write(tmp, y.astype(np.float32), SR, subtype="PCM_16")
    os.replace(tmp, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    manifest = json.loads((EXP205 / "selection-manifest.json").read_text(encoding="utf-8"))
    speakers = manifest["speakers"][: args.limit] if args.limit else manifest["speakers"]
    summary = {}
    for spk in speakers:
        sid = spk["speaker"]
        audio = {(arm, mic): load(spk["audio"][f"{arm}_{mic}"]["path"]) for arm in ("A", "B") for mic in ("mic1", "mic2")}
        # Pair-level targets are computed per microphone from the two events.
        for mic in ("mic1", "mic2"):
            a, b = audio[("A", mic)], audio[("B", mic)]
            f0_target = float(np.mean([v for v in (voiced_mean_f0(a), voiced_mean_f0(b)) if v > 0]))
            mean_duration = (a.size + b.size) / 2.0
            ltas_target = (ltas_db(a) + ltas_db(b)) / 2.0
            for arm, x in (("A", a), ("B", b)):
                outputs = {
                    "flat": lambda: world_flat(x, f0_target),
                    "stretch": lambda: librosa.effects.time_stretch(x.astype(np.float32), rate=x.size / mean_duration).astype(np.float64),
                    "ltas": lambda: equalise(x, ltas_target),
                }
                for cond, fn in outputs.items():
                    target = args.run / "prompts" / cond / f"{sid}_{arm}_{mic}.wav"
                    if target.exists():
                        continue
                    write(target, fn())
            summary[f"{sid}_{mic}"] = {"f0_target_hz": f0_target, "mean_duration_s": mean_duration / SR}
        print(f"[done] {sid}", flush=True)
    (args.run / "prompts" / "targets.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
