#!/usr/bin/env python3
"""Extract a frozen roster of readouts for every EXP-205 clone and real candidate.

Files: the 3,456 clones listed in the EXP-205 score table plus the 216 real captures
of the selection manifest (54 speakers x 2 events x 2 microphones). Each readout is
written once, as an .npz keyed by file path, so a resubmitted job resumes.

Readouts (roster fixed in PREREG.md before running):
  ecapa            SpeechBrain spkrec-ecapa-voxceleb embedding (the EXP-205 primary readout)
  wavlmsv_xvector  microsoft/wavlm-base-plus-sv x-vector head (the EXP-205 corroborating readout)
  wavlmsv_L00..L12 the same model's hidden states, mean-pooled over time, per layer
  wavlm_L00..L12   microsoft/wavlm-base-plus (no SV fine-tune), mean-pooled hidden states per layer
  w2v2_L00..L12    facebook/wav2vec2-base, mean-pooled hidden states per layer
  whisper_enc      openai/whisper-large-v3-turbo encoder output, mean over the unpadded frames
  ltas_mfcc        handcrafted: 80-band log-mel long-term average spectrum + 20 MFCC means (CPU)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess

import librosa
import numpy as np
import soundfile as sf
import torch


import os

# Historical run layout: these scripts document what ran. Point the two roots at copies of
# the private run directories and the experiment tree; nothing here reads the release itself.
RUNS = Path(os.environ.get("ICASSP_RUNS", "/runs"))
EXPERIMENTS = Path(os.environ.get("ICASSP_EXPERIMENTS", "/experiments"))
HF = Path(os.environ.get("HF_HUB_CACHE", "~/.cache/huggingface/hub")).expanduser()
EXP205 = EXPERIMENTS / "EXP-205-f2-crossmic-crossover"
RUN205 = RUNS / "EXP-205-f2-crossmic-crossover"
SNAPSHOT = {
    "ecapa": HF / "models--speechbrain--spkrec-ecapa-voxceleb/snapshots/0f99f2d0ebe89ac095bcc5903c4dd8f72b367286",
    "wavlmsv": HF / "models--microsoft--wavlm-base-plus-sv/snapshots/feb593a6c23c1cc3d9510425c29b0a14d2b07b1e",
    "wavlm": HF / "models--microsoft--wavlm-base-plus/snapshots/4c66d4806a428f2e922ccfa1a962776e232d487b",
    "w2v2": HF / "models--facebook--wav2vec2-base/snapshots/0b5b8e868dd84f03fd87d01f9c4ff0f080fecfe8",
    "whisper": HF / "models--openai--whisper-large-v3-turbo/snapshots/41f01f3fe87f28c78e2fbf8b568835947dd65ed9",
}
SR = 16000


def publish_safe_point(completed: int, total: int, out_dir: Path) -> None:
    """Publish only atomically completed readouts to gpu-queue v2."""
    if not os.environ.get("GPUQ_PAUSE_REQUEST_PATH"):
        return
    command = [
        shutil.which("gpuq2") or "gpuq2",
        "safe-point",
        "--current",
        str(completed),
        "--total",
        str(total),
        "--unit",
        "readout",
        "--phase",
        "extraction",
    ]
    if completed:
        command.extend(
            [
                "--checkpoint-id",
                f"readouts-{completed:02d}",
                "--checkpoint-durable",
            ]
        )
    result = subprocess.run(command, check=False, cwd=out_dir)
    if result.returncode:
        raise SystemExit(result.returncode)


def file_list() -> list[str]:
    paths = []
    with (RUN205 / "scores.tsv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            paths.append(str(RUN205 / row["clone_path"]))
    manifest = json.loads((EXP205 / "selection-manifest.json").read_text(encoding="utf-8"))
    for speaker in manifest["speakers"]:
        for key in ("A_mic1", "A_mic2", "B_mic1", "B_mic2"):
            paths.append(speaker["audio"][key]["path"])
    if len(paths) != 3456 + 216 or len(set(paths)) != len(paths):
        raise RuntimeError(f"unexpected file census: {len(paths)}")
    return paths


def load_16k(path: str) -> np.ndarray:
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if sr != SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SR)
    return audio.astype(np.float32)


def run_readout(name: str, paths: list[str], out_dir: Path, device: str) -> None:
    """Write out_dir/<name>.npz with arrays 'paths' and per-layer embedding matrices."""
    target = out_dir / f"{name}.npz"
    if target.exists():
        print(f"[skip] {name}: exists")
        return
    print(f"[run] {name}: {len(paths)} files", flush=True)
    layers: dict[str, list[np.ndarray]] = {}

    if name == "ecapa":
        from speechbrain.inference.speaker import EncoderClassifier
        model = EncoderClassifier.from_hparams(source=str(SNAPSHOT["ecapa"]), savedir=str(out_dir / "rt-ecapa"),
                                               run_opts={"device": device})
        for path in paths:
            wav = torch.from_numpy(load_16k(path)).unsqueeze(0).to(device)
            with torch.no_grad():
                emb = model.encode_batch(wav).squeeze().float().cpu().numpy()
            layers.setdefault("ecapa", []).append(emb)

    elif name in ("wavlmsv", "wavlm", "w2v2"):
        from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector, WavLMModel, Wav2Vec2Model
        src = str(SNAPSHOT[name])
        if name == "wavlmsv":
            model = WavLMForXVector.from_pretrained(src).to(device).eval()
            backbone = model.wavlm
        elif name == "wavlm":
            model = WavLMModel.from_pretrained(src).to(device).eval()
            backbone = model
        else:
            model = Wav2Vec2Model.from_pretrained(src).to(device).eval()
            backbone = model
        if (Path(src) / "preprocessor_config.json").is_file():
            extractor = Wav2Vec2FeatureExtractor.from_pretrained(src)
        else:
            # facebook/wav2vec2-base ships no preprocessor file in this cache; its published
            # preprocessor is 16 kHz, zero-mean unit-variance normalisation, no attention mask.
            extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=SR, do_normalize=True, return_attention_mask=False)
        for path in paths:
            inputs = extractor(load_16k(path), sampling_rate=SR, return_tensors="pt").to(device)
            with torch.no_grad():
                hidden = backbone(**inputs, output_hidden_states=True).hidden_states
                if name == "wavlmsv":
                    xvec = model(**inputs).embeddings.squeeze().float().cpu().numpy()
                    layers.setdefault("wavlmsv_xvector", []).append(xvec)
            for index, state in enumerate(hidden):
                layers.setdefault(f"{name}_L{index:02d}", []).append(state.mean(dim=1).squeeze().float().cpu().numpy())

    elif name == "whisper":
        from transformers import WhisperFeatureExtractor, WhisperModel
        src = str(SNAPSHOT["whisper"])
        model = WhisperModel.from_pretrained(src, torch_dtype=torch.float16).to(device).eval()
        extractor = WhisperFeatureExtractor.from_pretrained(src)
        for path in paths:
            audio = load_16k(path)
            features = extractor(audio, sampling_rate=SR, return_tensors="pt").input_features.to(device, torch.float16)
            with torch.no_grad():
                enc = model.encoder(features).last_hidden_state[0].float().cpu().numpy()
            # The encoder runs at 50 frames per second over a 30 s padded window; keep the real span.
            frames = max(1, min(enc.shape[0], int(round(len(audio) / SR * 50))))
            layers.setdefault("whisper_enc", []).append(enc[:frames].mean(axis=0))

    elif name == "ltas_mfcc":
        for path in paths:
            audio = load_16k(path)
            mel = librosa.feature.melspectrogram(y=audio, sr=SR, n_fft=1024, hop_length=256, n_mels=80)
            logmel = np.log(mel + 1e-8)
            mfcc = librosa.feature.mfcc(S=logmel, n_mfcc=20)
            layers.setdefault("ltas_mfcc", []).append(np.concatenate([logmel.mean(axis=1), mfcc.mean(axis=1)]).astype(np.float32))
    else:
        raise ValueError(name)

    payload = {key: np.stack(values).astype(np.float32) for key, values in layers.items()}
    tmp = target.with_suffix(".tmp.npz")
    np.savez(tmp, paths=np.array(paths), **payload)
    os.replace(tmp, target)
    print(f"[done] {name}: {sorted(payload)}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--readouts", nargs="+", default=["ecapa", "wavlmsv", "wavlm", "w2v2", "whisper", "ltas_mfcc"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit", type=int, default=0, help="smoke test: only the first N files")
    args = parser.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = args.device
    paths = file_list()
    if args.limit:
        paths = paths[: args.limit]
    (out_dir / "files.json").write_text(json.dumps(paths, indent=0), encoding="utf-8")
    completed = sum((out_dir / f"{name}.npz").is_file() for name in args.readouts)
    publish_safe_point(completed, len(args.readouts), out_dir)
    for name in args.readouts:
        existed = (out_dir / f"{name}.npz").is_file()
        run_readout(name, paths, out_dir, device)
        if not existed:
            completed += 1
        publish_safe_point(completed, len(args.readouts), out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
