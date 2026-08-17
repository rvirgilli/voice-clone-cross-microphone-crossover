"""Resumable EXP-205 generation for F5, XTTS-v2 and CosyVoice2.

Run only through the frozen one-job driver and gpu-submit.  This module prints
counts and progress, never similarities or scientific outcomes.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_hash(path: Path, expected: str, label: str) -> None:
    observed = sha256(path)
    if observed != expected:
        raise RuntimeError(
            f"{label} hash mismatch expected={expected} observed={observed}"
        )


def verify_pin_records(records: list[dict[str, str]], label: str) -> None:
    if not records:
        raise RuntimeError(f"{label} has no pinned model files")
    for record in records:
        require_hash(Path(record["path"]), record["sha256"], label)


def require_pinned_loader_file(
    pins: dict, field: str, label: str, *, suffix: str | None = None
) -> Path:
    path = Path(pins[field])
    matches = [record for record in pins["files"] if record["path"] == str(path)]
    if len(matches) != 1:
        raise RuntimeError(f"{label} loader path is not the unique pinned logical file")
    if suffix is not None and path.suffix != suffix:
        raise RuntimeError(
            f"{label} loader path suffix mismatch expected={suffix} got={path.suffix!r}"
        )
    return path


def authenticate_reference(job: dict, authenticated: set[tuple[str, str]]) -> None:
    path = job["reference"].resolve()
    key = (str(path), job["reference_sha256"])
    if key not in authenticated:
        require_hash(path, job["reference_sha256"], "GENERATION_REFERENCE")
        authenticated.add(key)


def set_rng(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def valid_audio(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size <= 1_000:
        return False
    import soundfile as sf

    try:
        info = sf.info(path)
    except RuntimeError:
        return False
    return info.frames > 0 and info.samplerate > 0


def jobs(manifest: dict, run_root: Path, system: str):
    texts = manifest["generation"]["generated_texts"]
    seed_base = int(manifest["generation"]["rng_seed_base"])
    for speaker in manifest["speakers"]:
        speaker_id = speaker["speaker"]
        for prompt_mic in ("mic1", "mic2"):
            for arm in ("A", "B"):
                reference = Path(speaker["audio"][f"{arm}_{prompt_mic}"]["path"])
                reference_sha256 = speaker["audio"][f"{arm}_{prompt_mic}"]["sha256"]
                reference_text = speaker["transcripts"][arm]["text"]
                reference_text_sha256 = speaker["transcripts"][arm]["sha256"]
                for text in texts:
                    text_index = int(text["index"])
                    out = (
                        run_root
                        / "clones"
                        / f"{system}__{prompt_mic}__seed{arm}"
                        / f"{speaker_id}_t{text_index}.wav"
                    )
                    yield {
                        "speaker": speaker_id,
                        "prompt_mic": prompt_mic,
                        "arm": arm,
                        "reference": reference,
                        "reference_sha256": reference_sha256,
                        "reference_text": reference_text,
                        "reference_text_sha256": reference_text_sha256,
                        "text_index": text_index,
                        "generated_text": text["text"],
                        "generated_text_sha256": text["sha256_utf8"],
                        "seed": seed_base + text_index,
                        "out": out,
                    }


def atomic_target(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".partial.wav")
    if temporary.exists():
        temporary.unlink()
    return temporary


def finish_audio(temporary: Path, output: Path) -> None:
    if not valid_audio(temporary):
        raise RuntimeError(f"generator produced invalid audio: {temporary}")
    os.replace(temporary, output)


def ledger_path(output: Path) -> Path:
    return output.with_suffix(output.suffix + ".ledger.json")


def expected_ledger(job: dict, system: str, context: dict) -> dict:
    return {
        "schema": "exp205-clone-ledger-v1",
        "execution_config_sha256": context["execution_config_sha256"],
        "manifest_sha256": context["manifest_sha256"],
        "generate_source_sha256": context["generate_source_sha256"],
        "generator_pin_sha256": context["generator_pin_sha256"],
        "system": system,
        "speaker": job["speaker"],
        "prompt_mic": job["prompt_mic"],
        "seed_arm": job["arm"],
        "text_index": job["text_index"],
        "reference_path": str(job["reference"]),
        "reference_sha256": job["reference_sha256"],
        "reference_text_sha256": job["reference_text_sha256"],
        "generated_text_sha256_utf8": job["generated_text_sha256"],
        "seed": job["seed"],
        "output_path": str(job["out"]),
    }


def resumable(job: dict, system: str, context: dict) -> bool:
    output = job["out"]
    sidecar = ledger_path(output)
    if not valid_audio(output) or not sidecar.is_file():
        return False
    try:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    expected = expected_ledger(job, system, context)
    for key, value in expected.items():
        if payload.get(key) != value:
            return False
    return payload.get("clone_sha256") == sha256(output)


def seal_clone(job: dict, system: str, context: dict) -> None:
    payload = expected_ledger(job, system, context)
    payload["clone_sha256"] = sha256(job["out"])
    sidecar = ledger_path(job["out"])
    temporary = sidecar.with_suffix(sidecar.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, sidecar)


def run_f5(todo: list[dict], pins: dict, context: dict) -> None:
    version = importlib.metadata.version("f5-tts")
    if version != pins["package_version"]:
        raise RuntimeError(f"F5 package mismatch expected={pins['package_version']} got={version}")
    import f5_tts.api as f5_api
    import f5_tts.infer.utils_infer as f5_utils
    from f5_tts.api import F5TTS

    require_hash(Path(f5_api.__file__).resolve(), pins["api_source_sha256"], "F5_API")
    require_hash(
        Path(f5_utils.__file__).resolve(), pins["infer_source_sha256"], "F5_INFER"
    )
    verify_pin_records(pins["files"], "F5_MODEL")
    require_pinned_loader_file(
        pins, "checkpoint_path", "F5_CHECKPOINT", suffix=".safetensors"
    )
    tts = F5TTS(
        ckpt_file=pins["checkpoint_path"],
        vocoder_local_path=pins["vocoder_path"],
    )
    authenticated_references: set[tuple[str, str]] = set()
    for index, job in enumerate(todo, start=1):
        authenticate_reference(job, authenticated_references)
        if resumable(job, "f5", context):
            continue
        temporary = atomic_target(job["out"])
        set_rng(job["seed"])
        tts.infer(
            ref_file=str(job["reference"]),
            ref_text=job["reference_text"],
            gen_text=job["generated_text"],
            file_wave=str(temporary),
            seed=job["seed"],
            show_info=lambda *_args, **_kwargs: None,
        )
        finish_audio(temporary, job["out"])
        seal_clone(job, "f5", context)
        if index % 50 == 0:
            print(f"F5_PROGRESS {index}/{len(todo)}", flush=True)


def run_xtts(todo: list[dict], pins: dict, context: dict) -> None:
    version = importlib.metadata.version("coqui-tts")
    if version != pins["package_version"]:
        raise RuntimeError(f"XTTS package mismatch expected={pins['package_version']} got={version}")
    os.environ.setdefault("COQUI_TOS_AGREED", "1")
    import TTS.api as tts_api
    from TTS.api import TTS

    require_hash(Path(tts_api.__file__).resolve(), pins["api_source_sha256"], "XTTS_API")
    verify_pin_records(pins["files"], "XTTS_MODEL")
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")
    authenticated_references: set[tuple[str, str]] = set()
    for index, job in enumerate(todo, start=1):
        authenticate_reference(job, authenticated_references)
        if resumable(job, "xtts", context):
            continue
        temporary = atomic_target(job["out"])
        set_rng(job["seed"])
        tts.tts_to_file(
            text=job["generated_text"],
            speaker_wav=str(job["reference"]),
            language="en",
            file_path=str(temporary),
        )
        finish_audio(temporary, job["out"])
        seal_clone(job, "xtts", context)
        if index % 50 == 0:
            print(f"XTTS_PROGRESS {index}/{len(todo)}", flush=True)


def run_cosy(todo: list[dict], pins: dict, context: dict) -> None:
    import torch
    import torchaudio
    import cosyvoice.cli.cosyvoice as cosy_api
    from cosyvoice.cli.cosyvoice import CosyVoice2

    require_hash(Path(cosy_api.__file__).resolve(), pins["api_source_sha256"], "COSY_API")
    verify_pin_records(pins["files"], "COSY_MODEL")
    snapshot = Path(pins["snapshot_path"])
    if snapshot.name != pins["snapshot_revision"] or not snapshot.is_dir():
        raise RuntimeError("CosyVoice snapshot revision/path mismatch")
    tts = CosyVoice2(str(snapshot), load_jit=False, load_trt=False, fp16=True)
    authenticated_references: set[tuple[str, str]] = set()
    for index, job in enumerate(todo, start=1):
        authenticate_reference(job, authenticated_references)
        if resumable(job, "cosy", context):
            continue
        temporary = atomic_target(job["out"])
        set_rng(job["seed"])
        chunks = list(
            tts.inference_zero_shot(
                job["generated_text"],
                job["reference_text"],
                str(job["reference"]),
                stream=False,
            )
        )
        if len(chunks) != 1:
            raise RuntimeError(f"unexpected CosyVoice chunk count: {len(chunks)}")
        torchaudio.save(str(temporary), chunks[0]["tts_speech"], tts.sample_rate)
        finish_audio(temporary, job["out"])
        seal_clone(job, "cosy", context)
        if index % 50 == 0:
            print(f"COSY_PROGRESS {index}/{len(todo)}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--system", choices=("f5", "xtts", "cosy"), required=True)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    require_hash(
        Path(__file__).resolve(),
        config["source_pins"]["generate"]["sha256"],
        "GENERATE_SOURCE",
    )
    manifest_path = Path(config["manifest"]["path"])
    require_hash(manifest_path, config["manifest"]["sha256"], "MANIFEST")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "exp205-selection-manifest-v1":
        raise RuntimeError("manifest schema mismatch")
    todo = list(jobs(manifest, Path(config["run_root"]), args.system))
    if len(todo) != 864:
        raise RuntimeError(f"expected 864 jobs per TTS system, got {len(todo)}")
    print(f"{args.system.upper()}_START total={len(todo)}", flush=True)
    context = {
        "execution_config_sha256": sha256(args.config),
        "manifest_sha256": sha256(manifest_path),
        "generate_source_sha256": config["source_pins"]["generate"]["sha256"],
        "generator_pin_sha256": hashlib.sha256(
            json.dumps(
                config["generators"][args.system], sort_keys=True
            ).encode("utf-8")
        ).hexdigest(),
    }
    if args.system == "f5":
        run_f5(todo, config["generators"]["f5"], context)
    elif args.system == "xtts":
        run_xtts(todo, config["generators"]["xtts"], context)
    else:
        run_cosy(todo, config["generators"]["cosy"], context)
    missing = [
        str(job["out"])
        for job in todo
        if not resumable(job, args.system, context)
    ]
    if missing:
        raise RuntimeError(f"generation incomplete: {len(missing)} invalid/missing files")
    print(f"{args.system.upper()}_COMPLETE total={len(todo)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
