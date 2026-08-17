"""Authenticated, per-clone resumable Seed-VC generation for EXP-205.

The project-owned loop keeps the Seed-VC models warm while sealing an ancestry
ledger immediately after every clone.  An audio file without its exact ledger
is never accepted as resumable work.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile

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


def require_record(record: dict[str, str], label: str) -> None:
    path = Path(record["path"])
    try:
        observed_resolved = str(path.resolve(strict=True))
    except OSError as exc:
        raise RuntimeError(f"{label} logical path is unavailable: {path}") from exc
    if observed_resolved != record.get("resolved_path"):
        raise RuntimeError(
            f"{label} resolved target mismatch expected={record.get('resolved_path')} "
            f"observed={observed_resolved}"
        )
    require_hash(path, record["sha256"], label)


def authenticate_seedvc_runtime(pins: dict) -> None:
    require_record(pins["inference_source"], "SEEDVC_INFERENCE")
    require_record(pins["dit_checkpoint"], "SEEDVC_DIT_CHECKPOINT")
    require_record(pins["dit_config"], "SEEDVC_DIT_CONFIG")
    for record in pins["files"]:
        require_record(record, "SEEDVC_MODEL_OR_REF")


def valid_audio(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size <= 1_000:
        return False
    import soundfile as sf

    try:
        info = sf.info(path)
    except RuntimeError:
        return False
    return info.frames > 0 and info.samplerate > 0


def set_rng(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def jobs(manifest: dict, run_root: Path):
    sources = {
        int(row["index"]): row for row in manifest["generation"]["seedvc_sources"]
    }
    texts = {
        int(row["index"]): row for row in manifest["generation"]["generated_texts"]
    }
    if set(sources) != set(texts) or set(sources) != set(range(4)):
        raise RuntimeError("Seed-VC source/text index census mismatch")
    seed_base = int(manifest["generation"]["rng_seed_base"])
    for index in range(4):
        source = sources[index]
        text = texts[index]
        if source["transcript"] != text["text"]:
            raise RuntimeError(f"Seed-VC source/text mismatch at index {index}")
        if source["generated_text_sha256_utf8"] != text["sha256_utf8"]:
            raise RuntimeError(f"Seed-VC source/text hash mismatch at index {index}")
    for speaker in manifest["speakers"]:
        speaker_id = speaker["speaker"]
        for prompt_mic in ("mic1", "mic2"):
            for arm in ("A", "B"):
                target = speaker["audio"][f"{arm}_{prompt_mic}"]
                for index in range(4):
                    source = sources[index]
                    out = (
                        run_root
                        / "clones"
                        / f"seedvc__{prompt_mic}__seed{arm}"
                        / f"{speaker_id}_t{index}.wav"
                    )
                    yield {
                        "speaker": speaker_id,
                        "prompt_mic": prompt_mic,
                        "arm": arm,
                        "text_index": index,
                        "source": Path(source["path"]),
                        "source_sha256": source["sha256"],
                        "source_transcript_sha256": source["transcript_sha256"],
                        "generated_text_sha256": source[
                            "generated_text_sha256_utf8"
                        ],
                        "reference": Path(target["path"]),
                        "reference_sha256": target["sha256"],
                        "seed": seed_base + index,
                        "out": out,
                    }


def ledger_path(output: Path) -> Path:
    return output.with_suffix(output.suffix + ".ledger.json")


def expected_ledger(job: dict, context: dict) -> dict:
    return {
        "schema": "exp205-clone-ledger-v1",
        "execution_config_sha256": context["execution_config_sha256"],
        "manifest_sha256": context["manifest_sha256"],
        "generate_source_sha256": context["generate_source_sha256"],
        "generator_pin_sha256": context["generator_pin_sha256"],
        "system": "seedvc",
        "speaker": job["speaker"],
        "prompt_mic": job["prompt_mic"],
        "seed_arm": job["arm"],
        "text_index": job["text_index"],
        "reference_path": str(job["reference"]),
        "reference_sha256": job["reference_sha256"],
        "source_path": str(job["source"]),
        "source_sha256": job["source_sha256"],
        "source_transcript_sha256": job["source_transcript_sha256"],
        "generated_text_sha256_utf8": job["generated_text_sha256"],
        "seed": job["seed"],
        "output_path": str(job["out"]),
    }


def resumable(job: dict, context: dict) -> bool:
    output = job["out"]
    sidecar = ledger_path(output)
    if not valid_audio(output) or not sidecar.is_file():
        return False
    try:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if any(payload.get(key) != value for key, value in expected_ledger(job, context).items()):
        return False
    return payload.get("clone_sha256") == sha256(output)


def seal(job: dict, context: dict) -> None:
    payload = expected_ledger(job, context)
    payload["clone_sha256"] = sha256(job["out"])
    sidecar = ledger_path(job["out"])
    temporary = sidecar.with_suffix(sidecar.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, sidecar)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    require_hash(
        Path(__file__).resolve(),
        config["source_pins"]["generate_seedvc"]["sha256"],
        "GENERATE_SEEDVC_SOURCE",
    )
    manifest_path = Path(config["manifest"]["path"])
    require_hash(manifest_path, config["manifest"]["sha256"], "MANIFEST")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "exp205-selection-manifest-v1":
        raise RuntimeError("selection manifest schema mismatch")

    pins = config["generators"]["seedvc"]
    authenticate_seedvc_runtime(pins)

    import inference

    require_hash(
        Path(inference.__file__).resolve(),
        pins["inference_source"]["sha256"],
        "SEEDVC_IMPORTED_INFERENCE",
    )
    models = None
    original_load = inference.load_models

    def cached_load(namespace):
        nonlocal models
        if models is None:
            models = original_load(namespace)
            # Recheck logical snapshot/ref mappings after the repository loader
            # has instantiated every model.  The runner is offline, so no hub
            # update is allowed between the two authentication boundaries.
            authenticate_seedvc_runtime(pins)
        return models

    inference.load_models = cached_load
    context = {
        "execution_config_sha256": sha256(args.config),
        "manifest_sha256": sha256(manifest_path),
        "generate_source_sha256": config["source_pins"]["generate_seedvc"][
            "sha256"
        ],
        "generator_pin_sha256": hashlib.sha256(
            json.dumps(pins, sort_keys=True).encode("utf-8")
        ).hexdigest(),
    }
    todo = list(jobs(manifest, Path(config["run_root"])))
    if len(todo) != 864:
        raise RuntimeError(f"expected 864 Seed-VC jobs, got {len(todo)}")

    def make_args(job: dict, outdir: Path):
        return argparse.Namespace(
            source=str(job["source"]),
            target=str(job["reference"]),
            output=str(outdir),
            diffusion_steps=25,
            length_adjust=1.0,
            inference_cfg_rate=0.7,
            f0_condition=False,
            auto_f0_adjust=False,
            semi_tone_shift=0,
            checkpoint=pins["dit_checkpoint"]["path"],
            config=pins["dit_config"]["path"],
            fp16=True,
        )

    for index, job in enumerate(todo, start=1):
        require_hash(job["source"], job["source_sha256"], "SEEDVC_CONTENT")
        require_hash(job["reference"], job["reference_sha256"], "SEEDVC_REFERENCE")
        if resumable(job, context):
            continue
        job["out"].parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            dir=job["out"].parent, prefix=f".{job['out'].stem}.partial."
        ) as temporary_dir:
            set_rng(job["seed"])
            inference.main(make_args(job, Path(temporary_dir)))
            produced = sorted(Path(temporary_dir).glob("*.wav"))
            if len(produced) != 1 or not valid_audio(produced[0]):
                raise RuntimeError(
                    f"Seed-VC produced {len(produced)} valid candidates for {job['out']}"
                )
            os.replace(produced[0], job["out"])
        seal(job, context)
        if index % 50 == 0:
            print(f"SEEDVC_PROGRESS {index}/{len(todo)}", flush=True)
    missing = [str(job["out"]) for job in todo if not resumable(job, context)]
    if missing:
        raise RuntimeError(f"Seed-VC generation incomplete: {len(missing)}")
    print(f"SEEDVC_COMPLETE total={len(todo)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
