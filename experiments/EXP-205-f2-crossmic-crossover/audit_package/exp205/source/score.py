"""Resumable authenticated ECAPA/WavLM scoring for EXP-205.

This stage verifies the complete clone census, extracts only the two declared
SV readouts, writes the 3,456-row similarity table and execution receipt, and
constructs the final hash-pinned analyzer config.  It prints no similarity,
follow rate, interval, or verdict.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


SYSTEMS = ("f5", "xtts", "cosy", "seedvc")
PROMPT_MICS = ("mic1", "mic2")
ARMS = ("A", "B")
SCORE_HEADER = (
    "speaker",
    "system",
    "text_index",
    "prompt_mic",
    "seed_arm",
    "clone_path",
    "clone_sha256",
    "candidate_A_path",
    "candidate_A_sha256",
    "candidate_B_path",
    "candidate_B_sha256",
    "same_candidate_A_path",
    "same_candidate_A_sha256",
    "same_candidate_B_path",
    "same_candidate_B_sha256",
    "ecapa_A",
    "ecapa_B",
    "wavlm_A",
    "wavlm_B",
    "ecapa_same_A",
    "ecapa_same_B",
    "wavlm_same_A",
    "wavlm_same_B",
)


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
        raise RuntimeError(f"{label} has no pinned files")
    for record in records:
        require_hash(Path(record["path"]), record["sha256"], label)


def expected_clone_rows(manifest: dict[str, Any], run_root: Path):
    for speaker in manifest["speakers"]:
        speaker_id = speaker["speaker"]
        for system in SYSTEMS:
            for text in manifest["generation"]["generated_texts"]:
                text_index = int(text["index"])
                for prompt_mic in PROMPT_MICS:
                    for arm in ARMS:
                        path = (
                            run_root
                            / "clones"
                            / f"{system}__{prompt_mic}__seed{arm}"
                            / f"{speaker_id}_t{text_index}.wav"
                        )
                        yield {
                            "speaker": speaker_id,
                            "system": system,
                            "text_index": text_index,
                            "prompt_mic": prompt_mic,
                            "seed_arm": arm,
                            "path": path,
                        }


def exact_clone_census(
    manifest: dict[str, Any], run_root: Path
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    rows = list(expected_clone_rows(manifest, run_root))
    expected = {row["path"].resolve() for row in rows}
    observed = {path.resolve() for path in (run_root / "clones").glob("**/*.wav")}
    if expected != observed:
        raise RuntimeError(
            f"clone census mismatch missing={len(expected-observed)} extra={len(observed-expected)}"
        )
    hashes = {}
    import soundfile as sf

    for index, path in enumerate(sorted(expected), start=1):
        if path.stat().st_size <= 1_000:
            raise RuntimeError(f"clone too small: {path}")
        info = sf.info(path)
        if info.frames <= 0 or info.samplerate <= 0:
            raise RuntimeError(f"invalid clone audio: {path}")
        hashes[str(path)] = sha256(path)
        if index % 500 == 0:
            print(f"CLONE_AUTH_PROGRESS {index}/{len(expected)}", flush=True)
    return rows, hashes


def verify_clone_ledgers(
    clone_rows: list[dict[str, Any]],
    clone_hashes: dict[str, str],
    manifest: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, dict[str, str]]:
    """Bind every clone to the exact prospective generation inputs."""
    manifest_by_speaker = {row["speaker"]: row for row in manifest["speakers"]}
    texts = {
        int(row["index"]): row
        for row in manifest["generation"]["generated_texts"]
    }
    seedvc_sources = {
        int(row["index"]): row
        for row in manifest["generation"]["seedvc_sources"]
    }
    execution_config_sha256 = sha256(Path(config["_config_path"]))
    manifest_sha256 = sha256(Path(config["manifest"]["path"]))
    result = {}
    for row in clone_rows:
        output = row["path"].resolve()
        ledger_path = output.with_suffix(output.suffix + ".ledger.json")
        try:
            ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"invalid/missing clone ledger: {ledger_path}") from exc
        speaker = manifest_by_speaker[row["speaker"]]
        reference = speaker["audio"][f"{row['seed_arm']}_{row['prompt_mic']}"]
        system = row["system"]
        source_pin_key = "generate_seedvc" if system == "seedvc" else "generate"
        generator_pin_sha256 = hashlib.sha256(
            json.dumps(config["generators"][system], sort_keys=True).encode("utf-8")
        ).hexdigest()
        expected = {
            "schema": "exp205-clone-ledger-v1",
            "execution_config_sha256": execution_config_sha256,
            "manifest_sha256": manifest_sha256,
            "generate_source_sha256": config["source_pins"][source_pin_key][
                "sha256"
            ],
            "generator_pin_sha256": generator_pin_sha256,
            "system": system,
            "speaker": row["speaker"],
            "prompt_mic": row["prompt_mic"],
            "seed_arm": row["seed_arm"],
            "text_index": row["text_index"],
            "reference_path": reference["path"],
            "reference_sha256": reference["sha256"],
            "generated_text_sha256_utf8": texts[row["text_index"]]["sha256_utf8"],
            "seed": int(manifest["generation"]["rng_seed_base"])
            + row["text_index"],
            "output_path": str(output),
            "clone_sha256": clone_hashes[str(output)],
        }
        if system != "seedvc":
            expected["reference_text_sha256"] = speaker["transcripts"][
                row["seed_arm"]
            ]["sha256"]
        else:
            source = seedvc_sources[row["text_index"]]
            expected.update(
                {
                    "source_path": source["path"],
                    "source_sha256": source["sha256"],
                    "source_transcript_sha256": source["transcript_sha256"],
                }
            )
        if any(ledger.get(key) != value for key, value in expected.items()):
            mismatched = [
                key for key, value in expected.items() if ledger.get(key) != value
            ]
            raise RuntimeError(
                f"clone ledger ancestry mismatch path={ledger_path} fields={mismatched}"
            )
        result[str(output)] = {
            "path": str(ledger_path),
            "sha256": sha256(ledger_path),
        }
    if len(result) != 3456:
        raise RuntimeError(f"expected 3,456 authenticated ledgers, got {len(result)}")
    return result


def cache_key(path: Path) -> str:
    return hashlib.sha256(str(path.resolve()).encode("utf-8")).hexdigest()[:24]


def load_cached(
    cache_path: Path, *, source_path: Path, source_hash: str, readout_pin: str
) -> tuple[np.ndarray, np.ndarray] | None:
    if not cache_path.is_file():
        return None
    try:
        payload = np.load(cache_path, allow_pickle=False)
        if str(payload["source_path"]) != str(source_path):
            return None
        if str(payload["source_sha256"]) != source_hash:
            return None
        if str(payload["readout_pin"]) != readout_pin:
            return None
        ecapa = np.asarray(payload["ecapa"], dtype=np.float32)
        wavlm = np.asarray(payload["wavlm"], dtype=np.float32)
    except (OSError, ValueError, KeyError):
        return None
    if ecapa.ndim != 1 or wavlm.ndim != 1 or not np.isfinite(ecapa).all() or not np.isfinite(wavlm).all():
        return None
    return ecapa, wavlm


def save_cached(
    cache_path: Path,
    *,
    source_path: Path,
    source_hash: str,
    readout_pin: str,
    ecapa: np.ndarray,
    wavlm: np.ndarray,
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(
            handle,
            source_path=str(source_path),
            source_sha256=source_hash,
            readout_pin=readout_pin,
            ecapa=np.asarray(ecapa, dtype=np.float32),
            wavlm=np.asarray(wavlm, dtype=np.float32),
        )
    os.replace(temporary, cache_path)


def extract_embeddings(
    paths: list[Path],
    hashes: dict[str, str],
    cache_dir: Path,
    config: dict[str, Any],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    readouts = config["readouts"]
    verify_pin_records(readouts["ecapa"]["files"], "ECAPA_MODEL")
    verify_pin_records(readouts["wavlm"]["files"], "WAVLM_MODEL")

    # Authenticate the exact loader implementations before accepting even a
    # fully populated feature cache.  Otherwise arbitrary cached vectors could
    # bypass the declared readout implementations.
    import speechbrain.inference.speaker as speechbrain_speaker
    import transformers.models.wav2vec2.feature_extraction_wav2vec2 as wav2vec_features
    import transformers.models.wavlm.modeling_wavlm as wavlm_source

    actual_loaders = {
        "ecapa_speaker": sha256(Path(speechbrain_speaker.__file__).resolve()),
        "wavlm_model": sha256(Path(wavlm_source.__file__).resolve()),
        "wavlm_feature_extractor": sha256(Path(wav2vec_features.__file__).resolve()),
    }
    expected_loaders = {
        **readouts["ecapa"]["loader_sha256"],
        **readouts["wavlm"]["loader_sha256"],
    }
    if actual_loaders != expected_loaders:
        raise RuntimeError(
            f"readout loader mismatch expected={expected_loaders} observed={actual_loaders}"
        )
    readout_pin = hashlib.sha256(
        json.dumps(
            {"readouts": readouts, "score_source": config["source_pins"]["score"]},
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()

    cached = {}
    missing = []
    for path in paths:
        cache_path = cache_dir / f"{cache_key(path)}.npz"
        value = load_cached(
            cache_path,
            source_path=path,
            source_hash=hashes[str(path)],
            readout_pin=readout_pin,
        )
        if value is None:
            missing.append(path)
        else:
            cached[str(path)] = value
    print(f"FEATURE_CACHE ready={len(cached)} missing={len(missing)}", flush=True)
    if not missing:
        return cached

    import librosa
    import torch
    from speechbrain.inference.speaker import EncoderClassifier
    from transformers import AutoFeatureExtractor, WavLMForXVector

    device = "cuda"
    ecapa = EncoderClassifier.from_hparams(
        source=readouts["ecapa"]["snapshot_path"],
        savedir=str(Path(config["run_root"]) / "model-runtime/ecapa"),
        run_opts={"device": device},
    )
    extractor = AutoFeatureExtractor.from_pretrained(
        readouts["wavlm"]["snapshot_path"], local_files_only=True
    )
    wavlm_model = WavLMForXVector.from_pretrained(
        readouts["wavlm"]["snapshot_path"], local_files_only=True
    ).to(device).eval()
    for index, path in enumerate(missing, start=1):
        audio, _ = librosa.load(path, sr=16_000, mono=True)
        tensor = torch.from_numpy(audio).unsqueeze(0).to(device)
        with torch.no_grad():
            ecapa_value = ecapa.encode_batch(tensor).squeeze().cpu().numpy()
            inputs = extractor(
                audio, sampling_rate=16_000, return_tensors="pt"
            ).to(device)
            wavlm_value = wavlm_model(**inputs).embeddings.squeeze().cpu().numpy()
        value = (
            np.asarray(ecapa_value, dtype=np.float32),
            np.asarray(wavlm_value, dtype=np.float32),
        )
        save_cached(
            cache_dir / f"{cache_key(path)}.npz",
            source_path=path,
            source_hash=hashes[str(path)],
            readout_pin=readout_pin,
            ecapa=value[0],
            wavlm=value[1],
        )
        cached[str(path)] = value
        if index % 100 == 0:
            print(f"FEATURE_PROGRESS {index}/{len(missing)}", flush=True)
    return cached


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 0.0 or not math_isfinite(denominator):
        raise RuntimeError("invalid embedding norm")
    value = float(np.dot(left, right) / denominator)
    if not math_isfinite(value):
        raise RuntimeError("non-finite cosine similarity")
    return value


def math_isfinite(value: float) -> bool:
    return bool(np.isfinite(value))


def write_score_table(
    path: Path,
    clone_rows: list[dict[str, Any]],
    clone_hashes: dict[str, str],
    manifest_by_speaker: dict[str, Any],
    embeddings: dict[str, tuple[np.ndarray, np.ndarray]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SCORE_HEADER, delimiter="\t")
        writer.writeheader()
        for clone in clone_rows:
            prompt_mic = clone["prompt_mic"]
            candidate_mic = "mic2" if prompt_mic == "mic1" else "mic1"
            speaker = manifest_by_speaker[clone["speaker"]]
            candidates = {
                arm: speaker["audio"][f"{arm}_{candidate_mic}"] for arm in ARMS
            }
            same_candidates = {
                arm: speaker["audio"][f"{arm}_{prompt_mic}"] for arm in ARMS
            }
            clone_path = str(clone["path"].resolve())
            clone_features = embeddings[clone_path]
            similarities = {}
            for arm in ARMS:
                candidate_features = embeddings[candidates[arm]["path"]]
                same_features = embeddings[same_candidates[arm]["path"]]
                similarities[f"ecapa_{arm}"] = cosine(
                    clone_features[0], candidate_features[0]
                )
                similarities[f"wavlm_{arm}"] = cosine(
                    clone_features[1], candidate_features[1]
                )
                similarities[f"ecapa_same_{arm}"] = cosine(
                    clone_features[0], same_features[0]
                )
                similarities[f"wavlm_same_{arm}"] = cosine(
                    clone_features[1], same_features[1]
                )
            writer.writerow(
                {
                    "speaker": clone["speaker"],
                    "system": clone["system"],
                    "text_index": clone["text_index"],
                    "prompt_mic": prompt_mic,
                    "seed_arm": clone["seed_arm"],
                    "clone_path": clone_path,
                    "clone_sha256": clone_hashes[clone_path],
                    "candidate_A_path": candidates["A"]["path"],
                    "candidate_A_sha256": candidates["A"]["sha256"],
                    "candidate_B_path": candidates["B"]["path"],
                    "candidate_B_sha256": candidates["B"]["sha256"],
                    "same_candidate_A_path": same_candidates["A"]["path"],
                    "same_candidate_A_sha256": same_candidates["A"]["sha256"],
                    "same_candidate_B_path": same_candidates["B"]["path"],
                    "same_candidate_B_sha256": same_candidates["B"]["sha256"],
                    **{key: format(value, ".17g") for key, value in similarities.items()},
                }
            )
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    config["_config_path"] = str(args.config.resolve())
    require_hash(
        Path(__file__).resolve(),
        config["source_pins"]["score"]["sha256"],
        "SCORE_SOURCE",
    )
    manifest_path = Path(config["manifest"]["path"])
    require_hash(manifest_path, config["manifest"]["sha256"], "MANIFEST")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "exp205-selection-manifest-v1":
        raise RuntimeError("manifest schema mismatch")
    run_root = Path(config["run_root"])
    clone_rows, clone_hashes = exact_clone_census(manifest, run_root)
    if len(clone_rows) != 3456:
        raise RuntimeError("expected 3,456 clones")
    clone_ledgers = verify_clone_ledgers(
        clone_rows, clone_hashes, manifest, config
    )

    manifest_by_speaker = {row["speaker"]: row for row in manifest["speakers"]}
    candidate_hashes = {
        record["path"]: record["sha256"]
        for speaker in manifest["speakers"]
        for record in speaker["audio"].values()
    }
    for path, expected in candidate_hashes.items():
        require_hash(Path(path), expected, "CANDIDATE")
    all_hashes = {**candidate_hashes, **clone_hashes}
    all_paths = [Path(path) for path in sorted(all_hashes)]
    embeddings = extract_embeddings(
        all_paths, all_hashes, run_root / "feature-cache", config
    )
    if set(embeddings) != set(all_hashes):
        raise RuntimeError("feature census mismatch")

    scores_path = Path(config["scores_path"])
    write_score_table(
        scores_path, clone_rows, clone_hashes, manifest_by_speaker, embeddings
    )
    scores_hash = sha256(scores_path)
    for path, expected in clone_hashes.items():
        require_hash(Path(path), expected, "CLONE_POST_SCORE")

    source_pins = config["source_pins"]
    for label, record in source_pins.items():
        require_hash(Path(record["path"]), record["sha256"], label.upper())
    clones_receipt = [
        {
            "speaker": row["speaker"],
            "system": row["system"],
            "text_index": row["text_index"],
            "prompt_mic": row["prompt_mic"],
            "seed_arm": row["seed_arm"],
            "path": str(row["path"].resolve()),
            "sha256": clone_hashes[str(row["path"].resolve())],
            "ledger_path": clone_ledgers[str(row["path"].resolve())]["path"],
            "ledger_sha256": clone_ledgers[str(row["path"].resolve())]["sha256"],
        }
        for row in clone_rows
    ]
    receipt_path = Path(config["execution_receipt_path"])
    receipt = {
        "schema": "exp205-execution-receipt-v1",
        "manifest_sha256": sha256(manifest_path),
        "scores_sha256": scores_hash,
        "execution_config_sha256": sha256(args.config),
        "clone_count": len(clones_receipt),
        "candidate_count": len(candidate_hashes),
        "expected_counts": config["expected_counts"],
        "generator_pins": config["generators"],
        "readout_pins": config["readouts"],
        "source_pins": source_pins,
        "clones": clones_receipt,
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")

    analysis_config_path = Path(config["analysis_config_path"])
    analysis_config = {
        "execution_config": {
            "path": str(args.config.resolve()),
            "sha256": sha256(args.config),
        },
        "manifest": {"path": str(manifest_path), "sha256": sha256(manifest_path)},
        "execution_receipt": {
            "path": str(receipt_path),
            "sha256": sha256(receipt_path),
        },
        "scores": {"path": str(scores_path), "sha256": scores_hash},
        "verdict": source_pins["verdict"],
        "output": config["scientific_result_path"],
    }
    analysis_config_path.parent.mkdir(parents=True, exist_ok=True)
    analysis_config_path.write_text(
        json.dumps(analysis_config, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"EXP205_SCORING_COMPLETE clones={len(clones_receipt)} candidates={len(candidate_hashes)}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
