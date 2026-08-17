"""Authenticated EXP-206 clone-to-clone crossover analyzer.

The program authenticates all inputs before loading embedding arrays, writes
scientific values only to the requested result JSON, and prints status only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np


SYSTEMS = ("f5", "xtts", "cosy", "seedvc")
MICS = ("mic1", "mic2")
ARMS = ("A", "B")
TEXTS = tuple(range(4))
DIRECTIONS = {
    "primary_mic1_to_mic2": ("mic1", "mic2"),
    "reverse_mic2_to_mic1": ("mic2", "mic1"),
}
READOUTS = ("ecapa", "wavlm")
BOOTSTRAPS = 100_000
BOOTSTRAP_SEED = 2062027
EXPECTED_COUNTS = {
    "speakers": 54,
    "systems": 4,
    "texts": 4,
    "prompt_microphones": 2,
    "seed_arms": 2,
    "clones": 3456,
    "comparisons_per_speaker_per_direction": 288,
}


class InfrastructureError(RuntimeError):
    """A failure that prevents a scientific result."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_hash(path: Path, expected: str, label: str) -> None:
    if not isinstance(expected, str) or len(expected) != 64:
        raise InfrastructureError(f"{label}_EXPECTED_HASH_INVALID")
    observed = sha256(path)
    if observed != expected:
        raise InfrastructureError(
            f"{label}_HASH_MISMATCH expected={expected} observed={observed}"
        )


def load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InfrastructureError(f"{label}_INVALID_JSON") from exc
    if not isinstance(value, dict):
        raise InfrastructureError(f"{label}_NOT_OBJECT")
    return value


def load_verdict(path: Path, expected_hash: str) -> ModuleType:
    source = path.read_bytes()
    if hashlib.sha256(source).hexdigest() != expected_hash:
        raise InfrastructureError("VERDICT_HASH_MISMATCH")
    name = "_exp206_authenticated_verdict"
    module = ModuleType(name)
    module.__file__ = str(path)
    previous = sys.modules.get(name)
    sys.modules[name] = module
    try:
        exec(compile(source, str(path), "exec"), module.__dict__)
    finally:
        if previous is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous
    if sha256(path) != expected_hash or not callable(getattr(module, "decide", None)):
        raise InfrastructureError("VERDICT_LOAD_INVALID")
    return module


def expected_keys(speakers: tuple[str, ...]) -> set[tuple[str, str, int, str, str]]:
    return {
        (speaker, system, text, mic, arm)
        for speaker in speakers
        for system in SYSTEMS
        for text in TEXTS
        for mic in MICS
        for arm in ARMS
    }


def validate_input_manifest(
    manifest: dict[str, Any],
) -> tuple[tuple[str, ...], dict[tuple[str, str, int, str, str], dict[str, Any]]]:
    if manifest.get("schema") != "exp206-input-manifest-v1":
        raise InfrastructureError("INPUT_MANIFEST_SCHEMA_INVALID")
    if manifest.get("status") != "FROZEN_BEFORE_EXP206_ANALYSIS":
        raise InfrastructureError("INPUT_MANIFEST_NOT_FROZEN")
    if manifest.get("source_experiment") != "EXP-205":
        raise InfrastructureError("SOURCE_EXPERIMENT_INVALID")
    if manifest.get("counts") != EXPECTED_COUNTS:
        raise InfrastructureError("INPUT_COUNTS_INVALID")
    speakers_value = manifest.get("speakers")
    if (
        not isinstance(speakers_value, list)
        or len(speakers_value) != 54
        or any(not isinstance(value, str) for value in speakers_value)
        or speakers_value != sorted(set(speakers_value))
    ):
        raise InfrastructureError("INPUT_SPEAKERS_INVALID")
    speakers = tuple(speakers_value)
    expected = expected_keys(speakers)
    rows = manifest.get("clones")
    if not isinstance(rows, list) or len(rows) != 3456:
        raise InfrastructureError("INPUT_CLONE_ROWS_INVALID")
    required_fields = {
        "speaker",
        "system",
        "text_index",
        "prompt_mic",
        "seed_arm",
        "clone_relative_path",
        "clone_sha256",
        "ledger_relative_path",
        "ledger_sha256",
        "cache_file",
        "cache_sha256",
    }
    indexed = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != required_fields:
            raise InfrastructureError("INPUT_CLONE_RECORD_SCHEMA_INVALID")
        key = (
            row["speaker"],
            row["system"],
            row["text_index"],
            row["prompt_mic"],
            row["seed_arm"],
        )
        if key not in expected or key in indexed:
            raise InfrastructureError(f"INPUT_CLONE_KEY_INVALID:{key}")
        speaker, system, text, mic, arm = key
        expected_clone = f"clones/{system}__{mic}__seed{arm}/{speaker}_t{text}.wav"
        if row["clone_relative_path"] != expected_clone:
            raise InfrastructureError(f"INPUT_CLONE_PATH_INVALID:{key}")
        if row["ledger_relative_path"] != expected_clone + ".ledger.json":
            raise InfrastructureError(f"INPUT_LEDGER_PATH_INVALID:{key}")
        cache_file = row["cache_file"]
        if (
            not isinstance(cache_file, str)
            or len(cache_file) != 28
            or not cache_file.endswith(".npz")
            or any(char not in "0123456789abcdef" for char in cache_file[:-4])
        ):
            raise InfrastructureError(f"INPUT_CACHE_NAME_INVALID:{key}")
        for field in ("clone_sha256", "ledger_sha256", "cache_sha256"):
            value = row[field]
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(char not in "0123456789abcdef" for char in value)
            ):
                raise InfrastructureError(f"INPUT_HASH_INVALID:{key}:{field}")
        indexed[key] = row
    if set(indexed) != expected:
        raise InfrastructureError("INPUT_CLONE_KEYSET_INVALID")
    for field in (
        "execution_config_sha256",
        "execution_receipt_sha256",
        "selection_manifest_sha256",
        "exp205_scores_sha256_not_opened",
        "readout_cache_pin",
    ):
        value = manifest.get(field)
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)
        ):
            raise InfrastructureError(f"INPUT_ROOT_HASH_INVALID:{field}")
    return speakers, indexed


def cache_key(path: Path) -> str:
    return hashlib.sha256(str(path.resolve()).encode("utf-8")).hexdigest()[:24]


def authenticate_contract(
    manifest: dict[str, Any],
    execution_config_path: Path,
    receipt_path: Path,
    indexed: dict[tuple[str, str, int, str, str], dict[str, Any]],
    run_root: Path,
) -> None:
    require_hash(
        execution_config_path,
        manifest["execution_config_sha256"],
        "EXECUTION_CONFIG",
    )
    require_hash(receipt_path, manifest["execution_receipt_sha256"], "RECEIPT")
    config = load_json(execution_config_path, "EXECUTION_CONFIG")
    receipt = load_json(receipt_path, "RECEIPT")
    if config.get("schema") != "exp205-execution-config-v1":
        raise InfrastructureError("EXECUTION_CONFIG_SCHEMA_INVALID")
    if receipt.get("schema") != "exp205-execution-receipt-v1":
        raise InfrastructureError("RECEIPT_SCHEMA_INVALID")
    if receipt.get("execution_config_sha256") != manifest["execution_config_sha256"]:
        raise InfrastructureError("RECEIPT_CONFIG_ROOT_INVALID")
    if receipt.get("manifest_sha256") != manifest["selection_manifest_sha256"]:
        raise InfrastructureError("RECEIPT_SELECTION_ROOT_INVALID")
    if receipt.get("scores_sha256") != manifest["exp205_scores_sha256_not_opened"]:
        raise InfrastructureError("RECEIPT_SCORE_ROOT_INVALID")
    if config.get("manifest", {}).get("sha256") != manifest["selection_manifest_sha256"]:
        raise InfrastructureError("EXECUTION_CONFIG_SELECTION_ROOT_INVALID")
    if config.get("expected_counts", {}).get("clones") != 3456:
        raise InfrastructureError("EXECUTION_CONFIG_COUNTS_INVALID")
    if receipt.get("clone_count") != 3456:
        raise InfrastructureError("RECEIPT_COUNTS_INVALID")
    resolved_root = run_root.resolve()
    if Path(config.get("run_root", "")).resolve() != resolved_root:
        raise InfrastructureError("EXECUTION_CONFIG_RUN_ROOT_INVALID")
    for config_field, receipt_field in (
        ("generators", "generator_pins"),
        ("readouts", "readout_pins"),
        ("source_pins", "source_pins"),
    ):
        if receipt.get(receipt_field) != config.get(config_field):
            raise InfrastructureError(
                f"RECEIPT_EXECUTION_CONTRACT_INVALID:{config_field}"
            )
    payload = {
        "readouts": config.get("readouts"),
        "score_source": config.get("source_pins", {}).get("score"),
    }
    observed_pin = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    if observed_pin != manifest["readout_cache_pin"]:
        raise InfrastructureError("READOUT_CACHE_PIN_INVALID")

    receipt_rows = receipt.get("clones")
    if not isinstance(receipt_rows, list) or len(receipt_rows) != 3456:
        raise InfrastructureError("RECEIPT_CLONE_ROWS_INVALID")
    receipt_index = {}
    for row in receipt_rows:
        if not isinstance(row, dict):
            raise InfrastructureError("RECEIPT_CLONE_RECORD_INVALID")
        try:
            key = (
                row["speaker"],
                row["system"],
                int(row["text_index"]),
                row["prompt_mic"],
                row["seed_arm"],
            )
            receipt_clone = Path(row["path"]).resolve()
            receipt_ledger = Path(row["ledger_path"]).resolve()
            receipt_clone_hash = row["sha256"]
            receipt_ledger_hash = row["ledger_sha256"]
        except (KeyError, TypeError, ValueError) as exc:
            raise InfrastructureError("RECEIPT_CLONE_RECORD_INVALID") from exc
        if key not in indexed or key in receipt_index:
            raise InfrastructureError(f"RECEIPT_CLONE_KEY_INVALID:{key}")
        frozen = indexed[key]
        expected_clone = (resolved_root / frozen["clone_relative_path"]).resolve()
        expected_ledger = (resolved_root / frozen["ledger_relative_path"]).resolve()
        if receipt_clone != expected_clone:
            raise InfrastructureError(f"RECEIPT_CLONE_PATH_MISMATCH:{key}")
        if receipt_ledger != expected_ledger:
            raise InfrastructureError(f"RECEIPT_LEDGER_PATH_MISMATCH:{key}")
        if receipt_clone_hash != frozen["clone_sha256"]:
            raise InfrastructureError(f"RECEIPT_CLONE_HASH_MISMATCH:{key}")
        if receipt_ledger_hash != frozen["ledger_sha256"]:
            raise InfrastructureError(f"RECEIPT_LEDGER_HASH_MISMATCH:{key}")
        receipt_index[key] = row
    if set(receipt_index) != set(indexed):
        raise InfrastructureError("RECEIPT_INPUT_KEYSET_MISMATCH")


def authenticate_and_load_embeddings(
    indexed: dict[tuple[str, str, int, str, str], dict[str, Any]],
    run_root: Path,
    expected_readout_pin: str,
) -> tuple[dict[tuple[str, str, int, str, str], dict[str, np.ndarray]], dict[str, int]]:
    embeddings = {}
    dimensions: dict[str, int] = {}
    resolved_root = run_root.resolve()
    for index, (key, row) in enumerate(sorted(indexed.items()), start=1):
        clone_path = (resolved_root / row["clone_relative_path"]).resolve()
        ledger_path = (resolved_root / row["ledger_relative_path"]).resolve()
        try:
            clone_path.relative_to(resolved_root)
            ledger_path.relative_to(resolved_root)
        except ValueError as exc:
            raise InfrastructureError(f"RUNTIME_PATH_ESCAPE:{key}") from exc
        if row["cache_file"] != f"{cache_key(clone_path)}.npz":
            raise InfrastructureError(f"CACHE_KEY_INVALID:{key}")
        cache_path = resolved_root / "feature-cache" / row["cache_file"]
        require_hash(clone_path, row["clone_sha256"], f"CLONE:{key}")
        require_hash(ledger_path, row["ledger_sha256"], f"LEDGER:{key}")
        require_hash(cache_path, row["cache_sha256"], f"CACHE:{key}")
        try:
            with np.load(cache_path, allow_pickle=False) as payload:
                if set(payload.files) != {
                    "source_path",
                    "source_sha256",
                    "readout_pin",
                    "ecapa",
                    "wavlm",
                }:
                    raise InfrastructureError(f"CACHE_SCHEMA_INVALID:{key}")
                if str(payload["source_path"]) != str(clone_path):
                    raise InfrastructureError(f"CACHE_SOURCE_PATH_INVALID:{key}")
                if str(payload["source_sha256"]) != row["clone_sha256"]:
                    raise InfrastructureError(f"CACHE_SOURCE_HASH_INVALID:{key}")
                if str(payload["readout_pin"]) != expected_readout_pin:
                    raise InfrastructureError(f"CACHE_READOUT_PIN_INVALID:{key}")
                value = {
                    readout: np.asarray(payload[readout], dtype=np.float64)
                    for readout in READOUTS
                }
        except (OSError, ValueError, KeyError) as exc:
            raise InfrastructureError(f"CACHE_LOAD_INVALID:{key}") from exc
        require_hash(cache_path, row["cache_sha256"], f"CACHE_POST_LOAD:{key}")
        for readout, vector in value.items():
            if vector.ndim != 1 or vector.size < 2 or not np.isfinite(vector).all():
                raise InfrastructureError(f"EMBEDDING_INVALID:{key}:{readout}")
            norm = float(np.linalg.norm(vector))
            if not math.isfinite(norm) or norm <= 0.0:
                raise InfrastructureError(f"EMBEDDING_NORM_INVALID:{key}:{readout}")
            dimension = dimensions.setdefault(readout, int(vector.size))
            if vector.size != dimension:
                raise InfrastructureError(f"EMBEDDING_DIMENSION_INVALID:{key}:{readout}")
            value[readout] = vector / norm
        embeddings[key] = value
        if index % 500 == 0:
            print(f"EXP206_AUTH_PROGRESS {index}/3456", flush=True)
    if len(embeddings) != 3456:
        raise InfrastructureError("EMBEDDING_CENSUS_INVALID")
    return embeddings, dimensions


def speaker_direction_values(
    embeddings: dict[tuple[str, str, int, str, str], dict[str, np.ndarray]],
    speaker: str,
    query_mic: str,
    candidate_mic: str,
) -> dict[str, tuple[float, float]]:
    follow: dict[str, list[float]] = defaultdict(list)
    margins: dict[str, list[float]] = defaultdict(list)
    for query_arm in ARMS:
        other_arm = "B" if query_arm == "A" else "A"
        for query_system in SYSTEMS:
            for query_text in TEXTS:
                query = embeddings[
                    (speaker, query_system, query_text, query_mic, query_arm)
                ]
                for candidate_system in SYSTEMS:
                    if candidate_system == query_system:
                        continue
                    for candidate_text in TEXTS:
                        if candidate_text == query_text:
                            continue
                        correct = embeddings[
                            (
                                speaker,
                                candidate_system,
                                candidate_text,
                                candidate_mic,
                                query_arm,
                            )
                        ]
                        incorrect = embeddings[
                            (
                                speaker,
                                candidate_system,
                                candidate_text,
                                candidate_mic,
                                other_arm,
                            )
                        ]
                        for readout in READOUTS:
                            own = float(np.dot(query[readout], correct[readout]))
                            other = float(np.dot(query[readout], incorrect[readout]))
                            margin = own - other
                            margins[readout].append(margin)
                            follow[readout].append(
                                1.0 if margin > 0.0 else 0.0 if margin < 0.0 else 0.5
                            )
    result = {}
    for readout in READOUTS:
        if len(follow[readout]) != 288 or len(margins[readout]) != 288:
            raise InfrastructureError(
                f"COMPARISON_COUNT_INVALID:{speaker}:{query_mic}:{readout}"
            )
        result[readout] = (
            float(np.mean(follow[readout])),
            float(np.mean(margins[readout])),
        )
    return result


def compute_result(
    embeddings: dict[tuple[str, str, int, str, str], dict[str, np.ndarray]],
    speakers: tuple[str, ...],
) -> tuple[dict[str, Any], dict[str, dict[str, dict[str, float]]]]:
    vectors: dict[tuple[str, str], list[float]] = defaultdict(list)
    margin_vectors: dict[tuple[str, str], list[float]] = defaultdict(list)
    for speaker in speakers:
        for direction, (query_mic, candidate_mic) in DIRECTIONS.items():
            values = speaker_direction_values(
                embeddings, speaker, query_mic, candidate_mic
            )
            for readout, (follow, margin) in values.items():
                vectors[(direction, readout)].append(follow)
                margin_vectors[(direction, readout)].append(margin)
    arrays = {key: np.asarray(value, dtype=np.float64) for key, value in vectors.items()}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(0, len(speakers), size=(BOOTSTRAPS, len(speakers)), dtype=np.int32)
    cells = {}
    verdict_cells = {}
    for direction in DIRECTIONS:
        cells[direction] = {}
        verdict_cells[direction] = {}
        for readout in READOUTS:
            vector = arrays[(direction, readout)]
            distribution = vector[indices].mean(axis=1)
            lo, hi = np.quantile(distribution, (0.025, 0.975))
            point = float(vector.mean())
            cells[direction][readout] = {
                "point": point,
                "stability_interval_95": [float(lo), float(hi)],
                "speaker_means": vector.tolist(),
                "speaker_mean_margins": margin_vectors[(direction, readout)],
            }
            verdict_cells[direction][readout] = {"point": point, "lcb": float(lo)}
    return cells, verdict_cells


def write_failure(path: Path, reason: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps({"status": "INFRASTRUCTURE_FAILURE", "reason": reason}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def run(args: argparse.Namespace) -> int:
    output = args.output
    try:
        require_hash(Path(__file__).resolve(), args.expected_source_sha256, "ANALYZER")
        require_hash(args.input_manifest, args.expected_input_manifest_sha256, "INPUT_MANIFEST")
        manifest = load_json(args.input_manifest, "INPUT_MANIFEST")
        speakers, indexed = validate_input_manifest(manifest)
        authenticate_contract(
            manifest,
            args.execution_config,
            args.receipt,
            indexed,
            args.run_root,
        )
        verdict_path = Path(__file__).with_name("verdict.py")
        verdict_module = load_verdict(verdict_path, args.expected_verdict_sha256)
        embeddings, dimensions = authenticate_and_load_embeddings(
            indexed, args.run_root, manifest["readout_cache_pin"]
        )
        cells, verdict_cells = compute_result(embeddings, speakers)
        verdict = verdict_module.decide(verdict_cells)
        result = {
            "schema": "exp206-scientific-result-v1",
            "status": "SCIENTIFIC_RESULT",
            "scope": "fixed_54_speaker_cross_generator_cross_text_clone_to_clone_crossover",
            "counts": {
                **EXPECTED_COUNTS,
                "bootstrap_replicates": BOOTSTRAPS,
                "bootstrap_seed": BOOTSTRAP_SEED,
            },
            "embedding_dimensions": dimensions,
            "directions": cells,
            "verdict": verdict.as_dict(),
            "artifact_hashes": {
                "analyzer": sha256(Path(__file__).resolve()),
                "verdict": sha256(verdict_path),
                "input_manifest": sha256(args.input_manifest),
                "execution_config": sha256(args.execution_config),
                "execution_receipt": sha256(args.receipt),
            },
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, output)
    except (InfrastructureError, OSError, ValueError, KeyError, TypeError) as exc:
        write_failure(output, f"{type(exc).__name__}:{exc}")
        print("EXP206_INFRASTRUCTURE_FAILURE", flush=True)
        return 2
    print("EXP206_ANALYSIS_COMPLETE", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--expected-input-manifest-sha256", required=True)
    parser.add_argument("--execution-config", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--expected-verdict-sha256", required=True)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
