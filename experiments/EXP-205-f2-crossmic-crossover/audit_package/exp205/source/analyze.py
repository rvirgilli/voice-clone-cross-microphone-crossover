"""Sealed-style EXP-205 analyzer for a fully authenticated score table.

The analyzer performs an identity-only first pass, verifies the complete clone
and candidate census against the selection manifest and execution receipt, and
only then parses similarities.  Scientific values are written to JSON and are
never printed to stdout.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np


EXPECTED_HEADER = (
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
SYSTEMS = ("f5", "xtts", "cosy", "seedvc")
PROMPT_MICS = ("mic1", "mic2")
ARMS = ("A", "B")
TEXT_INDICES = tuple(range(4))
BOOTSTRAPS = 100_000
BOOTSTRAP_SEED = 2052027


class InfrastructureError(RuntimeError):
    """A failure that invalidates the scientific result."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_hash(path: Path, expected: str, label: str) -> None:
    observed = sha256(path)
    if observed != expected:
        raise InfrastructureError(
            f"{label}_HASH_MISMATCH expected={expected} observed={observed}"
        )


def load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InfrastructureError(f"{label}_INVALID_JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise InfrastructureError(f"{label}_NOT_OBJECT")
    return value


def authenticated_verdict(path: Path, expected_hash: str) -> ModuleType:
    """Compile the exact authenticated verdict bytes without import lookup."""
    source = path.read_bytes()
    observed = hashlib.sha256(source).hexdigest()
    if observed != expected_hash:
        raise InfrastructureError(
            f"VERDICT_HASH_MISMATCH expected={expected_hash} observed={observed}"
        )
    name = "_exp205_authenticated_verdict"
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
    if sha256(path) != expected_hash:
        raise InfrastructureError("VERDICT_CHANGED_DURING_LOAD")
    required = ("primary_verdict", "reverse_verdict", "headline_permission")
    if any(not callable(getattr(module, item, None)) for item in required):
        raise InfrastructureError("VERDICT_NAMESPACE_INVALID")
    return module


def validate_manifest(manifest: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
    if manifest.get("schema") != "exp205-selection-manifest-v1":
        raise InfrastructureError("MANIFEST_SCHEMA_INVALID")
    counts = manifest.get("counts", {})
    expected_counts = {
        "speakers": 54,
        "seed_arms": 2,
        "prompt_microphones": 2,
        "generated_texts": 4,
        "systems": 4,
        "expected_clones": 3456,
        "comparisons_per_speaker_per_direction": 32,
    }
    if counts != expected_counts:
        raise InfrastructureError("MANIFEST_COUNTS_INVALID")
    rows = manifest.get("speakers")
    if not isinstance(rows, list) or len(rows) != 54:
        raise InfrastructureError("MANIFEST_SPEAKER_ROWS_INVALID")
    by_speaker: dict[str, Any] = {}
    for row in rows:
        speaker = row.get("speaker")
        if not isinstance(speaker, str) or speaker in by_speaker:
            raise InfrastructureError("MANIFEST_SPEAKER_ID_INVALID")
        audio = row.get("audio", {})
        if set(audio) != {"A_mic1", "A_mic2", "B_mic1", "B_mic2"}:
            raise InfrastructureError(f"MANIFEST_AUDIO_KEYS_INVALID:{speaker}")
        for record in audio.values():
            if set(record) != {
                "path",
                "sha256",
                "frames",
                "samplerate_hz",
                "duration_s",
            }:
                raise InfrastructureError(f"MANIFEST_AUDIO_RECORD_INVALID:{speaker}")
        by_speaker[speaker] = row
    speakers = sorted(by_speaker)
    if len(speakers) != len(set(speakers)):
        raise InfrastructureError("MANIFEST_DUPLICATE_SPEAKER")
    return speakers, by_speaker


def expected_keys(speakers: list[str]) -> set[tuple[str, str, int, str, str]]:
    return {
        (speaker, system, text_index, prompt_mic, arm)
        for speaker in speakers
        for system in SYSTEMS
        for text_index in TEXT_INDICES
        for prompt_mic in PROMPT_MICS
        for arm in ARMS
    }


def read_identity_pass(
    scores_path: Path,
    speakers: list[str],
    manifest_by_speaker: dict[str, Any],
) -> dict[tuple[str, str, int, str, str], dict[str, str]]:
    expected = expected_keys(speakers)
    observed: dict[tuple[str, str, int, str, str], dict[str, str]] = {}
    with scores_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if tuple(reader.fieldnames or ()) != EXPECTED_HEADER:
            raise InfrastructureError("SCORE_HEADER_INVALID")
        for row in reader:
            try:
                key = (
                    row["speaker"],
                    row["system"],
                    int(row["text_index"]),
                    row["prompt_mic"],
                    row["seed_arm"],
                )
            except (KeyError, ValueError) as exc:
                raise InfrastructureError(f"SCORE_IDENTITY_INVALID: {exc}") from exc
            if key not in expected:
                raise InfrastructureError(f"SCORE_KEY_UNEXPECTED:{key}")
            if key in observed:
                raise InfrastructureError(f"SCORE_KEY_DUPLICATE:{key}")
            speaker, _, _, prompt_mic, _ = key
            candidate_mic = "mic2" if prompt_mic == "mic1" else "mic1"
            manifest_audio = manifest_by_speaker[speaker]["audio"]
            expected_candidates = {
                "candidate_A_path": manifest_audio[f"A_{candidate_mic}"]["path"],
                "candidate_A_sha256": manifest_audio[f"A_{candidate_mic}"]["sha256"],
                "candidate_B_path": manifest_audio[f"B_{candidate_mic}"]["path"],
                "candidate_B_sha256": manifest_audio[f"B_{candidate_mic}"]["sha256"],
                "same_candidate_A_path": manifest_audio[f"A_{prompt_mic}"]["path"],
                "same_candidate_A_sha256": manifest_audio[f"A_{prompt_mic}"]["sha256"],
                "same_candidate_B_path": manifest_audio[f"B_{prompt_mic}"]["path"],
                "same_candidate_B_sha256": manifest_audio[f"B_{prompt_mic}"]["sha256"],
            }
            for field, value in expected_candidates.items():
                if row[field] != value:
                    raise InfrastructureError(f"SCORE_CANDIDATE_MISMATCH:{key}:{field}")
            observed[key] = row
    if set(observed) != expected:
        missing = len(expected - set(observed))
        extra = len(set(observed) - expected)
        raise InfrastructureError(f"SCORE_CENSUS_INVALID missing={missing} extra={extra}")
    return observed


def receipt_clone_index(receipt: dict[str, Any]) -> dict[tuple[str, str, int, str, str], dict[str, str]]:
    if receipt.get("schema") != "exp205-execution-receipt-v1":
        raise InfrastructureError("RECEIPT_SCHEMA_INVALID")
    clones = receipt.get("clones")
    if not isinstance(clones, list) or len(clones) != 3456:
        raise InfrastructureError("RECEIPT_CLONE_COUNT_INVALID")
    result = {}
    for row in clones:
        try:
            key = (
                row["speaker"],
                row["system"],
                int(row["text_index"]),
                row["prompt_mic"],
                row["seed_arm"],
            )
            value = {
                "path": row["path"],
                "sha256": row["sha256"],
                "ledger_path": row["ledger_path"],
                "ledger_sha256": row["ledger_sha256"],
            }
        except (KeyError, ValueError, TypeError) as exc:
            raise InfrastructureError(f"RECEIPT_CLONE_INVALID: {exc}") from exc
        if key in result:
            raise InfrastructureError(f"RECEIPT_CLONE_DUPLICATE:{key}")
        result[key] = value
    return result


def verify_files_before_scores(
    identities: dict[tuple[str, str, int, str, str], dict[str, str]],
    receipt_index: dict[tuple[str, str, int, str, str], dict[str, str]],
    manifest_by_speaker: dict[str, Any],
) -> None:
    if set(receipt_index) != set(identities):
        raise InfrastructureError("RECEIPT_SCORE_KEYSET_MISMATCH")
    verified_candidates: set[str] = set()
    for key, row in identities.items():
        receipt = receipt_index[key]
        if row["clone_path"] != receipt["path"]:
            raise InfrastructureError(f"CLONE_PATH_RECEIPT_MISMATCH:{key}")
        if row["clone_sha256"] != receipt["sha256"]:
            raise InfrastructureError(f"CLONE_HASH_RECEIPT_MISMATCH:{key}")
        clone_path = Path(receipt["path"])
        if not clone_path.is_file() or sha256(clone_path) != receipt["sha256"]:
            raise InfrastructureError(f"CLONE_FILE_INVALID:{key}")
        ledger_path = Path(receipt["ledger_path"])
        if not ledger_path.is_file() or sha256(ledger_path) != receipt["ledger_sha256"]:
            raise InfrastructureError(f"CLONE_LEDGER_FILE_INVALID:{key}")
        ledger = load_json(ledger_path, "CLONE_LEDGER")
        expected_ledger_identity = {
            "schema": "exp205-clone-ledger-v1",
            "system": key[1],
            "speaker": key[0],
            "text_index": key[2],
            "prompt_mic": key[3],
            "seed_arm": key[4],
            "output_path": receipt["path"],
            "clone_sha256": receipt["sha256"],
        }
        if any(
            ledger.get(field) != value
            for field, value in expected_ledger_identity.items()
        ):
            raise InfrastructureError(f"CLONE_LEDGER_IDENTITY_INVALID:{key}")
        for arm in ARMS:
            for prefix in ("candidate", "same_candidate"):
                candidate_path = row[f"{prefix}_{arm}_path"]
                if candidate_path in verified_candidates:
                    continue
                if sha256(Path(candidate_path)) != row[f"{prefix}_{arm}_sha256"]:
                    raise InfrastructureError(f"CANDIDATE_FILE_INVALID:{candidate_path}")
                verified_candidates.add(candidate_path)
    expected_candidate_paths = {
        record["path"]
        for speaker in manifest_by_speaker.values()
        for record in speaker["audio"].values()
    }
    if verified_candidates != expected_candidate_paths:
        raise InfrastructureError("CANDIDATE_CENSUS_INVALID")


def parse_scientific_rows(
    identities: dict[tuple[str, str, int, str, str], dict[str, str]],
) -> list[dict[str, Any]]:
    result = []
    for key in sorted(identities):
        row = identities[key]
        values = {}
        for field in (
            "ecapa_A",
            "ecapa_B",
            "wavlm_A",
            "wavlm_B",
            "ecapa_same_A",
            "ecapa_same_B",
            "wavlm_same_A",
            "wavlm_same_B",
        ):
            try:
                value = float(row[field])
            except ValueError as exc:
                raise InfrastructureError(f"SCORE_VALUE_INVALID:{key}:{field}") from exc
            if not math.isfinite(value) or not -1.000001 <= value <= 1.000001:
                raise InfrastructureError(f"SCORE_VALUE_INVALID:{key}:{field}")
            values[field] = value
        speaker, system, text_index, prompt_mic, arm = key
        encoded = {
            "speaker": speaker,
            "system": system,
            "text_index": text_index,
            "prompt_mic": prompt_mic,
            "seed_arm": arm,
        }
        for encoder in ("ecapa", "wavlm"):
            own = values[f"{encoder}_{arm}"]
            other_arm = "B" if arm == "A" else "A"
            other = values[f"{encoder}_{other_arm}"]
            encoded[f"{encoder}_follow"] = 1.0 if own > other else 0.0 if own < other else 0.5
            encoded[f"{encoder}_margin"] = own - other
            same_own = values[f"{encoder}_same_{arm}"]
            same_other = values[f"{encoder}_same_{other_arm}"]
            encoded[f"{encoder}_same_follow"] = (
                1.0 if same_own > same_other else 0.0 if same_own < same_other else 0.5
            )
            encoded[f"{encoder}_same_margin"] = same_own - same_other
        result.append(encoded)
    return result


def speaker_vectors(
    rows: list[dict[str, Any]], speakers: list[str]
) -> tuple[dict[tuple[str, str], np.ndarray], dict[tuple[str, str], np.ndarray]]:
    follow: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    margin: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in rows:
        direction = "primary_mic1_to_mic2" if row["prompt_mic"] == "mic1" else "reverse_mic2_to_mic1"
        for encoder in ("ecapa", "wavlm"):
            key = (direction, encoder, row["speaker"])
            follow[key].append(row[f"{encoder}_follow"])
            margin[key].append(row[f"{encoder}_margin"])

    follow_vectors = {}
    margin_vectors = {}
    for direction in ("primary_mic1_to_mic2", "reverse_mic2_to_mic1"):
        for encoder in ("ecapa", "wavlm"):
            f_values, m_values = [], []
            for speaker in speakers:
                key = (direction, encoder, speaker)
                if len(follow[key]) != 32 or len(margin[key]) != 32:
                    raise InfrastructureError(f"SPEAKER_CELL_COUNT_INVALID:{key}")
                f_values.append(float(np.mean(follow[key])))
                m_values.append(float(np.mean(margin[key])))
            follow_vectors[(direction, encoder)] = np.asarray(f_values, dtype=np.float64)
            margin_vectors[(direction, encoder)] = np.asarray(m_values, dtype=np.float64)
    return follow_vectors, margin_vectors


def same_microphone_vectors(
    rows: list[dict[str, Any]], speakers: list[str]
) -> tuple[dict[tuple[str, str], np.ndarray], dict[tuple[str, str], np.ndarray]]:
    follow: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    margin: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in rows:
        diagnostic = f"same_{row['prompt_mic']}"
        for encoder in ("ecapa", "wavlm"):
            key = (diagnostic, encoder, row["speaker"])
            follow[key].append(row[f"{encoder}_same_follow"])
            margin[key].append(row[f"{encoder}_same_margin"])
    follow_vectors = {}
    margin_vectors = {}
    for diagnostic in ("same_mic1", "same_mic2"):
        for encoder in ("ecapa", "wavlm"):
            f_values, m_values = [], []
            for speaker in speakers:
                key = (diagnostic, encoder, speaker)
                if len(follow[key]) != 32 or len(margin[key]) != 32:
                    raise InfrastructureError(f"SAME_MIC_CELL_COUNT_INVALID:{key}")
                f_values.append(float(np.mean(follow[key])))
                m_values.append(float(np.mean(margin[key])))
            follow_vectors[(diagnostic, encoder)] = np.asarray(
                f_values, dtype=np.float64
            )
            margin_vectors[(diagnostic, encoder)] = np.asarray(
                m_values, dtype=np.float64
            )
    return follow_vectors, margin_vectors


def bootstrap_intervals(
    vectors: dict[tuple[str, str], np.ndarray]
) -> dict[tuple[str, str], tuple[float, float]]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    n = len(next(iter(vectors.values())))
    indices = rng.integers(0, n, size=(BOOTSTRAPS, n), dtype=np.int32)
    result = {}
    for key, vector in vectors.items():
        distribution = vector[indices].mean(axis=1)
        lo, hi = np.quantile(distribution, (0.025, 0.975))
        result[key] = (float(lo), float(hi))
    return result


def system_points(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in rows:
        direction = "primary_mic1_to_mic2" if row["prompt_mic"] == "mic1" else "reverse_mic2_to_mic1"
        for encoder in ("ecapa", "wavlm"):
            grouped[(direction, encoder, row["system"])].append(row[f"{encoder}_follow"])
    return {
        direction: {
            encoder: {
                system: float(np.mean(grouped[(direction, encoder, system)]))
                for system in SYSTEMS
            }
            for encoder in ("ecapa", "wavlm")
        }
        for direction in ("primary_mic1_to_mic2", "reverse_mic2_to_mic1")
    }


def write_failure(path: Path, exc: InfrastructureError) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"status": "INFRASTRUCTURE_FAILURE", "reason": str(exc)}, indent=2
        )
        + "\n",
        encoding="utf-8",
    )


def run(config_path: Path, expected_execution_config_sha256: str) -> int:
    config = load_json(config_path, "CONFIG")
    output = Path(config["output"])
    try:
        execution_config_path = Path(config["execution_config"]["path"])
        if config["execution_config"]["sha256"] != expected_execution_config_sha256:
            raise InfrastructureError("EXECUTION_CONFIG_TRUST_ROOT_MISMATCH")
        require_hash(
            execution_config_path,
            expected_execution_config_sha256,
            "EXECUTION_CONFIG",
        )
        execution_config = load_json(execution_config_path, "EXECUTION_CONFIG")
        if execution_config.get("schema") != "exp205-execution-config-v1":
            raise InfrastructureError("EXECUTION_CONFIG_SCHEMA_INVALID")
        require_hash(
            Path(__file__).resolve(),
            execution_config["source_pins"]["analyze"]["sha256"],
            "ANALYZER",
        )
        manifest_path = Path(config["manifest"]["path"])
        receipt_path = Path(config["execution_receipt"]["path"])
        scores_path = Path(config["scores"]["path"])
        verdict_path = Path(config["verdict"]["path"])
        require_hash(manifest_path, config["manifest"]["sha256"], "MANIFEST")
        require_hash(receipt_path, config["execution_receipt"]["sha256"], "RECEIPT")
        require_hash(scores_path, config["scores"]["sha256"], "SCORES")
        verdict = authenticated_verdict(verdict_path, config["verdict"]["sha256"])

        manifest = load_json(manifest_path, "MANIFEST")
        receipt = load_json(receipt_path, "RECEIPT")
        expected_paths = {
            "manifest": execution_config["manifest"]["path"],
            "receipt": execution_config["execution_receipt_path"],
            "scores": execution_config["scores_path"],
            "output": execution_config["scientific_result_path"],
        }
        observed_paths = {
            "manifest": str(manifest_path),
            "receipt": str(receipt_path),
            "scores": str(scores_path),
            "output": str(output),
        }
        if observed_paths != expected_paths:
            raise InfrastructureError(
                f"ANALYSIS_PATH_CONTRACT_MISMATCH expected={expected_paths} observed={observed_paths}"
            )
        if config["manifest"] != execution_config["manifest"]:
            raise InfrastructureError("ANALYSIS_MANIFEST_PIN_MISMATCH")
        if config["verdict"] != execution_config["source_pins"]["verdict"]:
            raise InfrastructureError("ANALYSIS_VERDICT_PIN_MISMATCH")
        if receipt.get("manifest_sha256") != config["manifest"]["sha256"]:
            raise InfrastructureError("RECEIPT_MANIFEST_HASH_MISMATCH")
        if receipt.get("scores_sha256") != config["scores"]["sha256"]:
            raise InfrastructureError("RECEIPT_SCORE_HASH_MISMATCH")
        receipt_contract = {
            "execution_config_sha256": expected_execution_config_sha256,
            "clone_count": execution_config["expected_counts"]["clones"],
            "candidate_count": execution_config["expected_counts"][
                "real_candidates"
            ],
            "expected_counts": execution_config["expected_counts"],
            "generator_pins": execution_config["generators"],
            "readout_pins": execution_config["readouts"],
            "source_pins": execution_config["source_pins"],
        }
        mismatched_contract = [
            field
            for field, value in receipt_contract.items()
            if receipt.get(field) != value
        ]
        if mismatched_contract:
            raise InfrastructureError(
                f"RECEIPT_EXECUTION_CONTRACT_MISMATCH:{mismatched_contract}"
            )

        speakers, manifest_by_speaker = validate_manifest(manifest)
        identities = read_identity_pass(scores_path, speakers, manifest_by_speaker)
        receipt_index = receipt_clone_index(receipt)
        verify_files_before_scores(identities, receipt_index, manifest_by_speaker)

        rows = parse_scientific_rows(identities)
        follow, margins = speaker_vectors(rows, speakers)
        same_follow, same_margins = same_microphone_vectors(rows, speakers)
        intervals = bootstrap_intervals(follow)
        same_intervals = bootstrap_intervals(same_follow)
        primary_key = "primary_mic1_to_mic2"
        reverse_key = "reverse_mic2_to_mic1"
        point = {key: float(value.mean()) for key, value in follow.items()}
        primary = verdict.primary_verdict(
            ecapa_point=point[(primary_key, "ecapa")],
            ecapa_lcb=intervals[(primary_key, "ecapa")][0],
            wavlm_lcb=intervals[(primary_key, "wavlm")][0],
        )
        reverse = verdict.reverse_verdict(
            ecapa_lcb=intervals[(reverse_key, "ecapa")][0],
            wavlm_lcb=intervals[(reverse_key, "wavlm")][0],
        )
        result = {
            "status": "SCIENTIFIC_RESULT",
            "scope": "fixed_54_speaker_roster_speaker_composition_stability_not_population_coverage",
            "counts": {
                "speakers": len(speakers),
                "speaker_ids": speakers,
                "score_rows": len(rows),
                "comparisons_per_speaker_per_direction": 32,
                "bootstrap_replicates": BOOTSTRAPS,
                "bootstrap_seed": BOOTSTRAP_SEED,
            },
            "directions": {
                direction: {
                    encoder: {
                        "point": point[(direction, encoder)],
                        "stability_interval_95": list(intervals[(direction, encoder)]),
                        "speaker_means": follow[(direction, encoder)].tolist(),
                        "speaker_mean_margins": margins[(direction, encoder)].tolist(),
                    }
                    for encoder in ("ecapa", "wavlm")
                }
                for direction in (primary_key, reverse_key)
            },
            "system_points_no_intervals": system_points(rows),
            "same_microphone_diagnostics_no_verdict": {
                diagnostic: {
                    encoder: {
                        "point": float(same_follow[(diagnostic, encoder)].mean()),
                        "stability_interval_95": list(
                            same_intervals[(diagnostic, encoder)]
                        ),
                        "speaker_means": same_follow[(diagnostic, encoder)].tolist(),
                        "speaker_mean_margins": same_margins[
                            (diagnostic, encoder)
                        ].tolist(),
                    }
                    for encoder in ("ecapa", "wavlm")
                }
                for diagnostic in ("same_mic1", "same_mic2")
            },
            "primary": primary.as_dict(),
            "reverse": reverse.as_dict(),
            "headline_permission": verdict.headline_permission(primary, reverse),
            "artifact_hashes": {
                "config": sha256(config_path),
                "execution_config": expected_execution_config_sha256,
                "manifest": sha256(manifest_path),
                "execution_receipt": sha256(receipt_path),
                "scores": sha256(scores_path),
                "verdict": sha256(verdict_path),
            },
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    except (KeyError, TypeError, ValueError, OSError, InfrastructureError) as exc:
        failure = exc if isinstance(exc, InfrastructureError) else InfrastructureError(str(exc))
        write_failure(output, failure)
        print("INFRASTRUCTURE_FAILURE", flush=True)
        return 2
    print("EXP205_ANALYSIS_COMPLETE", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--expected-execution-config-sha256", required=True
    )
    args = parser.parse_args()
    return run(args.config, args.expected_execution_config_sha256)


if __name__ == "__main__":
    raise SystemExit(main())
