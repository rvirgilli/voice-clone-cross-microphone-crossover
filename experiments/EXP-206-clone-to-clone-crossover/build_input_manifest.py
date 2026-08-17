"""Build the outcome-blind EXP-206 input and cache trust root.

This builder authenticates EXP-205 configuration, clone identities, audio,
ledgers, and opaque feature-cache files. It never opens embedding arrays,
scores.tsv, or an EXP-205 scientific-result file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


SYSTEMS = ("f5", "xtts", "cosy", "seedvc")
MICS = ("mic1", "mic2")
ARMS = ("A", "B")
TEXTS = tuple(range(4))
EXPECTED_COUNTS = {
    "speakers": 54,
    "systems": 4,
    "prompt_microphones": 2,
    "seed_arms": 2,
    "texts": 4,
    "clones": 3456,
    "real_candidates": 216,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON root is not an object: {path}")
    return value


def cache_key(path: Path) -> str:
    return hashlib.sha256(str(path.resolve()).encode("utf-8")).hexdigest()[:24]


def expected_keys(speakers: tuple[str, ...]) -> set[tuple[str, str, int, str, str]]:
    return {
        (speaker, system, text, mic, arm)
        for speaker in speakers
        for system in SYSTEMS
        for text in TEXTS
        for mic in MICS
        for arm in ARMS
    }


def readout_pin(config: dict[str, Any]) -> str:
    payload = {
        "readouts": config["readouts"],
        "score_source": config["source_pins"]["score"],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


def relative_to(path: Path, root: Path, label: str) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise RuntimeError(f"{label} escapes run root: {path}") from exc


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execution-config", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    execution_config = load_json(args.execution_config)
    receipt = load_json(args.receipt)
    run_root = args.run_root.resolve()
    config_hash = sha256(args.execution_config)
    receipt_hash = sha256(args.receipt)

    if execution_config.get("schema") != "exp205-execution-config-v1":
        raise RuntimeError("execution config schema mismatch")
    if receipt.get("schema") != "exp205-execution-receipt-v1":
        raise RuntimeError("receipt schema mismatch")
    if receipt.get("execution_config_sha256") != config_hash:
        raise RuntimeError("receipt/config hash mismatch")
    if execution_config.get("expected_counts") != EXPECTED_COUNTS:
        raise RuntimeError("execution-config counts mismatch")
    if receipt.get("expected_counts") != EXPECTED_COUNTS:
        raise RuntimeError("receipt counts mismatch")
    if receipt.get("clone_count") != 3456 or receipt.get("candidate_count") != 216:
        raise RuntimeError("receipt census mismatch")
    if Path(execution_config.get("run_root", "")).resolve() != run_root:
        raise RuntimeError("run-root mismatch")
    for field, receipt_field in (
        ("generators", "generator_pins"),
        ("readouts", "readout_pins"),
        ("source_pins", "source_pins"),
    ):
        if receipt.get(receipt_field) != execution_config.get(field):
            raise RuntimeError(f"receipt/config contract mismatch: {field}")

    raw_clones = receipt.get("clones")
    if not isinstance(raw_clones, list) or len(raw_clones) != 3456:
        raise RuntimeError("receipt clone rows mismatch")
    speakers = tuple(sorted({row.get("speaker") for row in raw_clones}))
    if len(speakers) != 54 or any(not isinstance(item, str) for item in speakers):
        raise RuntimeError("speaker census mismatch")
    expected = expected_keys(speakers)
    observed: set[tuple[str, str, int, str, str]] = set()
    records = []
    for index, row in enumerate(raw_clones, start=1):
        try:
            key = (
                row["speaker"],
                row["system"],
                int(row["text_index"]),
                row["prompt_mic"],
                row["seed_arm"],
            )
            clone_path = Path(row["path"]).resolve()
            clone_hash = row["sha256"]
            ledger_path = Path(row["ledger_path"]).resolve()
            ledger_hash = row["ledger_sha256"]
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"invalid receipt clone row: {index}") from exc
        if key not in expected or key in observed:
            raise RuntimeError(f"unexpected or duplicate clone key: {key}")
        observed.add(key)
        speaker, system, text, mic, arm = key
        expected_clone = (
            run_root / "clones" / f"{system}__{mic}__seed{arm}" / f"{speaker}_t{text}.wav"
        ).resolve()
        if clone_path != expected_clone:
            raise RuntimeError(f"clone layout mismatch: {key}")
        if sha256(clone_path) != clone_hash:
            raise RuntimeError(f"clone hash mismatch: {key}")
        if ledger_path != clone_path.with_suffix(clone_path.suffix + ".ledger.json"):
            raise RuntimeError(f"ledger layout mismatch: {key}")
        if sha256(ledger_path) != ledger_hash:
            raise RuntimeError(f"ledger hash mismatch: {key}")
        cache_name = f"{cache_key(clone_path)}.npz"
        cache_path = run_root / "feature-cache" / cache_name
        if not cache_path.is_file():
            raise RuntimeError(f"missing cache: {key}")
        records.append(
            {
                "speaker": speaker,
                "system": system,
                "text_index": text,
                "prompt_mic": mic,
                "seed_arm": arm,
                "clone_relative_path": relative_to(clone_path, run_root, "clone"),
                "clone_sha256": clone_hash,
                "ledger_relative_path": relative_to(ledger_path, run_root, "ledger"),
                "ledger_sha256": ledger_hash,
                "cache_file": cache_name,
                "cache_sha256": sha256(cache_path),
            }
        )
        if index % 500 == 0:
            print(f"INPUT_AUTH_PROGRESS {index}/3456", flush=True)
    if observed != expected:
        raise RuntimeError(
            f"clone keyset mismatch missing={len(expected-observed)} extra={len(observed-expected)}"
        )

    manifest = {
        "schema": "exp206-input-manifest-v1",
        "status": "OUTCOME_BLIND_INPUT_FREEZE_CANDIDATE",
        "source_experiment": "EXP-205",
        "execution_config_sha256": config_hash,
        "execution_receipt_sha256": receipt_hash,
        "selection_manifest_sha256": receipt["manifest_sha256"],
        "exp205_scores_sha256_not_opened": receipt["scores_sha256"],
        "readout_cache_pin": readout_pin(execution_config),
        "counts": {
            "speakers": 54,
            "systems": 4,
            "texts": 4,
            "prompt_microphones": 2,
            "seed_arms": 2,
            "clones": 3456,
            "comparisons_per_speaker_per_direction": 288,
        },
        "speakers": list(speakers),
        "clones": sorted(
            records,
            key=lambda item: (
                item["speaker"],
                item["system"],
                item["text_index"],
                item["prompt_mic"],
                item["seed_arm"],
            ),
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("EXP206_INPUT_MANIFEST_BUILT clones=3456 outcomes_opened=0", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
