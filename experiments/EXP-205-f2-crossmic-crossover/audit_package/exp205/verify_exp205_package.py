#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy==2.4.6"]
# ///
"""Verify the portable EXP-205 package and independently recompute its result."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
PACKAGE = HERE / "audit_package" / "exp205" if (HERE / "audit_package" / "exp205").is_dir() else HERE
SYSTEMS = ("f5", "xtts", "cosy", "seedvc")
DIRECTIONS = ("primary_mic1_to_mic2", "reverse_mic2_to_mic1")
ENCODERS = ("ecapa", "wavlm")
BOOTSTRAPS = 100_000
SEED = 2052027


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(name: str):
    return json.loads((PACKAGE / name).read_text(encoding="utf-8"))


def close(observed, expected, label: str, tolerance: float = 2e-15) -> None:
    if not math.isclose(float(observed), float(expected), rel_tol=0.0, abs_tol=tolerance):
        raise AssertionError(f"{label}: {observed} != {expected}")


def main() -> int:
    audit = load("audit.json")
    if audit.get("schema") != "exp205-portable-audit-v1":
        raise AssertionError("audit schema invalid")
    for name, record in audit["artifacts"].items():
        path = PACKAGE / name
        if sha256(path) != record["sha256"] or path.stat().st_size != record["size"]:
            raise AssertionError(f"artifact mismatch: {name}")
    for label, record in audit["source_records"].items():
        if sha256(PACKAGE / record["path"]) != record["sha256"]:
            raise AssertionError(f"source mismatch: {label}")

    # Portable public files must not leak local machine roots.
    for name in ("manifest.portable.json", "receipt.portable.json", "execution-config.portable.json",
                 "analysis-config-recovery.portable.json", "scores.portable.tsv"):
        text = (PACKAGE / name).read_text(encoding="utf-8")
        if any(root in text for root in ("/home/", "/mnt/", "/media/")):
            raise AssertionError(f"private path in {name}")

    result = load("result.json")
    manifest = load("manifest.portable.json")
    receipt = load("receipt.portable.json")
    if result["status"] != "SCIENTIFIC_RESULT" or manifest["counts"]["speakers"] != 54:
        raise AssertionError("result/manifest scope invalid")
    speakers = result["counts"]["speaker_ids"]
    if len(speakers) != 54 or len(set(speakers)) != 54:
        raise AssertionError("speaker census invalid")

    with (PACKAGE / "scores.portable.tsv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if len(rows) != 3456:
        raise AssertionError("score row count invalid")
    identities = [tuple(row[k] for k in ("speaker", "system", "text_index", "prompt_mic", "seed_arm")) for row in rows]
    if len(set(identities)) != 3456:
        raise AssertionError("duplicate score identity")
    expected = {
        (speaker, system, str(text), mic, arm)
        for speaker in speakers
        for system in SYSTEMS
        for text in range(4)
        for mic in ("mic1", "mic2")
        for arm in ("A", "B")
    }
    if set(identities) != expected:
        raise AssertionError("score identity grid invalid")

    receipt_ids = {
        (r["speaker"], r["system"], str(r["text_index"]), r["prompt_mic"], r["seed_arm"]): r
        for r in receipt["clones"]
    }
    if set(receipt_ids) != expected or receipt["clone_count"] != 3456:
        raise AssertionError("receipt identity grid invalid")
    candidate_bindings = {}
    clone_hashes = set()
    for row in rows:
        key = tuple(row[k] for k in ("speaker", "system", "text_index", "prompt_mic", "seed_arm"))
        if receipt_ids[key]["sha256"] != row["clone_sha256"]:
            raise AssertionError(f"clone hash/receipt mismatch: {key}")
        clone_hashes.add(row["clone_sha256"])
        for prefix in ("candidate_A", "candidate_B", "same_candidate_A", "same_candidate_B"):
            path, digest = row[f"{prefix}_path"], row[f"{prefix}_sha256"]
            if path in candidate_bindings and candidate_bindings[path] != digest:
                raise AssertionError(f"candidate hash inconsistency: {path}")
            candidate_bindings[path] = digest
    if len(clone_hashes) != 3456 or len(candidate_bindings) != 216:
        raise AssertionError("clone/candidate census invalid")

    followed = defaultdict(list)
    same_followed = defaultdict(list)
    system_values = defaultdict(list)
    for row in rows:
        speaker, system, _, mic, arm = (row[k] for k in ("speaker", "system", "text_index", "prompt_mic", "seed_arm"))
        direction = DIRECTIONS[0] if mic == "mic1" else DIRECTIONS[1]
        diagnostic = f"same_{mic}"
        other = "B" if arm == "A" else "A"
        for encoder in ENCODERS:
            own, alt = float(row[f"{encoder}_{arm}"]), float(row[f"{encoder}_{other}"])
            same_own = float(row[f"{encoder}_same_{arm}"])
            same_alt = float(row[f"{encoder}_same_{other}"])
            value = 1.0 if own > alt else 0.0 if own < alt else 0.5
            same_value = 1.0 if same_own > same_alt else 0.0 if same_own < same_alt else 0.5
            followed[(direction, encoder, speaker)].append(value)
            same_followed[(diagnostic, encoder, speaker)].append(same_value)
            system_values[(direction, encoder, system)].append(value)

    vectors = {}
    same_vectors = {}
    for direction in DIRECTIONS:
        for encoder in ENCODERS:
            vectors[(direction, encoder)] = np.asarray(
                [np.mean(followed[(direction, encoder, speaker)]) for speaker in speakers], dtype=np.float64
            )
    for diagnostic in ("same_mic1", "same_mic2"):
        for encoder in ENCODERS:
            same_vectors[(diagnostic, encoder)] = np.asarray(
                [np.mean(same_followed[(diagnostic, encoder, speaker)]) for speaker in speakers], dtype=np.float64
            )
    if any(len(values) != 32 for values in followed.values()) or any(len(values) != 32 for values in same_followed.values()):
        raise AssertionError("speaker cell does not contain 32 comparisons")

    rng = np.random.default_rng(SEED)
    indices = rng.integers(0, 54, size=(BOOTSTRAPS, 54), dtype=np.int32)
    for (direction, encoder), vector in vectors.items():
        node = result["directions"][direction][encoder]
        close(vector.mean(), node["point"], f"{direction}/{encoder}/point")
        if not np.array_equal(vector, np.asarray(node["speaker_means"], dtype=np.float64)):
            raise AssertionError(f"speaker vector mismatch: {direction}/{encoder}")
        lo, hi = np.quantile(vector[indices].mean(axis=1), (0.025, 0.975))
        close(lo, node["stability_interval_95"][0], f"{direction}/{encoder}/lo")
        close(hi, node["stability_interval_95"][1], f"{direction}/{encoder}/hi")
    for (diagnostic, encoder), vector in same_vectors.items():
        node = result["same_microphone_diagnostics_no_verdict"][diagnostic][encoder]
        close(vector.mean(), node["point"], f"{diagnostic}/{encoder}/point")
        lo, hi = np.quantile(vector[indices].mean(axis=1), (0.025, 0.975))
        close(lo, node["stability_interval_95"][0], f"{diagnostic}/{encoder}/lo")
        close(hi, node["stability_interval_95"][1], f"{diagnostic}/{encoder}/hi")
    for direction in DIRECTIONS:
        for encoder in ENCODERS:
            for system in SYSTEMS:
                close(
                    np.mean(system_values[(direction, encoder, system)]),
                    result["system_points_no_intervals"][direction][encoder][system],
                    f"system/{direction}/{encoder}/{system}",
                )

    p_ecapa = result["directions"][DIRECTIONS[0]]["ecapa"]
    p_wavlm = result["directions"][DIRECTIONS[0]]["wavlm"]
    r_ecapa = result["directions"][DIRECTIONS[1]]["ecapa"]
    r_wavlm = result["directions"][DIRECTIONS[1]]["wavlm"]
    bars = (
        p_ecapa["stability_interval_95"][0] > .5,
        p_ecapa["point"] >= .8 and p_ecapa["stability_interval_95"][0] > .7,
        p_wavlm["stability_interval_95"][0] > .5,
        r_ecapa["stability_interval_95"][0] > .5,
        r_wavlm["stability_interval_95"][0] > .5,
    )
    if bars != (True, True, True, True, True):
        raise AssertionError(f"frozen bars did not all pass: {bars}")
    if result["headline_permission"] != "BIDIRECTIONAL_HEADLINE_PERMITTED":
        raise AssertionError("headline permission mismatch")

    print("PASS — portable EXP-205 census, provenance, statistics and verdict verify")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
