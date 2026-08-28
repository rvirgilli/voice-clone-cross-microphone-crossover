#!/usr/bin/env python3
"""Descriptive arm-balance and paired-capture integrity checks.

This post-result analysis authenticates the portable score table and manifest,
then reports every arm-specific follow rate and verifies that each selected
mic1/mic2 candidate pair has identical sample metadata.  It is descriptive,
does not alter the frozen verdict, and performs no model inference.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path


EXP = Path(__file__).resolve().parent
DATA = EXP.parent / "data"
SCORES = DATA / "scores.tsv"
MANIFEST = DATA / "trials.json"
OUT = EXP / "arm_pairing_sensitivity.json"
PINS = {
    SCORES: "535d3a5cba9f98bb830175004214b62977047f629113248ecca07c11cbd9c387",
    MANIFEST: "d4def5472b514c6a756315990c3decaa488d6aa7896223d9c1dc42825f430417",
}
SYSTEMS = ("f5", "xtts", "cosy", "seedvc")
ENCODERS = ("ecapa", "wavlm")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def followed(own: float, alternative: float) -> float:
    return 1.0 if own > alternative else 0.0 if own < alternative else 0.5


def summarize(values: list[float]) -> dict:
    if not values:
        raise RuntimeError("empty descriptive cell")
    return {
        "n": len(values),
        "follow_rate": sum(values) / len(values),
        "wins": sum(value == 1.0 for value in values),
        "ties": sum(value == 0.5 for value in values),
        "losses": sum(value == 0.0 for value in values),
    }


def main() -> int:
    for path, expected in PINS.items():
        observed = sha256(path)
        if observed != expected:
            raise RuntimeError(f"pin mismatch {path}: {observed}")

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    speakers = manifest["speakers"]
    if len(speakers) != 54:
        raise RuntimeError("manifest roster is not 54 speakers")
    pairing_rows = []
    for speaker in speakers:
        for arm in ("A", "B"):
            mic1 = speaker["audio"][f"{arm}_mic1"]
            mic2 = speaker["audio"][f"{arm}_mic2"]
            pairing_rows.append(
                {
                    "speaker": speaker["speaker"],
                    "arm": arm,
                    "mic1_frames": int(mic1["frames"]),
                    "mic2_frames": int(mic2["frames"]),
                    "mic1_samplerate_hz": int(mic1["samplerate_hz"]),
                    "mic2_samplerate_hz": int(mic2["samplerate_hz"]),
                }
            )
    frame_deltas = [abs(row["mic1_frames"] - row["mic2_frames"]) for row in pairing_rows]
    rate_mismatches = sum(row["mic1_samplerate_hz"] != row["mic2_samplerate_hz"] for row in pairing_rows)
    if len(pairing_rows) != 108 or max(frame_deltas) != 0 or rate_mismatches:
        raise RuntimeError("selected paired-capture metadata mismatch")

    cells: dict[tuple[str, ...], list[float]] = defaultdict(list)
    identities = set()
    with SCORES.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            identity = tuple(row[key] for key in ("speaker", "system", "text_index", "prompt_mic", "seed_arm"))
            if identity in identities:
                raise RuntimeError(f"duplicate score row: {identity}")
            identities.add(identity)
            speaker, system, _, prompt_mic, arm = identity
            direction = "primary_mic1_to_mic2" if prompt_mic == "mic1" else "reverse_mic2_to_mic1"
            other = "B" if arm == "A" else "A"
            for encoder in ENCODERS:
                value = followed(float(row[f"{encoder}_{arm}"]), float(row[f"{encoder}_{other}"]))
                cells[(direction, encoder, arm)].append(value)
                cells[(direction, encoder, arm, system)].append(value)
    if len(identities) != 3456:
        raise RuntimeError("score census is not 3,456 unique rows")

    arm_results = {}
    system_cells = []
    for direction in ("primary_mic1_to_mic2", "reverse_mic2_to_mic1"):
        arm_results[direction] = {}
        for encoder in ENCODERS:
            arm_results[direction][encoder] = {}
            for arm in ("A", "B"):
                arm_results[direction][encoder][arm] = summarize(cells[(direction, encoder, arm)])
                for system in SYSTEMS:
                    node = summarize(cells[(direction, encoder, arm, system)])
                    system_cells.append(
                        {"direction": direction, "encoder": encoder, "arm": arm, "system": system, **node}
                    )
    minimum = min(system_cells, key=lambda node: node["follow_rate"])
    output = {
        "schema": "exp205-postreview-arm-pairing-sensitivity-v1",
        "status": "POST_HOC_DESCRIPTIVE_NO_VERDICT_CHANGE",
        "pins": {path.name: expected for path, expected in PINS.items()},
        "paired_capture_integrity": {
            "selected_event_pairs": len(pairing_rows),
            "equal_frame_count_pairs": sum(delta == 0 for delta in frame_deltas),
            "maximum_absolute_frame_delta": max(frame_deltas),
            "sample_rate_mismatches": rate_mismatches,
            "interpretation": "Supports paired mic1/mic2 capture integrity; does not independently establish acquisition timing.",
        },
        "arm_results": arm_results,
        "system_direction_encoder_arm_cells": system_cells,
        "cell_census": len(system_cells),
        "cells_above_chance": sum(node["follow_rate"] > 0.5 for node in system_cells),
        "minimum_cell": minimum,
        "limitations": [
            "Arm and system cells are descriptive and have no individual intervals or multiplicity-adjusted tests.",
            "These checks cannot alter or rescue the frozen pooled verdict.",
            "Equal frame counts support pairing but are not a substitute for an acquisition-protocol statement about simultaneity.",
        ],
    }
    OUT.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"arm_results": arm_results, "minimum_cell": minimum}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
