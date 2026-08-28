#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy==2.4.6"]
# ///
"""Post-result descriptive sensitivity to this work's ECAPA-informed ancestry.

this work is the 54-speaker complement of this work tiers 1 and 2.  Tier 1 applied
an ECAPA gate to 105 speakers, then kept the first 30 passing speakers.  This
script stratifies the this work result by whether each remaining speaker passed
or failed that historical gate, while retaining a separate not-evaluated
stratum for speakers absent from its record.  It cannot undo tier-2
conditioning: all 24 tier-2-qualified speakers were consumed, so no
tier-2-qualified stratum remains inside this work.  No stratum changes the frozen
pooled verdict.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


EXP = Path(__file__).resolve().parent
ROOT = EXP.parent
DATA = ROOT / "data"
RESULT = DATA / "result.json"
GATE = DATA / "selection" / "gate.json"
TIER1 = DATA / "selection" / "pairs.json"
TIER2 = DATA / "selection" / "tier2_pairs.json"
OUT = DATA / "roster_ancestry_sensitivity.json"

PINS = {
    RESULT: "e9dcd48dcce8d44576c4915c743142faaba3318f0c43cb426c2cb4ef89593610",
    GATE: "e2136b97c13094cbed10e2194940abe4bb36d8147a7969c7f4bcb7fa94d6b910",
    TIER1: "2d5db209cf99684b42f2112bc62ae3bdeaefb9d6d6bce849580f75596cd627a9",
    TIER2: "c520ca6c976e61b1dfbcd3dbb350e9875924fe099c66525838b0e5ff3e1014b2",
}
BOOTSTRAPS = 100_000
SEED = 2052028


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summary(values: np.ndarray, indices: np.ndarray) -> dict:
    distribution = values[indices].mean(axis=1)
    lo, hi = np.quantile(distribution, (0.025, 0.975))
    return {
        "n_speakers": int(len(values)),
        "point": float(values.mean()),
        "descriptive_composition_interval_95": [float(lo), float(hi)],
        "speakers_above_half": int((values > .5).sum()),
        "speakers_equal_half": int((values == .5).sum()),
    }


def main() -> int:
    for path, expected in PINS.items():
        observed = sha256(path)
        if observed != expected:
            raise RuntimeError(f"pin mismatch {path}: {observed}")
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    gate = json.loads(GATE.read_text(encoding="utf-8"))
    tier1 = set(json.loads(TIER1.read_text(encoding="utf-8")))
    tier2 = set(json.loads(TIER2.read_text(encoding="utf-8")))
    roster = result["counts"]["speaker_ids"]
    roster_set = set(roster)
    if len(tier1) != 30 or len(tier2) != 24 or len(roster_set) != 54:
        raise RuntimeError("historical roster counts changed")
    if tier1 & tier2 or tier1 & roster_set or tier2 & roster_set:
        raise RuntimeError("historical rosters overlap")
    if len(tier1 | tier2 | roster_set) != 108:
        raise RuntimeError("historical roster union is not 108")

    gate_status = {row["spk"]: bool(row["passed"]) for row in gate["per_speaker"]}
    strata = {
        "tier1_gate_pass_not_selected": [speaker for speaker in roster if gate_status.get(speaker) is True],
        "tier1_gate_fail": [speaker for speaker in roster if gate_status.get(speaker) is False],
        "tier1_gate_not_evaluated": [speaker for speaker in roster if speaker not in gate_status],
    }
    if {key: len(value) for key, value in strata.items()} != {
        "tier1_gate_pass_not_selected": 36,
        "tier1_gate_fail": 15,
        "tier1_gate_not_evaluated": 3,
    }:
        raise RuntimeError("unexpected gate strata")

    output = {
        "schema": "exp205-postreview-roster-ancestry-sensitivity-v1",
        "status": "POST_HOC_DESCRIPTIVE_NO_VERDICT_CHANGE",
        "question": "Does the this work direction persist across the observable residual strata of the prior tier-1 ECAPA gate?",
        "limitations": [
            "The tier-1 gate was ECAPA-based and the 30 kept speakers were the lexicographically first passing speakers.",
            "Three this work speakers were absent from the historical tier-1 gate record and are reported separately rather than assigned a status.",
            "All 24 tier-2-qualified speakers were consumed; this work is tier-2-ineligible by construction.",
            "This sensitivity cannot recover an embedding-naive roster or population inference.",
            "Intervals are descriptive within-stratum composition summaries and are not multiplicity-adjusted.",
            "No stratum can alter or rescue the frozen pooled verdict.",
        ],
        "pins": {path.name: expected for path, expected in PINS.items()},
        "ancestry_counts": {
            "tier1": 30,
            "tier2": 24,
            "exp205_complement": 54,
            "tier1_gate_considered": len(gate_status),
            "tier1_gate_pass_total": sum(gate_status.values()),
            "tier1_gate_fail_total": len(gate_status) - sum(gate_status.values()),
            "exp205_gate_pass_not_selected": 36,
            "exp205_gate_fail": 15,
            "exp205_gate_not_evaluated": 3,
        },
        "strata": {},
        "bootstrap": {"replicates": BOOTSTRAPS, "seed": SEED},
    }
    rng = np.random.default_rng(SEED)
    index = {speaker: position for position, speaker in enumerate(roster)}
    for label, speakers in strata.items():
        node = {"speaker_ids": speakers, "directions": {}}
        positions = [index[speaker] for speaker in speakers]
        indices = rng.integers(0, len(speakers), size=(BOOTSTRAPS, len(speakers)), dtype=np.int32)
        for direction in ("primary_mic1_to_mic2", "reverse_mic2_to_mic1"):
            node["directions"][direction] = {}
            for encoder in ("ecapa", "wavlm"):
                vector = np.asarray(result["directions"][direction][encoder]["speaker_means"], dtype=np.float64)
                node["directions"][direction][encoder] = summary(vector[positions], indices)
        output["strata"][label] = node
    OUT.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({label: value["directions"] for label, value in output["strata"].items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
