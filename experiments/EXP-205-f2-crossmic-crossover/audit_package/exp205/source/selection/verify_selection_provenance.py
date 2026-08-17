#!/usr/bin/env python3
"""Verify the complete public roster and within-roster pair-selection history."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parents[1]
HISTORY = PACKAGE / "source" / "history"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    manifest = load(PACKAGE / "manifest.portable.json")
    feasibility = load(HERE / "feasibility.portable.json")
    gate = load(HISTORY / "gate.json")
    tier1_pairs = load(HISTORY / "pairs.json")
    tier2_pairs = load(HISTORY / "tier2_pairs.json")
    tier2_selection = load(HERE / "tier2_selection.json")

    original_feasibility = "9c42c3245c307e9f208a2dcc73d877352b2dfda83813d9672c762be9fa15b4f6"
    historical_builder = "95e243aba7f901773f0e7b42729aea7c2160e0919869479a5e7a5b4af2f392d1"
    if feasibility["publication_portability"]["historical_source_sha256"] != original_feasibility:
        raise AssertionError("portable feasibility does not identify the historical census")
    if manifest["input_hashes"]["feasibility"] != original_feasibility:
        raise AssertionError("selection manifest binds another feasibility census")
    if manifest["input_hashes"]["feasibility_builder"] != historical_builder:
        raise AssertionError("selection manifest binds another feasibility builder")
    if sha256(HERE / "feasibility.historical.py") != historical_builder:
        raise AssertionError("published historical feasibility source differs")
    if manifest["input_hashes"]["exp204_tier2"] != (
        "428d51bff3511beef37b39886cb36161ccf7969703f6829714f5c99143e16d53"
    ):
        raise AssertionError("historical Tier-2 trust root differs")

    passing = sorted(row["spk"] for row in gate["per_speaker"] if row["passed"])
    tier1 = set(tier1_pairs)
    tier2 = set(tier2_pairs)
    evaluation = {row["speaker"] for row in manifest["speakers"]}
    if tier1 != set(passing[:30]) or gate["n_speakers_kept"] != 30:
        raise AssertionError("Tier-1 is not the lexicographically first 30 passing speakers")
    tier2_from_report = {row["spk"] for row in tier2_selection["per_speaker"]}
    if tier2 != tier2_from_report or tier2_selection["n_speakers"] != 24:
        raise AssertionError("Tier-2 selection report and roster disagree")
    if any((tier1 & tier2, tier1 & evaluation, tier2 & evaluation)):
        raise AssertionError("development and evaluation rosters overlap")
    if len(tier1 | tier2 | evaluation) != 108 or feasibility["eligible_exp203_speakers"] != 108:
        raise AssertionError("108-speaker ancestry census does not close")
    if evaluation != set(feasibility["heldout_ids"]):
        raise AssertionError("portable feasibility and evaluation roster disagree")

    feasibility_rows = {row["speaker"]: row for row in feasibility["per_speaker"]}
    for record in manifest["speakers"]:
        speaker = record["speaker"]
        selected = feasibility_rows[speaker][
            "draft_pair_duration_le_0.25_relative_rate_le_0.05"
        ]
        if selected is None:
            raise AssertionError(f"no frozen-gate pair for {speaker}")
        expected = {
            "A_mic1": selected["a"],
            "A_mic2": selected["a_mic2"],
            "B_mic1": selected["b"],
            "B_mic2": selected["b_mic2"],
        }
        observed = {key: record["audio"][key]["path"] for key in expected}
        if observed != expected:
            raise AssertionError(f"manifest does not use the frozen selected pair for {speaker}")

    if len(feasibility_rows) != 54 or feasibility["heldout_with_existing_exp204_clone"]:
        raise AssertionError("held-out feasibility census differs")
    print(
        "PASS — 108-speaker ancestry, Tier-1/Tier-2 exclusions, "
        "54-speaker feasibility and every selected A/B pair verify"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
