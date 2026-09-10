#!/usr/bin/env python3
"""Build the outcome-blind selection manifest for EXP-213: the 54 tier speakers.

The roster is EXP-204 tier 1 (30 speakers, pairs.json) plus tier 2 (24 speakers,
tier2_pairs.json): exactly the complement of the EXP-205 roster within the 108 VCTK
speakers with paired mic1/mic2 captures. Pair selection is the EXP-205 rule, reused
from EXP-212's builder (feasibility.py imported there): 4-10 s paired utterances,
duration gap <= .25 s and relative seconds-per-UTF-8-byte gap <= .05 on each
microphone, ranked by the EXP-205 tuple. As in EXP-205, the only exclusion is the
speaker's EXP-203 seed utterance, which the records hold for every tier speaker; the
pair the rule would pick with no exclusion at all is recorded alongside. The EXP-204
tier pairs are not excluded; coincidence with them is recorded as information only.
Never opens clone audio, embeddings, scores or outcomes.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import os

# Historical run layout: these scripts document what ran. Point the two roots at copies of
# the private run directories and the experiment tree; nothing here reads the release itself.
RUNS = Path(os.environ.get("ICASSP_RUNS", "/runs"))
EXPERIMENTS = Path(os.environ.get("ICASSP_EXPERIMENTS", "/experiments"))
HF = Path(os.environ.get("HF_HUB_CACHE", "~/.cache/huggingface/hub")).expanduser()
EXP213 = EXPERIMENTS / "EXP-213-f2-full-roster-complement"
EXP204 = EXPERIMENTS / "EXP-204-f2-seed-crossover"
TIER1 = EXP204 / "pairs.json"
TIER2 = EXP204 / "tier2_pairs.json"
EXP203_POOL = EXPERIMENTS / "EXP-203-f2-vctk-crossmic/pools/pool_seed1_cand1.json"
EXP212_BUILDER = EXPERIMENTS / "EXP-212-f2-fresh-pair-replication/build_fresh_manifest.py"
PINS = {
    TIER1: "0ec8e8515ce4e478d4ccf37d3aca2c9f9d787d98ab813ee41321f85dab0d62d4",
    TIER2: "428d51bff3511beef37b39886cb36161ccf7969703f6829714f5c99143e16d53",
    EXP203_POOL: "8ef413f1cf9fb493516c54ed27c925710605b388341453fa69a436401ee16d7a",
}


def load_exp212():
    spec = importlib.util.spec_from_file_location("exp212_builder", EXP212_BUILDER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


B = load_exp212()


def tier_rosters() -> tuple[dict[str, list[set[str]]], dict[str, str]]:
    """Historical pairs (as sets of utterance stems) and tier label per speaker."""
    tier1 = json.loads(TIER1.read_text(encoding="utf-8"))
    tier2 = json.loads(TIER2.read_text(encoding="utf-8"))
    pairs = {s: [{B.stem(Path(r["seed_A"])), B.stem(Path(r["seed_B"]))}] for s, r in tier1.items()}
    pairs.update({s: [{B.stem(Path(r[d]["seed_A"])), B.stem(Path(r[d]["seed_B"]))} for d in ("low", "high")]
                  for s, r in tier2.items()})
    tier = {**{s: "tier1" for s in tier1}, **{s: "tier2" for s in tier2}}
    return pairs, tier


def paired_capture_roster() -> set[str]:
    """Every VCTK speaker directory holding both mic1 and mic2 captures."""
    return {d.name for d in B.VCTK_AUDIO.iterdir()
            if d.is_dir() and any(d.glob("*_mic1.flac")) and any(d.glob("*_mic2.flac"))}


def historical_overlap(pair: dict, history: list[set[str]]) -> dict:
    selected = {B.stem(Path(pair["a"])), B.stem(Path(pair["b"]))}
    shared = max(len(selected & h) for h in history)
    return {"shared_utterances_with_closest_tier_pair": shared,
            "coincides_with_tier_pair": shared == 2,
            "tier_pairs": [sorted(h) for h in history]}


def speaker_row(speaker: str, pair: dict, tier: str, seed: Path, history: list[set[str]]) -> dict:
    paths = {"A_mic1": Path(pair["a"]), "A_mic2": Path(pair["a_mic2"]),
             "B_mic1": Path(pair["b"]), "B_mic2": Path(pair["b_mic2"])}
    audio = {key: B.audio_record(path) for key, path in paths.items()}
    texts = {}
    for arm in ("A", "B"):
        text_path = B.FZ.transcript_path(paths[f"{arm}_mic1"])
        text = text_path.read_text(encoding="utf-8").strip()
        texts[arm] = {"path": str(text_path), "sha256": B.sha256(text_path),
                      "utf8_bytes": len(text.encode("utf-8")), "text": text}
    unexcluded = B.select_pair(speaker, set())
    return {
        "speaker": speaker,
        "tier": tier,
        "audio": audio,
        "transcripts": texts,
        "match": {k: pair[k] for k in ("duration_gap_mic1_s", "duration_gap_mic2_s",
                                       "relative_rate_gap_mic1", "relative_rate_gap_mic2")},
        "excluded_exp203_seed": {"path": str(seed), "sha256": B.sha256(seed)},
        "selection_without_seed_exclusion": "same" if unexcluded == pair else {"a": unexcluded["a"], "b": unexcluded["b"]},
        "historical_tier_pair_overlap": historical_overlap(pair, history),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=EXP213 / "selection-manifest-213.json")
    args = parser.parse_args()
    for path, expected in PINS.items():
        if B.sha256(path) != expected:
            raise ValueError(f"pin mismatch {path}")
    original = json.loads(B.ORIGINAL.read_text(encoding="utf-8"))
    exp205 = {s["speaker"] for s in original["speakers"]}
    history, tier = tier_rosters()
    roster = sorted(tier)
    if len(roster) != 54 or set(roster) & exp205 or (set(roster) | exp205) != paired_capture_roster():
        raise ValueError("tier roster is not the exact complement of EXP-205 within the paired-capture corpus")
    seeds = {s: Path(r["gen_ref"]["path"]) for s, r in json.loads(EXP203_POOL.read_text(encoding="utf-8"))["trials"].items()}

    speakers, dropped = [], []
    for speaker in roster:
        pair = B.select_pair(speaker, {B.stem(seeds[speaker])})
        if pair is None:
            dropped.append({"speaker": speaker, "tier": tier[speaker], "reason": "no pair satisfies the rule"})
            continue
        speakers.append(speaker_row(speaker, pair, tier[speaker], seeds[speaker], history[speaker]))
        print(f"{speaker} {tier[speaker]} {Path(pair['a']).name} {Path(pair['b']).name}", flush=True)

    n = len(speakers)
    result = {
        "schema": "exp213-full-roster-complement-manifest-v1",
        "scope": "real_audio_metadata_and_text_only_no_clone_or_score_outcomes",
        "corpus": original["corpus"],
        "selection": {
            **original["selection"],
            "roster": "the 54 EXP-204 tier-1 and tier-2 speakers: the complement of the EXP-205 roster within the 108 paired-capture VCTK speakers",
            "excluded_per_speaker": ["EXP-203 seed utterance (as in EXP-205)"],
            "exp204_tier_pairs_excluded": False,
        },
        "counts": {"speakers": n, "seed_arms": 2, "prompt_microphones": 2, "generated_texts": 4, "systems": 4,
                   "expected_clones": n * 64, "comparisons_per_speaker_per_direction": 32},
        "dropped_speakers": dropped,
        "generation": original["generation"],
        "speakers": speakers,
        "input_hashes": {
            "exp205_manifest": B.sha256(B.ORIGINAL),
            "feasibility_source": B.sha256(B.FEASIBILITY_SOURCE),
            "exp212_builder": B.sha256(EXP212_BUILDER),
            "exp204_tier1": PINS[TIER1], "exp204_tier2": PINS[TIER2], "exp203_pool": PINS[EXP203_POOL],
            "builder": B.sha256(Path(__file__).resolve()),
        },
    }
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "MANIFEST_BUILT_NO_OUTCOMES", "speakers": n, "dropped": len(dropped),
                      "expected_clones": n * 64, "manifest_sha256": B.sha256(args.out)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
