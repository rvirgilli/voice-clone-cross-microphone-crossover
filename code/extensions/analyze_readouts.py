#!/usr/bin/env python3
"""Score the EXP-205 A/B grid under every extracted readout, raw and speaker-centred.

For each readout the statistic is the EXP-205 estimand: per comparison, 1 if the clone
is closer (cosine) to the opposite-microphone capture of its own event than to the
other event's capture, 1/2 on ties, 0 otherwise; averaged within speaker, then equally
over the 54 speakers; percentile intervals from whole-speaker resampling with the
frozen seed. The speaker-centred variant subtracts, per speaker, the mean vector of
that speaker's four real captures from every vector before the cosine.

Output: readout_roster.json with every cell, plus the two predeclared readings.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np


# Historical run layout: this script documents what ran. Point the two roots at copies of
# the private run directories and the experiment tree; nothing here reads the release itself.
RUNS = Path(os.environ.get("ICASSP_RUNS", "/runs"))
EXPERIMENTS = Path(os.environ.get("ICASSP_EXPERIMENTS", "/experiments"))
EXP205 = EXPERIMENTS / "EXP-205-f2-crossmic-crossover"
RUN205 = RUNS / "EXP-205-f2-crossmic-crossover"
BOOTSTRAPS = 10_000
SEED = 2052027
SPEAKERS = 54
DIRECTIONS = {"mic1": "primary_mic1_to_mic2", "mic2": "reverse_mic2_to_mic1"}


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def followed(own: float, other: float) -> float:
    return 1.0 if own > other else 0.0 if own < other else 0.5


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    feat_dir = Path(args.features)

    manifest = json.loads((EXP205 / "selection-manifest.json").read_text(encoding="utf-8"))
    real = {}
    for speaker in manifest["speakers"]:
        for key in ("A_mic1", "A_mic2", "B_mic1", "B_mic2"):
            real[(speaker["speaker"], key)] = speaker["audio"][key]["path"]
    rows = []
    with (RUN205 / "scores.tsv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            rows.append(row)
    if len(rows) != 3456:
        raise RuntimeError("score census is not 3,456 rows")
    speakers = sorted({row["speaker"] for row in rows})
    if len(speakers) != SPEAKERS:
        raise RuntimeError("speaker census is not 54")

    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, SPEAKERS, size=(BOOTSTRAPS, SPEAKERS))
    output = {"schema": "exp207-readout-roster-v1", "status": "POST_HOC_DESCRIPTIVE_NO_VERDICT_CHANGE",
              "bootstrap": {"unit": "speaker", "replicates": BOOTSTRAPS, "seed": SEED}, "readouts": {}}

    for npz_path in sorted(feat_dir.glob("*.npz")):
        data = np.load(npz_path)
        index = {path: i for i, path in enumerate(data["paths"].tolist())}
        for readout in sorted(k for k in data.files if k != "paths"):
            matrix = data[readout].astype(np.float64)
            for centred in (False, True):
                vectors = matrix.copy()
                if centred:
                    for speaker in speakers:
                        real_idx = [index[real[(speaker, key)]] for key in ("A_mic1", "A_mic2", "B_mic1", "B_mic2")]
                        mean = vectors[real_idx].mean(axis=0)
                        member = [index[str(RUN205 / row["clone_path"])] for row in rows if row["speaker"] == speaker] + real_idx
                        vectors[member] -= mean
                per_speaker = {d: {s: [] for s in speakers} for d in DIRECTIONS.values()}
                for row in rows:
                    clone = vectors[index[str(RUN205 / row["clone_path"])]]
                    candidate_mic = "mic2" if row["prompt_mic"] == "mic1" else "mic1"
                    own = vectors[index[real[(row["speaker"], f"{row['seed_arm']}_{candidate_mic}")]]]
                    other_arm = "B" if row["seed_arm"] == "A" else "A"
                    other = vectors[index[real[(row["speaker"], f"{other_arm}_{candidate_mic}")]]]
                    per_speaker[DIRECTIONS[row["prompt_mic"]]][row["speaker"]].append(followed(cosine(clone, own), cosine(clone, other)))
                cell = {}
                for direction, table in per_speaker.items():
                    means = np.array([np.mean(table[s]) for s in speakers])
                    if any(len(table[s]) != 32 for s in speakers):
                        raise RuntimeError("expected 32 comparisons per speaker per direction")
                    boot = means[draws].mean(axis=1)
                    cell[direction] = {"point": float(means.mean()),
                                       "stability_interval_95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                                       "speakers_above_chance": int((means > 0.5).sum())}
                output["readouts"][f"{readout}{'__centred' if centred else ''}"] = cell
                print(f"{readout:18s} {'centred' if centred else 'raw':8s} "
                      f"P {cell['primary_mic1_to_mic2']['point']:.3f} [{cell['primary_mic1_to_mic2']['stability_interval_95'][0]:.3f},{cell['primary_mic1_to_mic2']['stability_interval_95'][1]:.3f}]  "
                      f"R {cell['reverse_mic2_to_mic1']['point']:.3f} [{cell['reverse_mic2_to_mic1']['stability_interval_95'][0]:.3f},{cell['reverse_mic2_to_mic1']['stability_interval_95'][1]:.3f}]", flush=True)

    # Predeclared readings (see PREREG.md).
    def lower(name: str, direction: str) -> float:
        return output["readouts"][name][direction]["stability_interval_95"][0]

    non_sv = [n for n in output["readouts"] if n.startswith(("wavlm_L", "w2v2_L", "whisper", "ltas"))]
    q1 = [n for n in non_sv if lower(n, "primary_mic1_to_mic2") > 0.70 and lower(n, "reverse_mic2_to_mic1") > 0.70]
    sv_layers = [n for n in output["readouts"] if n.startswith("wavlmsv_L") and not n.endswith("__centred")]
    best_layer = max(sv_layers, key=lambda n: output["readouts"][n]["primary_mic1_to_mic2"]["point"]) if sv_layers else None
    q2 = None
    if best_layer and "wavlmsv_xvector" in output["readouts"]:
        gaps = [output["readouts"][best_layer][d]["point"] - output["readouts"]["wavlmsv_xvector"][d]["point"] for d in DIRECTIONS.values()]
        q2 = {"best_hidden_layer": best_layer, "gap_over_xvector_head": gaps, "reading": "SV_HEAD_SUPPRESSES" if min(gaps) > 0.05 else "NO_SUPPRESSION_EVIDENCE"}
    output["readings"] = {
        "Q1_non_sv_readouts_with_both_lower_endpoints_above_.70": q1,
        "Q1_reading": "NOT_SV_SPECIFIC" if q1 else "NO_NON_SV_READOUT_CLEARS_.70",
        "Q2_wavlmsv_layer_profile": q2,
    }
    Path(args.out).write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output["readings"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
