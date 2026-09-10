#!/usr/bin/env python3
"""Score and analyse the EXP-212 fresh-pair replication.

extract: ECAPA and WavLM-SV for every clone and real capture with the EXP-207 extractor
         (the readouts used by EXP-210/211), written once per readout.
analyze: the EXP-205 estimand per direction (own-event cosine above other-event cosine,
         ties 1/2; averaged within speaker, then equally over speakers), the EXP-205
         bootstrap (imported: 100,000 whole-speaker draws, seed 2052027) and the frozen
         EXP-205 verdict module. Replication is the single condition
         headline_permission == BIDIRECTIONAL_HEADLINE_PERMITTED. Secondary, descriptive:
         per-speaker agreement with the original pairs, against the sealed EXP-205 speaker
         means and against the original grid rescored with this extractor (EXP-207 features).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

import os

# Historical run layout: these scripts document what ran. Point the two roots at copies of
# the private run directories and the experiment tree; nothing here reads the release itself.
RUNS = Path(os.environ.get("ICASSP_RUNS", "/runs"))
EXPERIMENTS = Path(os.environ.get("ICASSP_EXPERIMENTS", "/experiments"))
HF = Path(os.environ.get("HF_HUB_CACHE", "~/.cache/huggingface/hub")).expanduser()
HERE = Path(__file__).resolve().parent
EXP205 = EXPERIMENTS / "EXP-205-f2-crossmic-crossover"
EXP207 = EXPERIMENTS / "EXP-207-f2-readout-roster"
EXP212 = EXPERIMENTS / "EXP-212-f2-fresh-pair-replication"
RUN205 = RUNS / "EXP-205-f2-crossmic-crossover"
FEATURES207 = RUNS / "EXP-207-f2-readout-roster/features"
DIRECTION = {"mic1": "primary_mic1_to_mic2", "mic2": "reverse_mic2_to_mic1"}
READOUTS = {"ecapa": "ecapa", "wavlm": "wavlmsv"}
ARRAY = {"ecapa": "ecapa", "wavlm": "wavlmsv_xvector"}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def unit_vectors(npz_path: Path, array: str) -> tuple[dict[str, int], np.ndarray]:
    data = np.load(npz_path)
    vec = data[array].astype(np.float64)
    vec /= np.linalg.norm(vec, axis=1, keepdims=True)
    return {p: i for i, p in enumerate(data["paths"].tolist())}, vec


def follow(vec: np.ndarray, index: dict[str, int], clone: str, own: str, other: str) -> float:
    d = float(vec[index[clone]] @ vec[index[own]] - vec[index[clone]] @ vec[index[other]])
    return 1.0 if d > 0 else 0.0 if d < 0 else 0.5


def speaker_means(table: dict[str, list[float]], speakers: list[str]) -> np.ndarray:
    return np.array([np.mean(table[s]) for s in speakers], dtype=np.float64)


def original_grid_means(readout: str, speakers: list[str]) -> dict[str, np.ndarray]:
    """Per-speaker means of the original EXP-205 grid rescored with the EXP-207 features."""
    index, vec = unit_vectors(FEATURES207 / f"{READOUTS[readout]}.npz", ARRAY[readout])
    per = defaultdict(lambda: defaultdict(list))
    with (RUN205 / "scores.tsv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            own, other = (row["candidate_A_path"], row["candidate_B_path"]) if row["seed_arm"] == "A" else (row["candidate_B_path"], row["candidate_A_path"])
            per[DIRECTION[row["prompt_mic"]]][row["speaker"]].append(follow(vec, index, row["clone_path"], own, other))
    return {direction: speaker_means(table, speakers) for direction, table in per.items()}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=RUNS / "EXP-212-f2-fresh-pair-replication")
    parser.add_argument("stage", choices=("extract", "analyze"))
    args = parser.parse_args()
    jobs_path = args.run / "jobs.json"
    jobs = json.loads(jobs_path.read_text(encoding="utf-8"))["jobs"]
    manifest_path = EXP212 / "selection-manifest-fresh.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    real = {(s["speaker"], k): s["audio"][k]["path"] for s in manifest["speakers"] for k in s["audio"]}

    if args.stage == "extract":
        sys.path.insert(0, str(EXP207))
        import extract_readouts as er
        import torch
        paths = [j["out"] for j in jobs] + sorted(set(real.values()))
        device = "cuda" if torch.cuda.is_available() else "cpu"
        for name in READOUTS.values():
            er.run_readout(name, paths, args.run / "features", device)
        return 0

    analyze205 = load_module("exp205_analyze", EXP205 / "analyze.py")
    verdict = load_module("exp205_verdict", EXP205 / "verdict.py")
    speakers = sorted({s["speaker"] for s in manifest["speakers"]})
    sealed = json.loads((RUN205 / "sealed/scientific-result.json").read_text(encoding="utf-8"))
    sealed_pos = [sealed["counts"]["speaker_ids"].index(s) for s in speakers]

    vectors, systems, agreement = {}, {}, {}
    for readout in READOUTS:
        index, vec = unit_vectors(args.run / "features" / f"{READOUTS[readout]}.npz", ARRAY[readout])
        per = defaultdict(lambda: defaultdict(list))
        by_system = defaultdict(list)
        for job in jobs:
            direction = DIRECTION[job["prompt_mic"]]
            candidate_mic = "mic2" if job["prompt_mic"] == "mic1" else "mic1"
            other_arm = "B" if job["arm"] == "A" else "A"
            value = follow(vec, index, job["out"], real[(job["speaker"], f"{job['arm']}_{candidate_mic}")],
                           real[(job["speaker"], f"{other_arm}_{candidate_mic}")])
            per[direction][job["speaker"]].append(value)
            by_system[(direction, job["system"])].append(value)
        original = original_grid_means(readout, speakers)
        for direction in DIRECTION.values():
            if any(len(per[direction][s]) != 32 for s in speakers):
                raise RuntimeError(f"cell count invalid: {readout} {direction}")
            vectors[(direction, readout)] = speaker_means(per[direction], speakers)
            systems[(direction, readout)] = {sy: float(np.mean(by_system[(direction, sy)])) for sy in ("f5", "xtts", "cosy", "seedvc")}
            sealed_means = np.array(sealed["directions"][direction][readout]["speaker_means"])[sealed_pos]
            agreement[(direction, readout)] = {
                "sealed_original_means": sealed_means.tolist(),
                "rescored_original_means": original[direction].tolist(),
                "pearson_r_vs_sealed": float(np.corrcoef(vectors[(direction, readout)], sealed_means)[0, 1]),
                "pearson_r_vs_rescored": float(np.corrcoef(vectors[(direction, readout)], original[direction])[0, 1]),
                "sealed_original_point_retained_speakers": float(sealed_means.mean()),
                "rescored_original_point_retained_speakers": float(original[direction].mean()),
            }
    # One draw matrix for every cell, as in EXP-205 (shared seed, whole-speaker resampling).
    intervals = analyze205.bootstrap_intervals(vectors)
    differences = {key: vectors[key] - np.array(agreement[key]["rescored_original_means"]) for key in vectors}
    difference_intervals = analyze205.bootstrap_intervals(differences)
    point = {key: float(v.mean()) for key, v in vectors.items()}
    primary = verdict.primary_verdict(ecapa_point=point[("primary_mic1_to_mic2", "ecapa")],
                                      ecapa_lcb=intervals[("primary_mic1_to_mic2", "ecapa")][0],
                                      wavlm_lcb=intervals[("primary_mic1_to_mic2", "wavlm")][0])
    reverse = verdict.reverse_verdict(ecapa_lcb=intervals[("reverse_mic2_to_mic1", "ecapa")][0],
                                      wavlm_lcb=intervals[("reverse_mic2_to_mic1", "wavlm")][0])
    permission = verdict.headline_permission(primary, reverse)
    result = {
        "schema": "exp212-fresh-pair-result-v1",
        "counts": {"speakers": len(speakers), "speaker_ids": speakers, "dropped_speakers": manifest["dropped_speakers"],
                   "clones": len(jobs), "comparisons_per_speaker_per_direction": 32,
                   "bootstrap_replicates": analyze205.BOOTSTRAPS, "bootstrap_seed": analyze205.BOOTSTRAP_SEED},
        "directions": {direction: {readout: {
            "point": point[(direction, readout)],
            "stability_interval_95": list(intervals[(direction, readout)]),
            "speaker_means": vectors[(direction, readout)].tolist(),
            "system_points_no_intervals": systems[(direction, readout)],
            "agreement_with_original_pair_descriptive": {
                **agreement[(direction, readout)],
                "mean_difference_fresh_minus_rescored_original": float(differences[(direction, readout)].mean()),
                "mean_difference_interval_95": list(difference_intervals[(direction, readout)]),
            }} for readout in READOUTS} for direction in DIRECTION.values()},
        "primary": primary.as_dict(), "reverse": reverse.as_dict(), "headline_permission": permission,
        "replication": "REPLICATED" if permission == "BIDIRECTIONAL_HEADLINE_PERMITTED" else "NOT_REPLICATED",
        "input_hashes": {"manifest": sha256(manifest_path), "jobs": sha256(jobs_path),
                         **{f"features_{n}": sha256(args.run / "features" / f"{n}.npz") for n in READOUTS.values()},
                         "exp205_analyze": sha256(EXP205 / "analyze.py"), "exp205_verdict": sha256(EXP205 / "verdict.py"),
                         "sealed_scientific_result": sha256(RUN205 / "sealed/scientific-result.json")},
    }
    for (direction, readout), v in vectors.items():
        lo, hi = intervals[(direction, readout)]
        print(f"{direction:22s} {readout:6s} {v.mean():.3f} [{lo:.3f},{hi:.3f}]  r_vs_rescored={agreement[(direction, readout)]['pearson_r_vs_rescored']:.2f}", flush=True)
    (HERE / "fresh_pair_result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(primary.verdict, reverse.verdict, permission, result["replication"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
