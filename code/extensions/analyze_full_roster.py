#!/usr/bin/env python3
"""Score and analyse EXP-213: the EXP-205 crossover on the 54 tier speakers.

extract: ECAPA and WavLM-SV for every clone and real capture with the EXP-207 extractor
         (the readouts used by EXP-210/211/212), written once per readout.
analyze: the EXP-205 estimand per direction (own-event cosine above other-event cosine,
         ties 1/2; averaged within speaker, then equally over speakers), the EXP-205
         bootstrap (imported: 100,000 whole-speaker draws, seed 2052027) and the frozen
         EXP-205 verdict module, for two cohorts: (a) the tier speakers alone and (b) the
         108-speaker union of the sealed EXP-205 per-speaker means with the new ones.
         Union per-system points are speaker-count-weighted means of the two cohorts'
         clone means (every speaker contributes the same number of clones per cell).
"""

from __future__ import annotations

import argparse
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
EXP205 = EXPERIMENTS / "EXP-205-f2-crossmic-crossover"
EXP207 = EXPERIMENTS / "EXP-207-f2-readout-roster"
EXP213 = EXPERIMENTS / "EXP-213-f2-full-roster-complement"
SEALED = RUNS / "EXP-205-f2-crossmic-crossover/sealed/scientific-result.json"
SEALED_SHA256 = "fc51fc71625a44c18a2b566d81ef85a15ff8b318b07eda4f59ddf37753884e0b"
DIRECTION = {"mic1": "primary_mic1_to_mic2", "mic2": "reverse_mic2_to_mic1"}
READOUTS = {"ecapa": "ecapa", "wavlm": "wavlmsv"}
ARRAY = {"ecapa": "ecapa", "wavlm": "wavlmsv_xvector"}
SYSTEMS = ("f5", "xtts", "cosy", "seedvc")


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


def cohort(analyze205, verdict, speakers: list[str], vectors: dict, systems: dict) -> dict:
    """Points, intervals, per-system points and the frozen EXP-205 verdict for one speaker set."""
    intervals = analyze205.bootstrap_intervals(vectors)
    point = {key: float(v.mean()) for key, v in vectors.items()}
    primary = verdict.primary_verdict(ecapa_point=point[("primary_mic1_to_mic2", "ecapa")],
                                      ecapa_lcb=intervals[("primary_mic1_to_mic2", "ecapa")][0],
                                      wavlm_lcb=intervals[("primary_mic1_to_mic2", "wavlm")][0])
    reverse = verdict.reverse_verdict(ecapa_lcb=intervals[("reverse_mic2_to_mic1", "ecapa")][0],
                                      wavlm_lcb=intervals[("reverse_mic2_to_mic1", "wavlm")][0])
    permission = verdict.headline_permission(primary, reverse)
    return {
        "counts": {"speakers": len(speakers), "speaker_ids": speakers, "comparisons_per_speaker_per_direction": 32,
                   "bootstrap_replicates": analyze205.BOOTSTRAPS, "bootstrap_seed": analyze205.BOOTSTRAP_SEED},
        "directions": {direction: {readout: {
            "point": point[(direction, readout)],
            "stability_interval_95": list(intervals[(direction, readout)]),
            "speaker_means": vectors[(direction, readout)].tolist(),
            "system_points_no_intervals": systems[(direction, readout)],
        } for readout in READOUTS} for direction in DIRECTION.values()},
        "primary": primary.as_dict(), "reverse": reverse.as_dict(), "headline_permission": permission,
        "rule_met": permission == "BIDIRECTIONAL_HEADLINE_PERMITTED",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=RUNS / "EXP-213-f2-full-roster-complement")
    parser.add_argument("--out", type=Path, default=EXP213 / "full_roster_result.json")
    parser.add_argument("stage", choices=("extract", "analyze"))
    args = parser.parse_args()
    jobs_path = args.run / "jobs.json"
    jobs = json.loads(jobs_path.read_text(encoding="utf-8"))["jobs"]
    manifest_path = EXP213 / "selection-manifest-213.json"
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

    if sha256(SEALED) != SEALED_SHA256:
        raise RuntimeError("sealed EXP-205 result hash mismatch")
    analyze205 = load_module("exp205_analyze", EXP205 / "analyze.py")
    verdict = load_module("exp205_verdict", EXP205 / "verdict.py")
    speakers = sorted({s["speaker"] for s in manifest["speakers"]})
    sealed = json.loads(SEALED.read_text(encoding="utf-8"))
    sealed_ids = sealed["counts"]["speaker_ids"]
    if set(sealed_ids) & set(speakers):
        raise RuntimeError("cohorts overlap")

    vectors, systems = {}, {}
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
        for direction in DIRECTION.values():
            if any(len(per[direction][s]) != 32 for s in speakers):
                raise RuntimeError(f"cell count invalid: {readout} {direction}")
            vectors[(direction, readout)] = np.array([np.mean(per[direction][s]) for s in speakers], dtype=np.float64)
            systems[(direction, readout)] = {sy: float(np.mean(by_system[(direction, sy)])) for sy in SYSTEMS}

    n_new, n_sealed = len(speakers), len(sealed_ids)
    union_vectors = {key: np.concatenate([np.array(sealed["directions"][key[0]][key[1]]["speaker_means"], dtype=np.float64), v])
                     for key, v in vectors.items()}
    union_systems = {key: {sy: (n_sealed * sealed["system_points_no_intervals"][key[0]][key[1]][sy] + n_new * s[sy]) / (n_sealed + n_new)
                           for sy in SYSTEMS} for key, s in systems.items()}
    result = {
        "schema": "exp213-full-roster-result-v1",
        "clones": len(jobs), "dropped_speakers": manifest["dropped_speakers"],
        "tier_cohort": cohort(analyze205, verdict, speakers, vectors, systems),
        "full_roster_union": cohort(analyze205, verdict, sealed_ids + speakers, union_vectors, union_systems),
        "exp205_sealed": {"counts": sealed["counts"], "headline_permission": sealed["headline_permission"],
                          "directions": {d: {r: {k: sealed["directions"][d][r][k] for k in ("point", "stability_interval_95")}
                                             for r in READOUTS} for d in DIRECTION.values()}},
        "input_hashes": {"manifest": sha256(manifest_path), "jobs": sha256(jobs_path),
                         **{f"features_{n}": sha256(args.run / "features" / f"{n}.npz") for n in READOUTS.values()},
                         "exp205_analyze": sha256(EXP205 / "analyze.py"), "exp205_verdict": sha256(EXP205 / "verdict.py"),
                         "sealed_scientific_result": SEALED_SHA256},
    }
    for label in ("tier_cohort", "full_roster_union"):
        node = result[label]
        for direction in DIRECTION.values():
            for readout in READOUTS:
                cell = node["directions"][direction][readout]
                print(f"{label:17s} {direction:22s} {readout:6s} {cell['point']:.3f} [{cell['stability_interval_95'][0]:.3f},{cell['stability_interval_95'][1]:.3f}]", flush=True)
        print(label, node["primary"]["verdict"], node["reverse"]["verdict"], node["headline_permission"], "RULE_MET" if node["rule_met"] else "RULE_NOT_MET")
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
