#!/usr/bin/env python3
"""Score the EXP-211 clones: attribution under each manipulation, two candidate arms,
plus a synthesis-quality control (clone-to-prompt speaker cosine versus EXP-205)."""

from __future__ import annotations

import argparse
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
RUN207 = RUNS / "EXP-207-f2-readout-roster"
RUN205 = RUNS / "EXP-205-f2-crossmic-crossover"
HERE = Path(__file__).resolve().parent
CONDITIONS = ("flat", "stretch", "ltas")
BOOTSTRAPS = 10_000
SEED = 2052027


def load(feature_dir: Path, name: str):
    data = np.load(feature_dir / f"{name}.npz")
    vec = data["ecapa" if name == "ecapa" else "wavlmsv_xvector"].astype(np.float64)
    vec /= np.linalg.norm(vec, axis=1, keepdims=True)
    return {p: i for i, p in enumerate(data["paths"].tolist())}, vec


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("extract", "analyze"))
    parser.add_argument("--run", type=Path, default=RUNS / "EXP-211-f2-prompt-intervention")
    args = parser.parse_args()
    manifest = json.loads((EXP205 / "selection-manifest.json").read_text(encoding="utf-8"))
    real = {(s["speaker"], k): s["audio"][k]["path"] for s in manifest["speakers"] for k in ("A_mic1", "A_mic2", "B_mic1", "B_mic2")}
    jobs = {c: json.loads((args.run / f"jobs_{c}.json").read_text(encoding="utf-8"))["jobs"] for c in CONDITIONS}
    matched = {(c, s["speaker"], arm): str(args.run / "prompts" / c / f"{s['speaker']}_{arm}_mic2.wav")
               for c in CONDITIONS for s in manifest["speakers"] for arm in ("A", "B")}
    if args.stage == "extract":
        sys.path.insert(0, str(EXP207))
        import extract_readouts as er
        import torch
        paths = [j["out"] for c in CONDITIONS for j in jobs[c]] + sorted(set(real.values())) + sorted(set(matched.values()))
        paths += [j["reference"] for c in CONDITIONS for j in jobs[c][::16]]
        paths = list(dict.fromkeys(paths))
        device = "cuda" if torch.cuda.is_available() else "cpu"
        for name in ("ecapa", "wavlmsv"):
            er.run_readout(name, paths, args.run / "features", device)
        return 0

    speakers = sorted({s["speaker"] for s in manifest["speakers"]})
    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, len(speakers), size=(BOOTSTRAPS, len(speakers)))
    result = {"schema": "exp211-prompt-intervention-v1", "status": "POST_HOC_DESCRIPTIVE_NO_VERDICT_CHANGE",
              "bootstrap": {"unit": "speaker", "replicates": BOOTSTRAPS, "seed": SEED}, "readouts": {}}
    for name in ("ecapa", "wavlmsv"):
        index, vec = load(args.run / "features", name)
        base_index, base_vec = load(RUN207 / "features", name)
        result["readouts"][name] = {}
        # Baseline clone-to-own-prompt cosine from EXP-205 (mic1 prompts).
        import csv
        base_cos = []
        with (RUN205 / "scores.tsv").open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                if row["prompt_mic"] == "mic1":
                    c = base_vec[base_index[str(RUN205 / row["clone_path"])]]
                    p = base_vec[base_index[real[(row["speaker"], f"{row['seed_arm']}_mic1")]]]
                    base_cos.append(float(c @ p))
        result["readouts"][name]["baseline_clone_to_prompt_cosine"] = float(np.mean(base_cos))
        for cond in CONDITIONS:
            per = {"unmodified_candidates": defaultdict(list), "matched_candidates": defaultdict(list)}
            quality = []
            for job in jobs[cond]:
                clone = vec[index[job["out"]]]
                other_arm = "B" if job["arm"] == "A" else "A"
                own_u = vec[index[real[(job["speaker"], f"{job['arm']}_mic2")]]]
                oth_u = vec[index[real[(job["speaker"], f"{other_arm}_mic2")]]]
                own_m = vec[index[matched[(cond, job["speaker"], job["arm"])]]]
                oth_m = vec[index[matched[(cond, job["speaker"], other_arm)]]]
                for arm_name, own, oth in (("unmodified_candidates", own_u, oth_u), ("matched_candidates", own_m, oth_m)):
                    d = float(clone @ own - clone @ oth)
                    per[arm_name][job["speaker"]].append(1.0 if d > 0 else 0.0 if d < 0 else 0.5)
                if job["reference"] in index:
                    quality.append(float(clone @ vec[index[job["reference"]]]))
            cell = {"clone_to_prompt_cosine": float(np.mean(quality)) if quality else None}
            for arm_name, table in per.items():
                means = np.array([np.mean(table[s]) for s in speakers])
                boot = means[draws].mean(axis=1)
                cell[arm_name] = {"point": float(means.mean()),
                                  "stability_interval_95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]}
                print(f"{name:8s} {cond:8s} {arm_name:22s} {means.mean():.3f} [{np.percentile(boot, 2.5):.3f},{np.percentile(boot, 97.5):.3f}]", flush=True)
            result["readouts"][name][cond] = cell
    readings = {}
    for cond in CONDITIONS:
        lo = result["readouts"]["ecapa"][cond]["unmodified_candidates"]["stability_interval_95"][0]
        readings[cond] = "COLLAPSE" if lo < 0.60 else "SURVIVE" if lo >= 0.80 else "PARTIAL"
    result["readings"] = readings
    (HERE / "intervention_result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(readings))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
