#!/usr/bin/env python3
"""Score second-generation clones against the original event's mic2 captures.

Statistic: the EXP-205 estimand (own-event cosine above other-event cosine, ties 1/2),
averaged within speaker, then equally over speakers; whole-speaker percentile
bootstrap. Both readouts are extracted with the EXP-207 extractor.
"""

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
HERE = Path(__file__).resolve().parent
BOOTSTRAPS = 10_000
SEED = 2052027


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=RUNS / "EXP-210-f2-second-generation")
    parser.add_argument("stage", choices=("extract", "analyze"))
    args = parser.parse_args()
    jobs = json.loads((args.run / "jobs.json").read_text(encoding="utf-8"))["jobs"]
    manifest = json.loads((EXP205 / "selection-manifest.json").read_text(encoding="utf-8"))
    real = {(s["speaker"], k): s["audio"][k]["path"] for s in manifest["speakers"] for k in ("A_mic2", "B_mic2")}
    if args.stage == "extract":
        sys.path.insert(0, str(EXP207))
        import extract_readouts as er
        import torch
        paths = [j["out"] for j in jobs] + sorted(set(real.values()))
        device = "cuda" if torch.cuda.is_available() else "cpu"
        for name in ("ecapa", "wavlmsv"):
            er.run_readout(name, paths, args.run / "features", device)
        return 0

    speakers = sorted({j["speaker"] for j in jobs})
    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, len(speakers), size=(BOOTSTRAPS, len(speakers)))
    result = {"schema": "exp210-second-generation-v1", "status": "POST_HOC_DESCRIPTIVE_NO_VERDICT_CHANGE",
              "bootstrap": {"unit": "speaker", "replicates": BOOTSTRAPS, "seed": SEED}, "readouts": {}}
    for name in ("ecapa", "wavlmsv"):
        data = np.load(args.run / "features" / f"{name}.npz")
        index = {p: i for i, p in enumerate(data["paths"].tolist())}
        vec = data["ecapa" if name == "ecapa" else "wavlmsv_xvector"].astype(np.float64)
        vec /= np.linalg.norm(vec, axis=1, keepdims=True)
        per = defaultdict(lambda: defaultdict(list))
        for job in jobs:
            clone = vec[index[job["out"]]]
            own = vec[index[real[(job["speaker"], f"{job['arm']}_mic2")]]]
            other = vec[index[real[(job["speaker"], f"{'B' if job['arm'] == 'A' else 'A'}_mic2")]]]
            d = float(clone @ own - clone @ other)
            value = 1.0 if d > 0 else 0.0 if d < 0 else 0.5
            per["pooled"][job["speaker"]].append(value)
            per[f"gen2={job['system']}"][job["speaker"]].append(value)
            per[f"gen1={job['gen1_system']}"][job["speaker"]].append(value)
        result["readouts"][name] = {}
        for cell, table in per.items():
            means = np.array([np.mean(table[s]) for s in speakers])
            boot = means[draws].mean(axis=1)
            result["readouts"][name][cell] = {"point": float(means.mean()),
                                              "stability_interval_95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                                              "comparisons_per_speaker": int(np.mean([len(table[s]) for s in speakers]))}
            print(f"{name:8s} {cell:12s} {means.mean():.3f} [{np.percentile(boot, 2.5):.3f},{np.percentile(boot, 97.5):.3f}]", flush=True)
    e = result["readouts"]["ecapa"]["pooled"]
    result["reading"] = ("SURVIVES_LAUNDERING" if e["stability_interval_95"][0] > 0.70
                         else "WEAK_AFTER_LAUNDERING" if e["stability_interval_95"][0] > 0.50 else "LOST_AFTER_LAUNDERING")
    (HERE / "second_generation_result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(result["reading"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
