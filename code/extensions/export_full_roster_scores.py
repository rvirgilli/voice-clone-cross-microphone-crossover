#!/usr/bin/env python3
"""Export the EXP-213 full-roster comparison-level scores for the public release.

One row per comparison and readout: the clone (speaker, system, text, prompt microphone,
seed arm, ledger clone-hash prefix), the two opposite-microphone candidate captures and
the cosine of the clone to each, so the four tier-cohort points of
``data/full_roster_result.json`` recompute from text alone by speaker-weighted averaging,
and the 108-speaker union points recompute by concatenating those per-speaker means with
the released per-speaker means of the primary result. Paths are release-relative; no audio
or embeddings.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np

# Historical run layout: these scripts document what ran. Point the two roots at copies of
# the private run directories and the experiment tree; nothing here reads the release itself.
RUNS = Path(os.environ.get("ICASSP_RUNS", "/runs"))
EXPERIMENTS = Path(os.environ.get("ICASSP_EXPERIMENTS", "/experiments"))
HF = Path(os.environ.get("HF_HUB_CACHE", "~/.cache/huggingface/hub")).expanduser()
EXP213 = EXPERIMENTS / "EXP-213-f2-full-roster-complement"
RUN213 = RUNS / "EXP-213-f2-full-roster-complement"
VCTK = str(RUNS / "vctk") + "/"
DIRECTION = {"mic1": "primary_mic1_to_mic2", "mic2": "reverse_mic2_to_mic1"}
READOUTS = {"ecapa": ("ecapa", "ecapa"), "wavlm": ("wavlmsv", "wavlmsv_xvector")}
PREFIX = 16
HEADER = ["speaker", "direction", "readout", "system", "text_index", "prompt_mic", "seed_arm", "clone_sha256",
          "own_candidate", "other_candidate", "cos_own_event", "cos_other_event"]


def rel(path: str) -> str:
    if not path.startswith(VCTK):
        raise ValueError(path)
    return "inputs/vctk/" + path[len(VCTK):]


def load(name: str, array: str):
    data = np.load(RUN213 / "features" / f"{name}.npz")
    vec = data[array].astype(np.float64)
    vec /= np.linalg.norm(vec, axis=1, keepdims=True)
    return {p: i for i, p in enumerate(data["paths"].tolist())}, vec


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((EXP213 / "selection-manifest-213.json").read_text(encoding="utf-8"))
    real = {(s["speaker"], k): s["audio"][k]["path"] for s in manifest["speakers"] for k in s["audio"]}
    jobs = json.loads((RUN213 / "jobs.json").read_text(encoding="utf-8"))["jobs"]
    feats = {readout: load(*names) for readout, names in READOUTS.items()}
    rows = []
    for job in jobs:
        candidate_mic = "mic2" if job["prompt_mic"] == "mic1" else "mic1"
        other_arm = "B" if job["arm"] == "A" else "A"
        own = real[(job["speaker"], f"{job['arm']}_{candidate_mic}")]
        other = real[(job["speaker"], f"{other_arm}_{candidate_mic}")]
        ledger = json.loads(Path(job["out"] + ".ledger.json").read_text(encoding="utf-8"))
        for readout, (index, vec) in feats.items():
            q = vec[index[job["out"]]]
            rows.append([job["speaker"], DIRECTION[job["prompt_mic"]], readout, job["system"], job["text_index"],
                         job["prompt_mic"], job["arm"], ledger["clone_sha256"][:PREFIX], rel(own), rel(other),
                         f"{float(q @ vec[index[own]]):.17g}", f"{float(q @ vec[index[other]]):.17g}"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(HEADER)
        writer.writerows(rows)
    print(args.out, len(rows), "rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
