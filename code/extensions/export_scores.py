#!/usr/bin/env python3
"""Export EXP-209/210/211 comparison-level scores for the public release.

Each table has one row per comparison with the cosine of the query clone to every
candidate, so the printed points recompute from text alone (the bootstrap intervals
regenerate from the same rows). Paths are release-relative; no audio or embeddings.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

import os

# Historical run layout: these scripts document what ran. Point the two roots at copies of
# the private run directories and the experiment tree; nothing here reads the release itself.
RUNS = Path(os.environ.get("ICASSP_RUNS", "/runs"))
EXPERIMENTS = Path(os.environ.get("ICASSP_EXPERIMENTS", "/experiments"))
HF = Path(os.environ.get("HF_HUB_CACHE", "~/.cache/huggingface/hub")).expanduser()
EXP205 = EXPERIMENTS / "EXP-205-f2-crossmic-crossover"
RUN205 = RUNS / "EXP-205-f2-crossmic-crossover"
RUNS = {
    "209": RUNS / "EXP-209-f2-ncandidate-scaling",
    "210": RUNS / "EXP-210-f2-second-generation",
    "211": RUNS / "EXP-211-f2-prompt-intervention",
}
VCTK = str(RUNS / "vctk") + "/"


def rel(path: str) -> str:
    if path.startswith(str(RUN205) + "/"):
        return "runtime/" + path[len(str(RUN205)) + 1 :]
    if path.startswith(VCTK):
        return "inputs/vctk/" + path[len(VCTK) :]
    for key, run in RUNS.items():
        if path.startswith(str(run) + "/"):
            return f"exp{key}/" + path[len(str(run)) + 1 :]
    raise ValueError(path)


def load(run: Path, name: str):
    data = np.load(run / "features" / f"{name}.npz")
    vec = data["ecapa" if name == "ecapa" else "wavlmsv_xvector"].astype(np.float64)
    vec /= np.linalg.norm(vec, axis=1, keepdims=True)
    return {p: i for i, p in enumerate(data["paths"].tolist())}, vec


def write(path: Path, header: list[str], rows: list[list]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(header)
        writer.writerows(rows)
    print(path, len(rows), "rows")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((EXP205 / "selection-manifest.json").read_text(encoding="utf-8"))
    real = {(s["speaker"], k): s["audio"][k]["path"] for s in manifest["speakers"] for k in ("A_mic1", "A_mic2", "B_mic1", "B_mic2")}
    with (RUN205 / "scores.tsv").open(newline="", encoding="utf-8") as handle:
        rows205 = list(csv.DictReader(handle, delimiter="\t"))

    # EXP-209: query clone against own event, other event and the ranked distractors.
    distractors = json.loads((RUNS["209"] / "distractors.json").read_text(encoding="utf-8"))["roster"]
    feats = {n: load(RUNS["209"], n) for n in ("ecapa", "wavlmsv")}
    rows = []
    for row in rows205:
        cand_mic = "mic2" if row["prompt_mic"] == "mic1" else "mic1"
        other_arm = "B" if row["seed_arm"] == "A" else "A"
        pool = distractors[row["speaker"]][:14]
        cands = [real[(row["speaker"], f"{row['seed_arm']}_{cand_mic}")], real[(row["speaker"], f"{other_arm}_{cand_mic}")]] + [d[cand_mic]["path"] for d in pool]
        clone = str(RUN205 / row["clone_path"])
        for name, (index, vec) in feats.items():
            q = vec[index[clone]]
            scores = [float(q @ vec[index[c]]) for c in cands]
            rows.append([row["speaker"], row["system"], row["text_index"], row["prompt_mic"], row["seed_arm"], name,
                         rel(clone), len(pool)] + [f"{s:.17g}" for s in scores] + [rel(c) for c in cands[2:]])
    write(args.out / "ncandidate_scores.tsv",
          ["speaker", "system", "text_index", "prompt_mic", "seed_arm", "readout", "clone_path", "n_distractors",
           "cos_own_event", "cos_other_event"] + [f"cos_distractor_{i+1}" for i in range(14)] + [f"distractor_{i+1}" for i in range(14)], rows)

    # EXP-210: second-generation clone against the original event's mic2 captures.
    jobs = json.loads((RUNS["210"] / "jobs.json").read_text(encoding="utf-8"))["jobs"]
    feats = {n: load(RUNS["210"], n) for n in ("ecapa", "wavlmsv")}
    rows = []
    for job in jobs:
        own = real[(job["speaker"], f"{job['arm']}_mic2")]
        oth = real[(job["speaker"], f"{'B' if job['arm'] == 'A' else 'A'}_mic2")]
        for name, (index, vec) in feats.items():
            q = vec[index[job["out"]]]
            rows.append([job["speaker"], job["gen1_system"], job["gen1_text_index"], job["system"], job["text_index"], job["arm"], name,
                         rel(job["out"]), f"{float(q @ vec[index[own]]):.17g}", f"{float(q @ vec[index[oth]]):.17g}"])
    write(args.out / "second_generation_scores.tsv",
          ["speaker", "gen1_system", "gen1_text_index", "gen2_system", "gen2_text_index", "seed_arm", "readout", "clone_path", "cos_own_event", "cos_other_event"], rows)

    # EXP-211: intervention clones against unmodified and matched candidates, plus prompt cosine.
    feats = {n: load(RUNS["211"], n) for n in ("ecapa", "wavlmsv")}
    rows = []
    for cond in ("flat", "stretch", "ltas"):
        for job in json.loads((RUNS["211"] / f"jobs_{cond}.json").read_text(encoding="utf-8"))["jobs"]:
            other = "B" if job["arm"] == "A" else "A"
            own_u, oth_u = real[(job["speaker"], f"{job['arm']}_mic2")], real[(job["speaker"], f"{other}_mic2")]
            own_m = str(RUNS["211"] / "prompts" / cond / f"{job['speaker']}_{job['arm']}_mic2.wav")
            oth_m = str(RUNS["211"] / "prompts" / cond / f"{job['speaker']}_{other}_mic2.wav")
            for name, (index, vec) in feats.items():
                q = vec[index[job["out"]]]
                prompt = f"{float(q @ vec[index[job['reference']]]):.17g}" if job["reference"] in index else ""
                rows.append([cond, job["speaker"], job["system"], job["text_index"], job["arm"], name, rel(job["out"]),
                             f"{float(q @ vec[index[own_u]]):.17g}", f"{float(q @ vec[index[oth_u]]):.17g}",
                             f"{float(q @ vec[index[own_m]]):.17g}", f"{float(q @ vec[index[oth_m]]):.17g}", prompt])
    write(args.out / "intervention_scores.tsv",
          ["condition", "speaker", "system", "text_index", "seed_arm", "readout", "clone_path",
           "cos_own_event", "cos_other_event", "cos_own_event_matched", "cos_other_event_matched", "cos_prompt"], rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
