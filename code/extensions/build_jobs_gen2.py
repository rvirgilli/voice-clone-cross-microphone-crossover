#!/usr/bin/env python3
"""Build the second-generation job list for EXP-210 (see PREREG.md).

Every primary-direction EXP-205 clone (mic1 prompt; 1,728 files) becomes the prompt of
a second synthesis with a *different* system (cyclic rotation f5 -> xtts -> cosy ->
seedvc -> f5) and a *different* fixed text (index + 1 mod 4). The conditioning event
of the second-generation clone is still the original event, and its candidates are
the original mic2 captures of A and B.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import os

# Historical run layout: these scripts document what ran. Point the two roots at copies of
# the private run directories and the experiment tree; nothing here reads the release itself.
RUNS = Path(os.environ.get("ICASSP_RUNS", "/runs"))
EXPERIMENTS = Path(os.environ.get("ICASSP_EXPERIMENTS", "/experiments"))
HF = Path(os.environ.get("HF_HUB_CACHE", "~/.cache/huggingface/hub")).expanduser()
EXP205 = EXPERIMENTS / "EXP-205-f2-crossmic-crossover"
RUN205 = RUNS / "EXP-205-f2-crossmic-crossover"
RUN210 = RUNS / "EXP-210-f2-second-generation"
ROTATION = {"f5": "xtts", "xtts": "cosy", "cosy": "seedvc", "seedvc": "f5"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    manifest = json.loads((EXP205 / "selection-manifest.json").read_text(encoding="utf-8"))
    texts = {t["index"]: t for t in manifest["generation"]["generated_texts"]}
    sources = {s["index"]: s for s in manifest["generation"]["seedvc_sources"]}
    seed_base = int(manifest["generation"]["rng_seed_base"])
    jobs = []
    with (RUN205 / "scores.tsv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row["prompt_mic"] != "mic1":
                continue
            t1 = int(row["text_index"])
            t2 = (t1 + 1) % 4
            g1, g2 = row["system"], ROTATION[row["system"]]
            clone = RUN205 / row["clone_path"]
            job = {
                "speaker": row["speaker"], "prompt_mic": "mic1", "arm": row["seed_arm"],
                "gen1_system": g1, "gen1_text_index": t1, "system": g2, "text_index": t2,
                "reference": str(clone), "reference_sha256": row["clone_sha256"],
                "reference_text": texts[t1]["text"], "reference_text_sha256": texts[t1]["sha256_utf8"],
                "generated_text": texts[t2]["text"], "generated_text_sha256": texts[t2]["sha256_utf8"],
                "seed": seed_base + t2,
                "out": str(RUN210 / "clones" / f"{g2}__from_{g1}__seed{row['seed_arm']}" / f"{row['speaker']}_t{t2}.wav"),
            }
            if g2 == "seedvc":
                src = sources[t2]
                job.update({"source": src["path"], "source_sha256": src["sha256"],
                            "source_transcript_sha256": src["transcript_sha256"]})
            if sha256(clone) != row["clone_sha256"]:
                raise RuntimeError(f"gen-1 clone hash mismatch: {clone}")
            jobs.append(job)
    if len(jobs) != 1728:
        raise RuntimeError(f"expected 1,728 jobs, built {len(jobs)}")
    RUN210.mkdir(parents=True, exist_ok=True)
    out = RUN210 / "jobs.json"
    out.write_text(json.dumps({"schema": "exp210-second-generation-jobs-v1", "rotation": ROTATION, "jobs": jobs}, indent=1) + "\n", encoding="utf-8")
    counts = {}
    for job in jobs:
        counts[job["system"]] = counts.get(job["system"], 0) + 1
    print(json.dumps(counts), out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
