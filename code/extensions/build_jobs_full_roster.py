#!/usr/bin/env python3
"""Build the EXP-213 job list: the EXP-205 grid over the tier-speaker pairs.

Every retained speaker x 2 prompt microphones x 2 arms x 4 texts x 4 systems, in the job
schema consumed by EXP-210's gen_jobs.py (frozen EXP-205 generators, pins and ledgers),
with the EXP-205 clone layout {system}__{prompt_mic}__seed{arm}/{speaker}_t{i}.wav.
"""

from __future__ import annotations

import json
from pathlib import Path

import os

# Historical run layout: these scripts document what ran. Point the two roots at copies of
# the private run directories and the experiment tree; nothing here reads the release itself.
RUNS = Path(os.environ.get("ICASSP_RUNS", "/runs"))
EXPERIMENTS = Path(os.environ.get("ICASSP_EXPERIMENTS", "/experiments"))
HF = Path(os.environ.get("HF_HUB_CACHE", "~/.cache/huggingface/hub")).expanduser()
EXP213 = EXPERIMENTS / "EXP-213-f2-full-roster-complement"
RUN = RUNS / "EXP-213-f2-full-roster-complement"
SYSTEMS = ("f5", "xtts", "cosy", "seedvc")


def main() -> int:
    manifest = json.loads((EXP213 / "selection-manifest-213.json").read_text(encoding="utf-8"))
    if manifest["schema"] != "exp213-full-roster-complement-manifest-v1":
        raise RuntimeError("manifest schema mismatch")
    texts = manifest["generation"]["generated_texts"]
    sources = {s["index"]: s for s in manifest["generation"]["seedvc_sources"]}
    seed_base = int(manifest["generation"]["rng_seed_base"])
    jobs = []
    for speaker in manifest["speakers"]:
        for prompt_mic in ("mic1", "mic2"):
            for arm in ("A", "B"):
                reference = speaker["audio"][f"{arm}_{prompt_mic}"]
                transcript = speaker["transcripts"][arm]
                for text in texts:
                    t = int(text["index"])
                    for system in SYSTEMS:
                        job = {
                            "speaker": speaker["speaker"], "prompt_mic": prompt_mic, "arm": arm,
                            "system": system, "text_index": t,
                            "reference": reference["path"], "reference_sha256": reference["sha256"],
                            "reference_text": transcript["text"], "reference_text_sha256": transcript["sha256"],
                            "generated_text": text["text"], "generated_text_sha256": text["sha256_utf8"],
                            "seed": seed_base + t,
                            "out": str(RUN / "clones" / f"{system}__{prompt_mic}__seed{arm}" / f"{speaker['speaker']}_t{t}.wav"),
                        }
                        if system == "seedvc":
                            src = sources[t]
                            job.update({"source": src["path"], "source_sha256": src["sha256"],
                                        "source_transcript_sha256": src["transcript_sha256"]})
                        jobs.append(job)
    if len(jobs) != manifest["counts"]["expected_clones"]:
        raise RuntimeError(f"expected {manifest['counts']['expected_clones']} jobs, built {len(jobs)}")
    RUN.mkdir(parents=True, exist_ok=True)
    out = RUN / "jobs.json"
    out.write_text(json.dumps({"schema": "exp213-full-roster-complement-jobs-v1", "manifest": str(EXP213 / "selection-manifest-213.json"),
                               "jobs": jobs}, indent=1) + "\n", encoding="utf-8")
    print(f"jobs={len(jobs)} speakers={len(manifest['speakers'])} {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
