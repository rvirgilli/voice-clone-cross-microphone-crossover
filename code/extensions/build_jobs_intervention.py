#!/usr/bin/env python3
"""Build the EXP-211 job lists: one per manipulation, primary direction only (mic1 prompts)."""

from __future__ import annotations

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
RUN = RUNS / "EXP-211-f2-prompt-intervention"
CONDITIONS = ("flat", "stretch", "ltas")
SYSTEMS = ("f5", "xtts", "cosy", "seedvc")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    manifest = json.loads((EXP205 / "selection-manifest.json").read_text(encoding="utf-8"))
    texts = manifest["generation"]["generated_texts"]
    sources = {s["index"]: s for s in manifest["generation"]["seedvc_sources"]}
    seed_base = int(manifest["generation"]["rng_seed_base"])
    for cond in CONDITIONS:
        jobs = []
        for spk in manifest["speakers"]:
            for arm in ("A", "B"):
                prompt = RUN / "prompts" / cond / f"{spk['speaker']}_{arm}_mic1.wav"
                prompt_sha = sha256(prompt)
                for system in SYSTEMS:
                    for text in texts:
                        t = int(text["index"])
                        job = {
                            "speaker": spk["speaker"], "prompt_mic": "mic1", "arm": arm, "system": system,
                            "condition": cond, "text_index": t,
                            "reference": str(prompt), "reference_sha256": prompt_sha,
                            "reference_text": spk["transcripts"][arm]["text"],
                            "reference_text_sha256": spk["transcripts"][arm]["sha256"],
                            "generated_text": text["text"], "generated_text_sha256": text["sha256_utf8"],
                            "seed": seed_base + t,
                            "out": str(RUN / "clones" / cond / f"{system}__mic1__seed{arm}" / f"{spk['speaker']}_t{t}.wav"),
                        }
                        if system == "seedvc":
                            src = sources[t]
                            job.update({"source": src["path"], "source_sha256": src["sha256"],
                                        "source_transcript_sha256": src["transcript_sha256"]})
                        jobs.append(job)
        if len(jobs) != 1728:
            raise RuntimeError(f"{cond}: expected 1,728 jobs, built {len(jobs)}")
        out = RUN / f"jobs_{cond}.json"
        out.write_text(json.dumps({"schema": "exp211-prompt-intervention-jobs-v1", "condition": cond, "jobs": jobs}, indent=1) + "\n", encoding="utf-8")
        print(cond, len(jobs), out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
