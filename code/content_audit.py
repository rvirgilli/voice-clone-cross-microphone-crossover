#!/usr/bin/env python3
"""CPU ASR audit of EXP-205 clone content (see PREREG.md). Resumable per file."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import soundfile as sf
from faster_whisper import WhisperModel


# Inputs are outside this release: the EXP-205 run directory (clones, scores.tsv) and the
# VCTK captures named in data/selection_manifest.json. Point the environment at them.
ROOT = Path(__file__).resolve().parent.parent
RUN205 = Path(os.environ.get("EXP205_RUN", ROOT / "inputs" / "exp205-run"))
MANIFEST = ROOT / "data" / "selection_manifest.json"
MODEL = Path(os.environ.get("FASTER_WHISPER_TURBO", ROOT / "inputs" / "faster-whisper-large-v3-turbo"))
OVERLAP_WORDS = 4


def normalize(text: str) -> list[str]:
    return re.sub(r"[^a-z0-9' ]+", " ", text.lower()).split()


def wer(reference: list[str], hypothesis: list[str]) -> float:
    """Levenshtein word error rate."""
    previous = list(range(len(hypothesis) + 1))
    for i, ref_word in enumerate(reference, start=1):
        current = [i]
        for j, hyp_word in enumerate(hypothesis, start=1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ref_word != hyp_word)))
        previous = current
    return previous[-1] / max(1, len(reference))


def longest_shared_run(clone: list[str], prompt: list[str], requested: list[str]) -> int:
    """Longest run of consecutive clone words also consecutive in the prompt and absent from the requested text."""
    requested_text = " " + " ".join(requested) + " "
    best = 0
    for i in range(len(clone)):
        for j in range(len(prompt)):
            k = 0
            while i + k < len(clone) and j + k < len(prompt) and clone[i + k] == prompt[j + k]:
                k += 1
            if k > best and (" " + " ".join(clone[i : i + k]) + " ") not in requested_text:
                best = k
    return best


def transcribe(model: WhisperModel, path: str, cache: dict[str, str], cache_path: Path) -> str:
    if path not in cache:
        segments, _ = model.transcribe(path, beam_size=1, language="en")
        cache[path] = " ".join(segment.text for segment in segments).strip()
        with cache_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"path": path, "text": cache[path]}) + "\n")
    return cache[path]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / "transcripts.jsonl"
    cache = {}
    if cache_path.exists():
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            cache[record["path"]] = record["text"]

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    requested = {t["index"]: normalize(t["text"]) for t in manifest["generation"]["generated_texts"]}
    prompts = {(s["speaker"], key): str(ROOT / s["audio"][key]["path"]) for s in manifest["speakers"] for key in ("A_mic1", "A_mic2", "B_mic1", "B_mic2")}
    with (ROOT / "data" / "scores.tsv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))

    model = WhisperModel(str(MODEL), device="cpu", compute_type="int8")
    records = []
    durations = defaultdict(list)
    for n, row in enumerate(rows, start=1):
        clone_path = str(RUN205 / row["clone_path"])
        prompt_path = prompts[(row["speaker"], f"{row['seed_arm']}_{row['prompt_mic']}")]
        clone_words = normalize(transcribe(model, clone_path, cache, cache_path))
        prompt_words = normalize(transcribe(model, prompt_path, cache, cache_path))
        req = requested[int(row["text_index"])]
        info = sf.info(clone_path)
        duration = info.frames / info.samplerate
        durations[(row["system"], int(row["text_index"]))].append(duration)
        records.append({
            "speaker": row["speaker"], "system": row["system"], "text_index": int(row["text_index"]),
            "prompt_mic": row["prompt_mic"], "seed_arm": row["seed_arm"], "clone_path": row["clone_path"],
            "duration_s": duration, "wer_requested": wer(req, clone_words),
            "prompt_overlap": longest_shared_run(clone_words, prompt_words, req),
        })
        if n % 200 == 0:
            print(f"{n}/{len(rows)}", flush=True)

    medians = {key: sorted(v)[len(v) // 2] for key, v in durations.items()}
    for record in records:
        record["duration_ratio"] = record["duration_s"] / medians[(record["system"], record["text_index"])]
        record["contaminated"] = record["prompt_overlap"] >= OVERLAP_WORDS

    def summarize(subset: list[dict]) -> dict:
        wers = sorted(r["wer_requested"] for r in subset)
        return {"n": len(subset), "contaminated": sum(r["contaminated"] for r in subset),
                "contamination_rate": sum(r["contaminated"] for r in subset) / len(subset),
                "median_wer_requested": wers[len(wers) // 2],
                "max_prompt_overlap": max(r["prompt_overlap"] for r in subset)}

    summary = {"overall": summarize(records), "by_system": {s: summarize([r for r in records if r["system"] == s]) for s in ("f5", "xtts", "cosy", "seedvc")}}
    clean = summary["overall"]["contamination_rate"] <= 0.01 and all(
        v["contamination_rate"] <= 0.01 and v["median_wer_requested"] <= 0.20 for v in summary["by_system"].values())
    contaminated = any(v["contamination_rate"] > 0.05 for v in summary["by_system"].values())
    summary["reading"] = "CLEAN" if clean else "CONTAMINATED" if contaminated else "INTERMEDIATE"
    summary["flagged"] = [r for r in records if r["contaminated"]]
    (out_dir / "content_audit.json").write_text(json.dumps({"schema": "exp208-output-content-audit-v1",
        "status": "POST_HOC_DESCRIPTIVE_NO_VERDICT_CHANGE", "overlap_words": OVERLAP_WORDS,
        "summary": summary, "records": records}, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "flagged"}, indent=2))
    print("flagged:", len(summary["flagged"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
