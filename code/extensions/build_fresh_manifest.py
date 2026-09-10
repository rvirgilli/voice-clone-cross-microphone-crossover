#!/usr/bin/env python3
"""Build the outcome-blind fresh-pair selection manifest for EXP-212.

Same 54 speakers and the same metadata-only rule as EXP-205 (feasibility.py): 4-10 s
paired mic1/mic2 utterances, duration gap <= .25 s and relative seconds-per-UTF-8-byte
gap <= .05 on each microphone, ranked by the EXP-205 tuple. Two exclusions per speaker:
the EXP-203 seed utterance (as the original selection) and both utterances of the
original EXP-205 pair. The builder proves the rule is identical by re-selecting with
only the seed excluded and requiring the original pair back for every speaker.
Never opens clone audio, embeddings, scores or outcomes.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import soundfile as sf

import os

# Historical run layout: these scripts document what ran. Point the two roots at copies of
# the private run directories and the experiment tree; nothing here reads the release itself.
RUNS = Path(os.environ.get("ICASSP_RUNS", "/runs"))
EXPERIMENTS = Path(os.environ.get("ICASSP_EXPERIMENTS", "/experiments"))
HF = Path(os.environ.get("HF_HUB_CACHE", "~/.cache/huggingface/hub")).expanduser()
HERE = Path(__file__).resolve().parent
EXP205 = EXPERIMENTS / "EXP-205-f2-crossmic-crossover"
ORIGINAL = EXP205 / "selection-manifest.json"
FEASIBILITY_SOURCE = EXP205 / "feasibility.py"
VCTK_AUDIO = RUNS / "vctk/wav48_silence_trimmed"
DURATION_TOL = 0.25
RATE_TOL = 0.05


def load_feasibility():
    spec = importlib.util.spec_from_file_location("exp205_feasibility", FEASIBILITY_SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


FZ = load_feasibility()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stem(path: Path) -> str:
    return path.name.split("_mic", 1)[0]


def candidates(speaker: str, excluded_stems: set[str]) -> list[dict]:
    result = []
    for audio in sorted((VCTK_AUDIO / speaker).glob(f"{speaker}_*_mic1.flac")):
        if stem(audio) in excluded_stems:
            continue
        info = sf.info(audio)
        duration = info.frames / info.samplerate
        mic2 = FZ.mic2_counterpart(audio)
        if not (4.0 <= duration <= 10.0 and mic2.is_file()):
            continue
        mic2_info = sf.info(mic2)
        mic2_duration = mic2_info.frames / mic2_info.samplerate
        if not 4.0 <= mic2_duration <= 10.0:
            continue
        result.append({
            "path": audio, "mic2_path": mic2,
            "duration_mic1": duration, "duration_mic2": mic2_duration,
            "rate_mic1": FZ.seconds_per_utf8_byte(audio, duration),
            "rate_mic2": FZ.seconds_per_utf8_byte(mic2, mic2_duration),
        })
    return result


def select_pair(speaker: str, excluded_stems: set[str]) -> dict | None:
    """The EXP-205 draft-pair rule, verbatim from feasibility.py, over the given exclusions."""
    pool = candidates(speaker, excluded_stems)
    pairs = []
    for i, left in enumerate(pool):
        for right in pool[i + 1:]:
            pairs.append({
                "a": str(left["path"]), "b": str(right["path"]),
                "a_mic2": str(left["mic2_path"]), "b_mic2": str(right["mic2_path"]),
                "duration_gap_mic1_s": abs(left["duration_mic1"] - right["duration_mic1"]),
                "duration_gap_mic2_s": abs(left["duration_mic2"] - right["duration_mic2"]),
                "relative_rate_gap_mic1": FZ.relative_gap(left["rate_mic1"], right["rate_mic1"]),
                "relative_rate_gap_mic2": FZ.relative_gap(left["rate_mic2"], right["rate_mic2"]),
            })
    eligible = [
        row for row in pairs
        if max(row["duration_gap_mic1_s"], row["duration_gap_mic2_s"]) <= DURATION_TOL
        and max(row["relative_rate_gap_mic1"], row["relative_rate_gap_mic2"]) <= RATE_TOL
    ]
    return min(
        eligible,
        key=lambda row: (
            max(row["duration_gap_mic1_s"] / DURATION_TOL, row["duration_gap_mic2_s"] / DURATION_TOL,
                row["relative_rate_gap_mic1"] / RATE_TOL, row["relative_rate_gap_mic2"] / RATE_TOL),
            row["duration_gap_mic1_s"] / DURATION_TOL + row["duration_gap_mic2_s"] / DURATION_TOL
            + row["relative_rate_gap_mic1"] / RATE_TOL + row["relative_rate_gap_mic2"] / RATE_TOL,
            max(row["relative_rate_gap_mic1"], row["relative_rate_gap_mic2"]),
            max(row["duration_gap_mic1_s"], row["duration_gap_mic2_s"]),
            row["a"], row["b"],
        ),
        default=None,
    )


def audio_record(path: Path) -> dict:
    info = sf.info(path)
    return {"path": str(path), "sha256": sha256(path), "frames": int(info.frames),
            "samplerate_hz": int(info.samplerate), "duration_s": float(info.frames / info.samplerate)}


def speaker_row(speaker: str, pair: dict, original: dict) -> dict:
    paths = {"A_mic1": Path(pair["a"]), "A_mic2": Path(pair["a_mic2"]),
             "B_mic1": Path(pair["b"]), "B_mic2": Path(pair["b_mic2"])}
    audio = {key: audio_record(path) for key, path in paths.items()}
    texts = {}
    for arm in ("A", "B"):
        text_path = FZ.transcript_path(paths[f"{arm}_mic1"])
        text = text_path.read_text(encoding="utf-8").strip()
        texts[arm] = {"path": str(text_path), "sha256": sha256(text_path),
                      "utf8_bytes": len(text.encode("utf-8")), "text": text}
    return {
        "speaker": speaker,
        "audio": audio,
        "transcripts": texts,
        "match": {
            "duration_gap_mic1_s": pair["duration_gap_mic1_s"],
            "duration_gap_mic2_s": pair["duration_gap_mic2_s"],
            "relative_rate_gap_mic1": pair["relative_rate_gap_mic1"],
            "relative_rate_gap_mic2": pair["relative_rate_gap_mic2"],
        },
        "excluded_exp203_seed": original["excluded_exp203_seed"],
        "excluded_original_pair": {key: {"path": rec["path"], "sha256": rec["sha256"]}
                                   for key, rec in original["audio"].items()},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=HERE / "selection-manifest-fresh.json")
    args = parser.parse_args()
    original = json.loads(ORIGINAL.read_text(encoding="utf-8"))
    if original["schema"] != "exp205-selection-manifest-v1" or len(original["speakers"]) != 54:
        raise ValueError("unexpected original manifest")

    speakers, dropped = [], []
    for row in original["speakers"]:
        speaker = row["speaker"]
        seed_stem = stem(Path(row["excluded_exp203_seed"]["path"]))
        original_pair = {"a": row["audio"]["A_mic1"]["path"], "b": row["audio"]["B_mic1"]["path"],
                         "a_mic2": row["audio"]["A_mic2"]["path"], "b_mic2": row["audio"]["B_mic2"]["path"]}
        reproduced = select_pair(speaker, {seed_stem})
        if reproduced is None or any(reproduced[k] != v for k, v in original_pair.items()):
            raise ValueError(f"selection rule does not reproduce the original pair for {speaker}")
        excluded = {seed_stem, stem(Path(original_pair["a"])), stem(Path(original_pair["b"]))}
        fresh = select_pair(speaker, excluded)
        if fresh is None:
            dropped.append({"speaker": speaker, "reason": "no pair satisfies the rule after excluding the original pair"})
            continue
        speakers.append(speaker_row(speaker, fresh, row))
        print(f"{speaker} {Path(fresh['a']).name} {Path(fresh['b']).name}", flush=True)

    n = len(speakers)
    result = {
        "schema": "exp212-fresh-pair-manifest-v1",
        "scope": "real_audio_metadata_and_text_only_no_clone_or_score_outcomes",
        "corpus": original["corpus"],
        "selection": {
            **original["selection"],
            "roster": "the 54 EXP-205 speakers; second pair per speaker under the identical rule",
            "excluded_per_speaker": ["EXP-203 seed utterance", "both utterances of the EXP-205 pair (both microphones)"],
            "rule_reproduces_original_pair_when_only_seed_excluded": True,
        },
        "counts": {"speakers": n, "seed_arms": 2, "prompt_microphones": 2, "generated_texts": 4, "systems": 4,
                   "expected_clones": n * 64, "comparisons_per_speaker_per_direction": 32},
        "dropped_speakers": dropped,
        "generation": original["generation"],
        "speakers": speakers,
        "input_hashes": {
            "original_manifest": sha256(ORIGINAL),
            "feasibility_source": sha256(FEASIBILITY_SOURCE),
            "builder": sha256(Path(__file__).resolve()),
        },
    }
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "MANIFEST_BUILT_NO_OUTCOMES", "speakers": n, "dropped": len(dropped),
                      "expected_clones": n * 64, "manifest_sha256": sha256(args.out)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
