"""Metadata-only feasibility for a held-out cross-mic seed crossover.

This script never opens clone audio, clone features, EXP-204 outcomes, or model
scores. It uses only:

* the names of real VCTK files cached by EXP-203;
* real-audio headers for duration;
* VCTK transcripts; and
* the speaker keys already consumed by EXP-204 tiers 1 and 2.

It reports how many untouched speakers admit two 4--10 s references matched on
duration and the exact seconds-per-UTF-8-byte cue used by the F5-TTS harness in
both microphone directions.
It does not select a final pair or authorize generation.
"""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import soundfile as sf


ROOT = Path(__file__).resolve().parents[2]
EXP203_RUN = Path.home() / "icassp-runs/EXP-203-f2-vctk-crossmic"
EXP204_RUN = Path.home() / "icassp-runs/EXP-204-f2-seed-crossover"
VCTK_TXT = Path.home() / "icassp-runs/vctk/txt"
VCTK_AUDIO = Path.home() / "icassp-runs/vctk/wav48_silence_trimmed"
POOL_NPZ = EXP203_RUN / "features/pool_mic1.npz"
EXP203_POOL = ROOT / "experiments/EXP-203-f2-vctk-crossmic/pools/pool_seed1_cand1.json"
TIER1 = ROOT / "experiments/EXP-204-f2-seed-crossover/pairs.json"
TIER2 = ROOT / "experiments/EXP-204-f2-seed-crossover/tier2_pairs.json"

DURATION_TOLS = (0.25, 0.50, 1.00)
RELATIVE_RATE_TOLS = (0.02, 0.05, 0.10)


def real_paths_from_npz_names(path: Path) -> list[Path]:
    """Read ZIP member names only; no feature array is opened."""
    with zipfile.ZipFile(path) as archive:
        suffix = "|sv.npy"
        names = {
            member[: -len(suffix)]
            for member in archive.namelist()
            if member.endswith(suffix)
        }
    return [Path(name) for name in sorted(names)]


def transcript_path(audio: Path) -> Path:
    stem = audio.stem.split("_mic", 1)[0]
    return VCTK_TXT / audio.parent.name / f"{stem}.txt"


def seconds_per_utf8_byte(audio: Path, duration: float) -> float:
    text = transcript_path(audio).read_text(encoding="utf-8").strip()
    n_bytes = len(text.encode("utf-8"))
    if n_bytes == 0:
        raise ValueError(f"empty transcript for {audio}")
    return duration / n_bytes


def mic2_counterpart(mic1: Path) -> Path:
    name = mic1.name
    if "_mic1.flac" not in name:
        raise ValueError(f"not a mic1 path: {mic1}")
    return mic1.with_name(name.replace("_mic1.flac", "_mic2.flac"))


def relative_gap(a: float, b: float) -> float:
    return abs(a - b) / ((a + b) / 2.0)


def generated_speakers() -> set[str]:
    if not EXP204_RUN.exists():
        return set()
    result: set[str] = set()
    for wav in EXP204_RUN.glob("**/clones/*.wav"):
        result.add(wav.name.split("_", 1)[0])
    for wav in EXP204_RUN.glob("**/tier2/clones/*.wav"):
        result.add(wav.name.split("_", 1)[0])
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    tier1 = set(json.loads(TIER1.read_text(encoding="utf-8")))
    tier2 = set(json.loads(TIER2.read_text(encoding="utf-8")))
    consumed = tier1 | tier2
    prior_trials = json.loads(EXP203_POOL.read_text(encoding="utf-8"))["trials"]
    prior_seed = {
        speaker: Path(record["gen_ref"]["path"])
        for speaker, record in prior_trials.items()
    }

    cached_by_speaker: dict[str, list[Path]] = {}
    for audio in real_paths_from_npz_names(POOL_NPZ):
        cached_by_speaker.setdefault(audio.parent.name, []).append(audio)

    eligible_speakers = set(cached_by_speaker)
    heldout = sorted(eligible_speakers - consumed)
    generated_overlap = sorted(set(heldout) & generated_speakers())

    grids = {
        f"duration_le_{dur:.2f}__relative_rate_le_{rate:.2f}": []
        for dur in DURATION_TOLS
        for rate in RELATIVE_RATE_TOLS
    }
    rows = []
    for speaker in heldout:
        candidates = []
        # EXP-203's 50-file cache defines the eligible 108-speaker roster, but
        # it is not the acquisition universe for a new experiment. Search all
        # simultaneous VCTK recordings of each held-out speaker so the old
        # random pool cap cannot manufacture a feasibility failure.
        for audio in sorted((VCTK_AUDIO / speaker).glob(f"{speaker}_*_mic1.flac")):
            # The speaker was present in EXP-203, but neither member of the new
            # pair may be the exact utterance that seeded that earlier clone.
            if audio == prior_seed[speaker]:
                continue
            info = sf.info(audio)
            duration = info.frames / info.samplerate
            mic2 = mic2_counterpart(audio)
            if not (4.0 <= duration <= 10.0 and mic2.is_file()):
                continue
            mic2_info = sf.info(mic2)
            mic2_duration = mic2_info.frames / mic2_info.samplerate
            if not 4.0 <= mic2_duration <= 10.0:
                continue
            candidates.append(
                {
                    "path": audio,
                    "mic2_path": mic2,
                    "duration_mic1": duration,
                    "duration_mic2": mic2_duration,
                    "rate_mic1": seconds_per_utf8_byte(audio, duration),
                    "rate_mic2": seconds_per_utf8_byte(mic2, mic2_duration),
                }
            )

        pairs = []
        for i, left in enumerate(candidates):
            for right in candidates[i + 1 :]:
                pairs.append(
                    {
                        "a": str(left["path"]),
                        "b": str(right["path"]),
                        "a_mic2": str(left["mic2_path"]),
                        "b_mic2": str(right["mic2_path"]),
                        "duration_gap_mic1_s": abs(
                            left["duration_mic1"] - right["duration_mic1"]
                        ),
                        "duration_gap_mic2_s": abs(
                            left["duration_mic2"] - right["duration_mic2"]
                        ),
                        "relative_rate_gap_mic1": relative_gap(
                            left["rate_mic1"], right["rate_mic1"]
                        ),
                        "relative_rate_gap_mic2": relative_gap(
                            left["rate_mic2"], right["rate_mic2"]
                        ),
                    }
                )

        best = min(
            pairs,
            key=lambda row: (
                max(row["relative_rate_gap_mic1"], row["relative_rate_gap_mic2"]),
                max(row["duration_gap_mic1_s"], row["duration_gap_mic2_s"]),
                row["a"],
                row["b"],
            ),
            default=None,
        )
        draft_eligible = [
            row
            for row in pairs
            if max(row["duration_gap_mic1_s"], row["duration_gap_mic2_s"])
            <= 0.25
            and max(
                row["relative_rate_gap_mic1"], row["relative_rate_gap_mic2"]
            )
            <= 0.05
        ]
        draft_pair = min(
            draft_eligible,
            key=lambda row: (
                max(
                    row["duration_gap_mic1_s"] / 0.25,
                    row["duration_gap_mic2_s"] / 0.25,
                    row["relative_rate_gap_mic1"] / 0.05,
                    row["relative_rate_gap_mic2"] / 0.05,
                ),
                row["duration_gap_mic1_s"] / 0.25
                + row["duration_gap_mic2_s"] / 0.25
                + row["relative_rate_gap_mic1"] / 0.05
                + row["relative_rate_gap_mic2"] / 0.05,
                max(row["relative_rate_gap_mic1"], row["relative_rate_gap_mic2"]),
                max(row["duration_gap_mic1_s"], row["duration_gap_mic2_s"]),
                row["a"],
                row["b"],
            ),
            default=None,
        )
        for dur in DURATION_TOLS:
            for rate in RELATIVE_RATE_TOLS:
                key = f"duration_le_{dur:.2f}__relative_rate_le_{rate:.2f}"
                n_pairs = sum(
                    max(row["duration_gap_mic1_s"], row["duration_gap_mic2_s"])
                    <= dur
                    and max(
                        row["relative_rate_gap_mic1"],
                        row["relative_rate_gap_mic2"],
                    )
                    <= rate
                    for row in pairs
                )
                if n_pairs:
                    grids[key].append(speaker)

        rows.append(
            {
                "speaker": speaker,
                "eligible_utterances": len(candidates),
                "possible_pairs": len(pairs),
                "best_metadata_match": best,
                "draft_pair_duration_le_0.25_relative_rate_le_0.05": draft_pair,
            }
        )

    result = {
        "scope": "metadata_only_no_clone_audio_features_scores_or_outcomes",
        "candidate_universe": "all paired VCTK mic1/mic2 files for EXP-203's 108-speaker roster",
        "pair_gate": "duration and seconds-per-UTF8-byte tolerances must hold on both microphones",
        "eligible_exp203_speakers": len(eligible_speakers),
        "exp204_tier1_speakers": len(tier1),
        "exp204_tier2_speakers": len(tier2),
        "consumed_unique_speakers": len(consumed),
        "heldout_speakers": len(heldout),
        "heldout_ids": heldout,
        "heldout_with_existing_exp204_clone": generated_overlap,
        "exp203_prior_seed_excluded_per_heldout_speaker": True,
        "exp203_prior_seed_paths_excluded": {
            speaker: str(prior_seed[speaker]) for speaker in heldout
        },
        "feasible_speaker_count_by_grid": {
            key: len(value) for key, value in grids.items()
        },
        "feasible_speaker_ids_by_grid": grids,
        "per_speaker": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "eligible_exp203_speakers": len(eligible_speakers),
                "consumed_unique_speakers": len(consumed),
                "heldout_speakers": len(heldout),
                "generated_overlap": len(generated_overlap),
                "feasible_speaker_count_by_grid": result[
                    "feasible_speaker_count_by_grid"
                ],
            },
            indent=2,
        )
    )
    if (
        len(heldout) != 54
        or generated_overlap
        or any(
            row["draft_pair_duration_le_0.25_relative_rate_le_0.05"] is None
            for row in rows
        )
    ):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
