#!/usr/bin/env python3
"""EXP-209: N-candidate closed-set scaling with metadata-matched same-speaker distractors.

Stage 'select' builds the distractor roster from VCTK metadata only (durations and
transcript byte counts) and writes distractors.json. Stage 'extract' embeds every
needed file with the paper's two readouts (reusing the EXP-207 extractor). Stage
'analyze' ranks each clone's own event among N candidates and writes the result.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

import os

# Historical run layout: these scripts document what ran. Point the two roots at copies of
# the private run directories and the experiment tree; nothing here reads the release itself.
RUNS = Path(os.environ.get("ICASSP_RUNS", "/runs"))
EXPERIMENTS = Path(os.environ.get("ICASSP_EXPERIMENTS", "/experiments"))
HF = Path(os.environ.get("HF_HUB_CACHE", "~/.cache/huggingface/hub")).expanduser()
EXP205 = EXPERIMENTS / "EXP-205-f2-crossmic-crossover"
EXP207 = EXPERIMENTS / "EXP-207-f2-readout-roster"
RUN205 = RUNS / "EXP-205-f2-crossmic-crossover"
VCTK = RUNS / "vctk"
HERE = Path(__file__).resolve().parent
DUR_RANGE = (4.0, 10.0)
DUR_TOL = 0.50
RATE_TOL = 0.10
N_VALUES = (2, 4, 8, 16)
BOOTSTRAPS = 10_000
SEED = 2052027
DIRECTIONS = {"mic1": "primary_mic1_to_mic2", "mic2": "reverse_mic2_to_mic1"}


def relative_gap(a: float, b: float) -> float:
    return abs(a - b) / ((a + b) / 2.0)


def utterance_meta(speaker: str, stem: str) -> dict | None:
    txt = VCTK / "txt" / speaker / f"{stem}.txt"
    mic1 = VCTK / "wav48_silence_trimmed" / speaker / f"{stem}_mic1.flac"
    mic2 = VCTK / "wav48_silence_trimmed" / speaker / f"{stem}_mic2.flac"
    if not (txt.is_file() and mic1.is_file() and mic2.is_file()):
        return None
    nbytes = len(txt.read_text(encoding="utf-8").strip().encode("utf-8"))
    out = {"stem": stem, "utf8_bytes": nbytes}
    for mic, path in (("mic1", mic1), ("mic2", mic2)):
        info = sf.info(str(path))
        duration = info.frames / info.samplerate
        out[mic] = {"path": str(path), "duration_s": duration, "rate": duration / nbytes}
    return out


def select(manifest: dict) -> dict:
    roster = {}
    for spk in manifest["speakers"]:
        speaker = spk["speaker"]
        used = {Path(spk["audio"][k]["path"]).name.rsplit("_", 1)[0] for k in ("A_mic1", "B_mic1")}
        pair = {}
        for arm in ("A", "B"):
            pair[arm] = {mic: {"duration_s": spk["audio"][f"{arm}_{mic}"]["duration_s"],
                               "rate": spk["audio"][f"{arm}_{mic}"]["duration_s"] / spk["transcripts"][arm]["utf8_bytes"]}
                         for mic in ("mic1", "mic2")}
        candidates = []
        for path in sorted((VCTK / "wav48_silence_trimmed" / speaker).glob(f"{speaker}_*_mic1.flac")):
            stem = path.name.rsplit("_", 1)[0]
            if stem in used:
                continue
            meta = utterance_meta(speaker, stem)
            if meta is None:
                continue
            ok = True
            gaps = []
            for mic in ("mic1", "mic2"):
                d, r = meta[mic]["duration_s"], meta[mic]["rate"]
                if not (DUR_RANGE[0] <= d <= DUR_RANGE[1]):
                    ok = False
                # Distractors are matched to the A/B pair mean, not to each member.
                mean_d = (pair["A"][mic]["duration_s"] + pair["B"][mic]["duration_s"]) / 2.0
                mean_r = (pair["A"][mic]["rate"] + pair["B"][mic]["rate"]) / 2.0
                dg = abs(d - mean_d)
                rg = relative_gap(r, mean_r)
                if dg > DUR_TOL or rg > RATE_TOL:
                    ok = False
                gaps.append((dg / DUR_TOL, rg / RATE_TOL, rg, dg))
            if not ok:
                continue
            # The paper's pair rule: largest tolerance-normalised gap, their sum, largest
            # rate gap, largest duration gap, then the path.
            key = (max(max(g[0], g[1]) for g in gaps), sum(g[0] + g[1] for g in gaps),
                   max(g[2] for g in gaps), max(g[3] for g in gaps), stem)
            candidates.append((key, meta))
        candidates.sort(key=lambda item: item[0])
        roster[speaker] = [meta for _, meta in candidates[: max(N_VALUES) - 2]]
    counts = {str(n): sum(len(v) >= n - 2 for v in roster.values()) for n in N_VALUES}
    return {"tolerances": {"duration_range_s": DUR_RANGE, "duration_tol_s": DUR_TOL, "rate_tol": RATE_TOL},
            "speakers_with_enough_distractors": counts, "roster": roster}


def load_features(feature_dir: Path, name: str) -> tuple[dict[str, int], np.ndarray]:
    data = np.load(feature_dir / f"{name}.npz")
    key = "ecapa" if name == "ecapa" else "wavlmsv_xvector"
    return {p: i for i, p in enumerate(data["paths"].tolist())}, data[key].astype(np.float64)


def analyze(manifest: dict, distractors: dict, feature_dir: Path) -> dict:
    with (RUN205 / "scores.tsv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    speakers = sorted({r["speaker"] for r in rows})
    real = {(s["speaker"], k): s["audio"][k]["path"] for s in manifest["speakers"] for k in ("A_mic1", "A_mic2", "B_mic1", "B_mic2")}
    rng = np.random.default_rng(SEED)
    result = {"schema": "exp209-ncandidate-scaling-v1", "status": "POST_HOC_DESCRIPTIVE_NO_VERDICT_CHANGE",
              "bootstrap": {"unit": "speaker", "replicates": BOOTSTRAPS, "seed": SEED}, "readouts": {}}
    for name in ("ecapa", "wavlmsv"):
        index, vectors = load_features(feature_dir, name)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        result["readouts"][name] = {}
        for n in N_VALUES:
            per = {d: defaultdict(list) for d in DIRECTIONS.values()}
            for row in rows:
                pool = distractors["roster"][row["speaker"]]
                if len(pool) < n - 2:
                    continue
                cand_mic = "mic2" if row["prompt_mic"] == "mic1" else "mic1"
                own = real[(row["speaker"], f"{row['seed_arm']}_{cand_mic}")]
                other = real[(row["speaker"], f"{'B' if row['seed_arm'] == 'A' else 'A'}_{cand_mic}")]
                cands = [own, other] + [d[cand_mic]["path"] for d in pool[: n - 2]]
                clone = vectors[index[str(RUN205 / row["clone_path"])]]
                scores = np.array([float(clone @ vectors[index[c]]) for c in cands])
                rank = 1 + int((scores[1:] > scores[0]).sum()) + 0.5 * int((scores[1:] == scores[0]).sum())
                per[DIRECTIONS[row["prompt_mic"]]][row["speaker"]].append(rank)
            cell = {}
            for direction, table in per.items():
                spk = [s for s in speakers if table[s]]
                r1 = np.array([np.mean([rk == 1 for rk in table[s]]) for s in spk])
                mrr = np.array([np.mean([1.0 / rk for rk in table[s]]) for s in spk])
                bits = np.array([np.log2(n) - np.mean([np.log2(rk) for rk in table[s]]) for s in spk])
                draws = rng.integers(0, len(spk), size=(BOOTSTRAPS, len(spk)))
                boot = r1[draws].mean(axis=1)
                cell[direction] = {"n_speakers": len(spk), "chance": 1.0 / n,
                                   "rank1": float(r1.mean()),
                                   "rank1_interval_95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                                   "mrr": float(mrr.mean()), "rank_disclosure_bits": float(bits.mean())}
            result["readouts"][name][str(n)] = cell
            print(f"{name:8s} N={n:2d} " + "  ".join(f"{d[:7]} rank1 {c['rank1']:.3f} [{c['rank1_interval_95'][0]:.3f},{c['rank1_interval_95'][1]:.3f}] n={c['n_speakers']}" for d, c in cell.items()), flush=True)
    e16 = result["readouts"]["ecapa"]["16"]
    e8 = result["readouts"]["ecapa"]["8"]
    strong = all(c["rank1_interval_95"][0] >= 0.50 for c in e16.values())
    bounded = all(c["rank1"] <= 0.25 for c in e8.values())
    result["reading"] = "STRONG_IN_POOL" if strong else "BOUNDED_AT_PAIR" if bounded else "AS_MEASURED"
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("select", "extract", "analyze"))
    parser.add_argument("--run", default=str(RUNS / "EXP-209-f2-ncandidate-scaling"))
    args = parser.parse_args()
    run = Path(args.run)
    run.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((EXP205 / "selection-manifest.json").read_text(encoding="utf-8"))
    if args.stage == "select":
        out = select(manifest)
        (run / "distractors.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
        print(json.dumps(out["speakers_with_enough_distractors"]))
        return 0
    distractors = json.loads((run / "distractors.json").read_text(encoding="utf-8"))
    if args.stage == "extract":
        sys.path.insert(0, str(EXP207))
        import extract_readouts as er
        paths = er.file_list()
        for pool in distractors["roster"].values():
            for d in pool:
                paths.extend([d["mic1"]["path"], d["mic2"]["path"]])
        paths = list(dict.fromkeys(paths))
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        for name in ("ecapa", "wavlmsv"):
            er.run_readout(name, paths, run / "features", device)
        return 0
    result = analyze(manifest, distractors, run / "features")
    result["distractor_census"] = distractors["speakers_with_enough_distractors"]
    (HERE / "ncandidate_result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(result["reading"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
