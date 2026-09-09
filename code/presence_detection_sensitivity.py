#!/usr/bin/env python3
"""Descriptive within-speaker presence-detection boundary for EXP-205.

The frozen EXP-205 analysis ranks two candidate events per clone.  This post-hoc
analysis asks the harder single-threshold question on the same scores: with one
global cosine threshold, can the clone's source event be told apart from the other
event of the same speaker, seen through the same (opposite-to-prompt) microphone?

Per direction and encoder, the target score of a clone is its cosine to the
candidate of its own conditioning event and the non-target score is its cosine to
the other event's candidate, so every clone contributes exactly one target and one
non-target trial.  We report the equal error rate and the normalized minimum
detection cost (C_miss = C_fa = 1, normalized by the cheaper trivial decision, so
rejecting everything costs 1.0) at target prior .01 and at the .5 prior of the
two-candidate threat model, with 95% percentile intervals from whole-speaker
resampling.  The threshold grid includes a value above every score so the
reject-all decision is representable.  It is descriptive, does not alter the frozen
verdict, and performs no model inference.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


EXP = Path(__file__).resolve().parent
DATA = EXP.parent / "data"
SCORES = DATA / "scores.tsv"
RESULT = DATA / "result.json"
OUT = DATA / "presence_detection_sensitivity.json"
PINS = {
    SCORES: "535d3a5cba9f98bb830175004214b62977047f629113248ecca07c11cbd9c387",
}
DIRECTIONS = {"mic1": "primary_mic1_to_mic2", "mic2": "reverse_mic2_to_mic1"}
ENCODERS = ("ecapa", "wavlm")
TARGET_PRIOR = 0.01
# The two-candidate threat model has one source among two candidates, prior .5.
THREAT_MODEL_PRIOR = 0.5
COST_MISS = 1.0
COST_FALSE_ALARM = 1.0
BOOTSTRAPS = 100_000
BOOTSTRAP_SEED = 2052027
SPEAKERS = 54
TRIALS_PER_SPEAKER = 32


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def error_rates(target: np.ndarray, nontarget: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Miss and false-alarm rates over every distinct score plus one threshold above all.

    A trial is accepted when its score is at or above the threshold, so the miss rate
    counts targets strictly below it and the false-alarm rate counts non-targets at or
    above it.
    """
    target = np.sort(target)
    nontarget = np.sort(nontarget)
    thresholds = np.unique(np.concatenate([target, nontarget]))
    thresholds = np.append(thresholds, thresholds[-1] + 1.0)
    miss = np.searchsorted(target, thresholds, side="left") / target.size
    false_alarm = 1.0 - np.searchsorted(nontarget, thresholds, side="left") / nontarget.size
    return miss, false_alarm


def equal_error_rate(target: np.ndarray, nontarget: np.ndarray) -> float:
    miss, false_alarm = error_rates(target, nontarget)
    index = int(np.argmin(np.abs(miss - false_alarm)))
    return float((miss[index] + false_alarm[index]) / 2.0)


def normalized_min_dcf(target: np.ndarray, nontarget: np.ndarray, prior: float = TARGET_PRIOR) -> float:
    miss, false_alarm = error_rates(target, nontarget)
    cost = prior * COST_MISS * miss + (1.0 - prior) * COST_FALSE_ALARM * false_alarm
    floor = min(prior * COST_MISS, (1.0 - prior) * COST_FALSE_ALARM)
    return float(cost.min() / floor)


def percentile_interval(values: np.ndarray) -> list[float]:
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def load_trials() -> dict[tuple[str, str], tuple[np.ndarray, np.ndarray]]:
    """Per (direction, encoder): speaker-major target and non-target score matrices."""
    per_speaker: dict[tuple[str, str], dict[str, list[tuple[float, float]]]] = defaultdict(lambda: defaultdict(list))
    identities = set()
    with SCORES.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            identity = tuple(row[key] for key in ("speaker", "system", "text_index", "prompt_mic", "seed_arm"))
            if identity in identities:
                raise RuntimeError(f"duplicate score row: {identity}")
            identities.add(identity)
            speaker, _, _, prompt_mic, arm = identity
            other = "B" if arm == "A" else "A"
            for encoder in ENCODERS:
                per_speaker[(DIRECTIONS[prompt_mic], encoder)][speaker].append(
                    (float(row[f"{encoder}_{arm}"]), float(row[f"{encoder}_{other}"]))
                )
    if len(identities) != 3456:
        raise RuntimeError("score census is not 3,456 unique rows")
    cells = {}
    for key, speakers in per_speaker.items():
        if len(speakers) != SPEAKERS or any(len(v) != TRIALS_PER_SPEAKER for v in speakers.values()):
            raise RuntimeError(f"unexpected speaker census in {key}")
        ordered = [speakers[s] for s in sorted(speakers)]
        target = np.array([[t for t, _ in s] for s in ordered])
        nontarget = np.array([[n for _, n in s] for s in ordered])
        cells[key] = (target, nontarget)
    return cells


def main() -> int:
    for path, expected in PINS.items():
        observed = sha256(path)
        if observed != expected:
            raise RuntimeError(f"pin mismatch {path}: {observed}")
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    if result["counts"]["bootstrap_seed"] != BOOTSTRAP_SEED or result["counts"]["bootstrap_replicates"] != BOOTSTRAPS:
        raise RuntimeError("bootstrap settings differ from the frozen result")

    cells = load_trials()
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.integers(0, SPEAKERS, size=(BOOTSTRAPS, SPEAKERS))
    output_cells: dict[str, dict[str, dict]] = {}
    for direction in DIRECTIONS.values():
        output_cells[direction] = {}
        for encoder in ENCODERS:
            target, nontarget = cells[(direction, encoder)]
            eers = np.empty(BOOTSTRAPS)
            dcfs = np.empty(BOOTSTRAPS)
            dcfs_threat = np.empty(BOOTSTRAPS)
            for i in range(BOOTSTRAPS):
                t = target[draws[i]].ravel()
                n = nontarget[draws[i]].ravel()
                eers[i] = equal_error_rate(t, n)
                dcfs[i] = normalized_min_dcf(t, n)
                dcfs_threat[i] = normalized_min_dcf(t, n, THREAT_MODEL_PRIOR)
            ndcf = normalized_min_dcf(target.ravel(), nontarget.ravel())
            ndcf_ci = percentile_interval(dcfs)
            output_cells[direction][encoder] = {
                "n_target_trials": int(target.size),
                "n_nontarget_trials": int(nontarget.size),
                "eer": equal_error_rate(target.ravel(), nontarget.ravel()),
                "eer_ci95": percentile_interval(eers),
                "ndcf": ndcf,
                "ndcf_ci95": ndcf_ci,
                "ndcf_ci_excludes_reject_all": bool(ndcf_ci[1] < 1.0),
                "ndcf_threat_prior": normalized_min_dcf(target.ravel(), nontarget.ravel(), THREAT_MODEL_PRIOR),
                "ndcf_threat_prior_ci95": percentile_interval(dcfs_threat),
            }
    output = {
        "schema": "exp205-postreview-presence-detection-sensitivity-v1",
        "status": "POST_HOC_DESCRIPTIVE_NO_VERDICT_CHANGE",
        "pins": {path.name: expected for path, expected in PINS.items()},
        "trial_definition": (
            "Per clone, target = cosine to the opposite-microphone candidate of its own "
            "conditioning event; non-target = cosine to the same speaker's other event "
            "through that microphone. One global threshold per direction and encoder."
        ),
        "cost_model": {
            "target_prior": TARGET_PRIOR,
            "threat_model_prior": THREAT_MODEL_PRIOR,
            "cost_miss": COST_MISS,
            "cost_false_alarm": COST_FALSE_ALARM,
            "normalization": "divided by min(prior*C_miss, (1-prior)*C_fa); reject-all = 1.0",
            "threshold_grid": "every distinct score plus one value above the maximum",
        },
        "bootstrap": {
            "unit": "speaker",
            "replicates": BOOTSTRAPS,
            "seed": BOOTSTRAP_SEED,
            "interval": "percentile 2.5/97.5",
        },
        "directions": output_cells,
        "limitations": [
            "Trials are the 1,728 per-direction two-candidate comparisons split into one target and one non-target score each; they are not an independent open-set search over unknown candidates.",
            "Non-targets are same-speaker, paired-capture events, the hardest within-speaker case; cross-speaker impostors are not included.",
            "This analysis is descriptive and cannot alter or rescue the frozen pooled EXP-205 verdict.",
        ],
    }
    OUT.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output_cells, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
