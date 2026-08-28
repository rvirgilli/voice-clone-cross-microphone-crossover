"""EXP-204 tier 2: select duration-matched low/high dose seed pairs (CPU).

Within-speaker paired design: each speaker supplies a LOW-dose pair
(1-cos(A,B) <= 0.18) and a HIGH-dose pair (>= 0.40), MATCHED on mean duration
within 0.5 s.

Duration matching is not cosmetic. Selecting on dose extremes CREATES a
confound the phenomenon does not have: unmatched extreme selection gives
high-dose pairs 1.68 s shorter than low-dose ones, because shorter utterances
give less stable embeddings and therefore lower cosine. Tier 1 itself is clean
(partial r of follow on dose controlling duration = +.436 against a raw +.421);
the confound would have been manufactured by this design, not measured by it.

Writes tier2_pairs.json and tier2_selection.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

EXP = Path(__file__).resolve().parents[1]
TXT = Path.home() / "icassp-runs/vctk/txt"
FEAT = Path.home() / "icassp-runs/EXP-203-f2-vctk-crossmic/features/pool_mic1.npz"
# Thresholds relaxed 0.18/0.40 -> 0.20/0.38 before running: excluding tier-1
# speakers left only 18 at the tighter pair, and n=18 is UNDERPOWERED (min
# detectable .944 against a predicted .923). 24 speakers at contrast .245
# gives a predicted difference of .210 with min detectable .875.
LO_MAX, HI_MIN, DUR_TOL = 0.20, 0.38, 0.5
DUR_LO, DUR_HI, POOL_N, N_SPK, SEED = 4.0, 10.0, 50, 25, 0


def utt_text(spk, path):
    return (TXT / spk / f"{Path(path).stem.split('_mic')[0]}.txt").read_text().strip()


def main():
    sys.path.insert(0, str(EXP.parents[0] / "EXP-202-f2-campaign/scripts"))
    import analyze_campaign as ac

    pf = ac.load_npz(FEAT)
    by = {}
    for p in pf:
        by.setdefault(Path(p).parts[-2], []).append(p)
    rng = np.random.default_rng(SEED)

    # tier-1 speakers are excluded so the strata are not the same seeds re-used
    used = set(json.loads((EXP / "pairs.json").read_text()))
    out, sel = {}, []
    for spk, paths in sorted(by.items()):
        if spk in used:
            continue
        dur = {}
        for p in paths:
            i = sf.info(p)
            dur[p] = i.frames / i.samplerate
        elig = [p for p in paths if DUR_LO <= dur[p] <= DUR_HI]
        if len(elig) < 4:
            continue
        E = np.stack([pf[p]["sv"] for p in elig])
        E = E / np.linalg.norm(E, axis=1, keepdims=True)
        D = 1 - (E @ E.T)
        n = len(elig)
        lows = [(D[a, b], a, b) for a in range(n) for b in range(a + 1, n) if D[a, b] <= LO_MAX]
        highs = [(D[a, b], a, b) for a in range(n) for b in range(a + 1, n) if D[a, b] >= HI_MIN]
        if not lows or not highs:
            continue
        best = None
        for dl, a, b in lows:
            ml = (dur[elig[a]] + dur[elig[b]]) / 2
            for dh, c, e in highs:
                mh = (dur[elig[c]] + dur[elig[e]]) / 2
                gap = abs(mh - ml)
                if gap <= DUR_TOL and (best is None or gap < best[0]):
                    best = (gap, dl, dh, a, b, c, e, ml, mh)
        if best is None:
            continue
        gap, dl, dh, a, b, c, e = best[:7]
        cand = [elig[i] for i in rng.choice(len(elig), min(POOL_N, len(elig)), replace=False)]
        pool = sorted(set(cand) | {elig[a], elig[b], elig[c], elig[e]})
        out[spk] = {
            "low":  {"seed_A": elig[a], "seed_B": elig[b], "dose": round(float(dl), 4),
                     "text_A": utt_text(spk, elig[a]), "text_B": utt_text(spk, elig[b])},
            "high": {"seed_A": elig[c], "seed_B": elig[e], "dose": round(float(dh), 4),
                     "text_A": utt_text(spk, elig[c]), "text_B": utt_text(spk, elig[e])},
            "pool": pool}
        sel.append({"spk": spk, "dose_low": round(float(dl), 4), "dose_high": round(float(dh), 4),
                    "dur_gap_s": round(float(gap), 3),
                    "dur_low_s": round(float(best[7]), 2), "dur_high_s": round(float(best[8]), 2)})

    keep = [r["spk"] for r in sel][:N_SPK]
    out = {k: out[k] for k in keep}
    sel = [r for r in sel if r["spk"] in keep]
    dl = np.array([r["dose_low"] for r in sel]); dh = np.array([r["dose_high"] for r in sel])
    gaps = np.array([r["dur_gap_s"] for r in sel])
    summary = {"n_speakers": len(out), "tier1_speakers_excluded": len(used),
               "dose_low_mean": round(float(dl.mean()), 4),
               "dose_high_mean": round(float(dh.mean()), 4),
               "contrast_mean": round(float((dh - dl).mean()), 4),
               "duration_gap_mean_s": round(float(gaps.mean()), 3),
               "duration_gap_max_s": round(float(gaps.max()), 3),
               "per_speaker": sel}
    (EXP / "tier2_pairs.json").write_text(json.dumps(out, indent=1))
    (EXP / "tier2_selection.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "per_speaker"}, indent=1))
    if len(out) < 22:
        print(f"TOO FEW SPEAKERS: {len(out)}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
