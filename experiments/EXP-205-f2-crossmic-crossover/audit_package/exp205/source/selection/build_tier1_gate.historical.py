"""EXP-204: draw seed pairs and run the SV-distinguishability gate (CPU).

Gate as amended 2026-08-16, BEFORE running: seed A and seed B must be
distinguishable in SV-embedding space -- cos(A,B) below the 90th percentile of
that speaker's own within-speaker between-utterance cosine distribution. The
pair must not be among the 10% most similar that speaker produces.

The gate exists so that a null result means "the clone does not track its seed"
rather than "no method could have succeeded". It sits on the SV axis, not the
channel family: VCTK records each speaker in one session through fixed
microphones, so channel is nearly constant between two utterances of one speaker,
and EXP-203 already showed retrieval survives a complete pickup-path change --
channel is demonstrably not the cue under test.

Because the gate conditions the result, it reports how many candidate pairs it
REJECTED and out of how many, so the claim can be scoped to "when two seeds from
a speaker are distinguishable".

Writes pairs.json and gate.json. Exit 1 if too few speakers survive.
"""
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

EXP = Path(__file__).resolve().parents[1]
VCTK = Path.home() / "icassp-runs/vctk/wav48_silence_trimmed"
TXT = Path.home() / "icassp-runs/vctk/txt"
FEAT = Path.home() / "icassp-runs/EXP-203-f2-vctk-crossmic/features/pool_mic1.npz"
N_SPK, POOL_N, SEED = 30, 50, 0
MIN_SPK = 25
DUR_LO, DUR_HI = 4.0, 10.0
GATE_PCTL = 90


def utt_text(spk, path):
    """VCTK transcripts drop the _micN suffix the audio carries."""
    return (TXT / spk / f"{Path(path).stem.split('_mic')[0]}.txt").read_text().strip()


def unit(v):
    v = np.asarray(v, dtype=np.float64)
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)


def main():
    sys.path.insert(0, str(EXP.parents[0] / "EXP-202-f2-campaign/scripts"))
    import analyze_campaign as ac

    pool_f = ac.load_npz(FEAT)          # mic1 features already extracted by EXP-203
    by_spk = {}
    for path in pool_f:
        by_spk.setdefault(Path(path).parts[-2], []).append(path)

    rng = np.random.default_rng(SEED)
    speakers = sorted(s for s, v in by_spk.items() if len(v) >= 20)
    print(f"speakers with >=20 cached mic1 utterances: {len(speakers)}", flush=True)

    considered = rejected = 0
    pairs, gate_rows = {}, []
    for spk in speakers:
        paths = sorted(by_spk[spk])
        E = unit(np.stack([pool_f[p]["sv"] for p in paths]))
        C = E @ E.T

        durs = {}
        for p in paths:
            info = sf.info(p)
            durs[p] = info.frames / info.samplerate
        elig = [i for i, p in enumerate(paths) if DUR_LO <= durs[p] <= DUR_HI]
        if len(elig) < 2:
            continue
        # Reference distribution over the SAME population the pair is drawn from.
        # Taking it over all pairs rejected 51% at a 90th-percentile gate: longer
        # utterances give more stable embeddings and so higher cosines, so
        # duration-eligible pairs are systematically more similar than the
        # general pool and were being rejected for length, not similarity.
        sub = C[np.ix_(elig, elig)]
        iu = np.triu_indices(len(elig), 1)
        thresh = float(np.percentile(sub[iu], GATE_PCTL))
        # draw A and B from opposite halves of the utterance list, not adjacent
        half = len(elig) // 2
        a_pool, b_pool = elig[:half], elig[half:]
        if not a_pool or not b_pool:
            continue
        ia = int(rng.choice(a_pool))
        ib = int(rng.choice(b_pool))
        considered += 1
        cos_ab = float(C[ia, ib])
        passed = bool(cos_ab < thresh)
        gate_rows.append({"spk": spk, "cos_ab": round(cos_ab, 4),
                          "within_p90": round(thresh, 4), "passed": passed})
        if not passed:
            rejected += 1
            continue
        cand = [paths[i] for i in
                rng.choice(len(paths), min(POOL_N, len(paths)), replace=False)]
        for k in (ia, ib):
            if paths[k] not in cand:
                cand[rng.integers(len(cand))] = paths[k]
        pairs[spk] = {"seed_A": paths[ia], "seed_B": paths[ib],
                      "pool": sorted(set(cand) | {paths[ia], paths[ib]}),
                      "text_A": utt_text(spk, paths[ia]),
                      "text_B": utt_text(spk, paths[ib])}

    keep = sorted(pairs)[:N_SPK]
    pairs = {k: pairs[k] for k in keep}
    gate = {"n_pairs_considered": considered, "n_rejected": rejected,
            "rejection_rate": round(rejected / max(considered, 1), 4),
            "gate_percentile": GATE_PCTL, "n_speakers_kept": len(pairs),
            "per_speaker": gate_rows}
    (EXP / "gate.json").write_text(json.dumps(gate, indent=1))
    (EXP / "pairs.json").write_text(json.dumps(pairs, indent=1))
    print(f"considered {considered} pairs, REJECTED {rejected} "
          f"({100*rejected/max(considered,1):.1f}%), kept {len(pairs)} speakers")
    if len(pairs) < MIN_SPK:
        print(f"GATE FAIL: {len(pairs)} < {MIN_SPK} speakers")
        return 1
    print("GATE PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
