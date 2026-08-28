#!/usr/bin/env python3
"""Post-hoc check: are the two microphone channels genuinely different captures?

The crossover's validity rests on generation and attribution passing through DIFFERENT
captures. `arm_pairing_sensitivity.py` established paired-capture INTEGRITY (equal frame
counts, matching sample rates), but an equal frame count is also what duplication produces,
so it cannot establish distinctness. This does.

Duplication is tested transform-aware. An exact copy, a gain-scaled copy and a short-shifted
copy are all duplicates; hashing catches only the first and zero-lag correlation only the
first two. Raw correlation is the wrong statistic in both directions: a shifted duplicate
scores near zero, while genuine mic1/mic2 pairs score 0.92 median once aligned, because two
microphones on one event differ mostly by delay and a mild filter.

The primary statistic is therefore the RESIDUAL after optimal gain and delay alignment,
min over lag and gain of ||y - g*shift(x)|| / ||y||. A duplicate under any of the three
transforms drives it to zero; a genuinely different capture cannot, because no single gain
and delay turns one microphone's transfer function into another's.

Descriptive, post-hoc, no verdict change and no model inference.
"""

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

EXP = Path(__file__).resolve().parent
MANIFEST = EXP.parent / "data" / "selection_manifest.json"
OUT = EXP.parent / "data" / "channel_distinctness.json"

EXPECTED_SPEAKERS = 54
EXPECTED_EVENTS = ("A", "B")
EXPECTED_PAIRS = EXPECTED_SPEAKERS * len(EXPECTED_EVENTS)
LAG_MS = 50.0
# A pair is a duplicate if the residual after optimal gain+delay alignment falls below this.
# Measured separation on this roster: injected duplicates reach 0.000000 (exact), 0.000000
# (gain x0.4) and 0.000981 (shift +120 samples); the closest real pair is 0.2141. The bar
# sits ~20x above the worst injection and ~10x below the closest real pair.
DUPLICATE_BAR = 0.02


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def best_alignment(x, y, sr, lag_ms=LAG_MS):
    """(max |normalised correlation| over +-lag_ms, residual after that alignment).

    The correlation is gain- and shift-invariant; the residual is what separates a
    different microphone from a transformed copy.
    """
    n = min(len(x), len(y))
    a = np.asarray(x[:n], dtype=np.float64)
    b = np.asarray(y[:n], dtype=np.float64)
    a -= a.mean()
    b -= b.mean()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return float("nan")
    lag = int(sr * lag_ms / 1000.0)
    size = 1 << int(np.ceil(np.log2(2 * n)))
    corr = np.fft.irfft(np.fft.rfft(a, size) * np.conj(np.fft.rfft(b, size)), size)
    window = np.concatenate([corr[: lag + 1], corr[-lag:]]) if lag else corr[:1]
    r = float(np.abs(window).max() / (na * nb))
    return r, float(np.sqrt(max(0.0, 1.0 - r * r)))


def injection_controls(x, sr):
    """A6: the test must catch every transform class, checked on every run.

    Injections are built from a real capture rather than synthetic noise: a wrapped shift
    of white noise mismatches its wrapped tail completely, which understates detection.
    A delayed copy is modelled by padding, which is what a real delay produces.
    """
    delayed = np.concatenate([np.zeros(120), x[:-120]])
    return {"exact_copy": round(best_alignment(x, x.copy(), sr)[1], 6),
            "gain_scaled_copy": round(best_alignment(x, 0.4 * x, sr)[1], 6),
            "shifted_copy_120": round(best_alignment(x, delayed, sr)[1], 6)}


def read_pair(a_meta, b_meta):
    """Authenticate both files against the frozen manifest, then load them."""
    pa, pb = Path(a_meta["path"]), Path(b_meta["path"])
    for p, meta in ((pa, a_meta), (pb, b_meta)):
        if not p.is_file():
            raise FileNotFoundError(f"manifest file missing: {p}")
        if sha256(p) != meta["sha256"]:
            raise ValueError(f"hash mismatch against frozen manifest: {p}")
    x, sr1 = sf.read(pa)
    y, sr2 = sf.read(pb)
    if sr1 != sr2:
        raise ValueError(f"sample-rate mismatch: {pa} {sr1} vs {pb} {sr2}")
    if len(x) != len(y):
        raise ValueError(f"frame-count mismatch: {pa} {len(x)} vs {pb} {len(y)}")
    return x, y, sr1, a_meta["sha256"] == b_meta["sha256"]


def measure():
    man = json.loads(MANIFEST.read_text())
    speakers = man["speakers"]
    if len(speakers) != EXPECTED_SPEAKERS:
        raise ValueError(f"expected {EXPECTED_SPEAKERS} speakers, manifest has {len(speakers)}")

    rows, dup_hash, injections = [], 0, None
    for spk in speakers:
        for event in EXPECTED_EVENTS:
            a = spk["audio"].get(f"{event}_mic1")
            b = spk["audio"].get(f"{event}_mic2")
            if a is None or b is None:
                # A4: a missing arm is a census failure, never a silent skip.
                raise ValueError(f"{spk['speaker']} event {event}: missing mic1 or mic2 entry")
            x, y, sr, same_hash = read_pair(a, b)
            dup_hash += int(same_hash)
            n = min(len(x), len(y))
            xa, ya = x[:n], y[:n]
            if injections is None:
                injections = injection_controls(xa, sr)
            aligned, resid = best_alignment(xa, ya, sr)
            rows.append({
                "speaker": spk["speaker"], "event": event, "frames": n,
                "corr_zero_lag": float(np.corrcoef(xa, ya)[0, 1]),
                "corr_max_lag": aligned, "residual_after_alignment": resid,
                "rms_ratio": float(np.sqrt(np.mean(ya ** 2)) /
                                   max(np.sqrt(np.mean(xa ** 2)), 1e-12)),
            })

    if len(rows) != EXPECTED_PAIRS:
        raise ValueError(f"expected {EXPECTED_PAIRS} authenticated pairs, measured {len(rows)}")

    z = np.array([r["corr_zero_lag"] for r in rows])
    l = np.array([r["corr_max_lag"] for r in rows])
    res = np.array([r["residual_after_alignment"] for r in rows])
    ratio = np.array([r["rms_ratio"] for r in rows])
    worst = min(rows, key=lambda r: r["residual_after_alignment"])
    return {
        "schema": "exp205-posthoc-channel-distinctness-v2",
        "manifest_sha256": sha256(MANIFEST),
        "status": "POST_HOC_DESCRIPTIVE_NO_VERDICT_CHANGE",
        "mic1": man["corpus"]["mic1"], "mic2": man["corpus"]["mic2"],
        "n_speakers": len(speakers), "n_event_captures": len(rows),
        "byte_identical_pairs": dup_hash,
        "duplicate_bar_alignment_residual": DUPLICATE_BAR, "lag_window_ms": LAG_MS,
        "waveform_correlation_zero_lag": {
            "median": round(float(np.median(z)), 4),
            "min": round(float(z.min()), 4), "max": round(float(z.max()), 4)},
        "waveform_correlation_max_lag": {
            "median": round(float(np.median(l)), 4),
            "min": round(float(l.min()), 4), "max": round(float(l.max()), 4)},
        "residual_after_alignment": {
            "median": round(float(np.median(res)), 4),
            "min": round(float(res.min()), 4), "max": round(float(res.max()), 4),
            "closest_pair": f"{worst['speaker']}/{worst['event']}"},
        "injection_controls": injections,
        "rms_ratio": {"median": round(float(np.median(ratio)), 4),
                      "min": round(float(ratio.min()), 4),
                      "max": round(float(ratio.max()), 4)},
    }


def assertions(out):
    """Every failure condition, evaluated BEFORE anything is published (A5)."""
    problems = []
    if out["n_event_captures"] != EXPECTED_PAIRS:
        problems.append(f"census: {out['n_event_captures']} pairs, expected {EXPECTED_PAIRS}")
    if out["byte_identical_pairs"] != 0:
        problems.append("a mic1/mic2 pair is byte-identical: the crossover control is vacuous")
    r = out["residual_after_alignment"]
    if r["min"] <= DUPLICATE_BAR:
        problems.append(
            f"pair {r['closest_pair']} has alignment residual {r['min']}, at or below the "
            f"{DUPLICATE_BAR} duplicate bar")
    inj = out["injection_controls"]
    if any(v > DUPLICATE_BAR for v in inj.values()):
        problems.append(f"injected duplicates not detected: {inj} — the test cannot fail")
    return problems


def main():
    out = measure()
    problems = assertions(out)
    if problems:
        for p in problems:
            print(f"FAIL {p}", file=sys.stderr)
        sys.exit(1)
    # A5: publish only after every assertion has passed, and atomically.
    tmp = OUT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(out, indent=1))
    os.replace(tmp, OUT)
    print(json.dumps(out, indent=1))
    print(f"\nOK: {out['n_event_captures']} authenticated pairs, none duplicated")


if __name__ == "__main__":
    main()
