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
import tempfile
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
# This is a numerical-tolerance bar for the explicitly tested exact gain/delay transforms,
# not a universal perceptual near-duplicate threshold.  Boundary-shift injections are run
# on every source capture; the closest genuine pair is more than four orders above the bar.
DUPLICATE_BAR = 1e-5
FROZEN_MANIFEST_SHA256 = "4b879491f02badf252365aa4d2b3caa22402c04301c60ed5e02bd06d43f19b2d"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def capture_census_sha256(speakers):
    """Bind the measurement to the exact 108 manifest pairs without machine paths."""
    census = []
    for spk in speakers:
        for event in EXPECTED_EVENTS:
            audio = spk["audio"]
            census.append({
                "speaker": spk["speaker"],
                "event": event,
                "mic1_sha256": audio[f"{event}_mic1"]["sha256"],
                "mic2_sha256": audio[f"{event}_mic2"]["sha256"],
            })
    payload = json.dumps(census, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def best_alignment(x, y, sr, lag_ms=LAG_MS):
    """Return max projection, least-squares residual and lag over ``+-lag_ms``.

    For lag d, ``shift(x)[t] = x[t-d]`` inside the signal and zero outside it.  The
    denominator uses the norm of that zero-padded, truncated shift—not the norm of the
    unshifted source.  This detail makes a real padded delay score zero even at the search
    boundary.  The residual is the exact optimum of ``||y-g*shift(x)||/||y||`` over a
    scalar gain g and the declared lag window.
    """
    n = min(len(x), len(y))
    a = np.asarray(x[:n], dtype=np.float64)
    b = np.asarray(y[:n], dtype=np.float64)
    if n < 2:
        raise ValueError("alignment needs at least two samples")
    max_lag = min(int(sr * lag_ms / 1000.0), n - 1)
    size = 1 << int(np.ceil(np.log2(2 * n - 1)))
    # ifft(fft(y)*conj(fft(x)))[d] = sum_t y[t] x[t-d]
    corr = np.fft.irfft(np.fft.rfft(b, size) * np.conj(np.fft.rfft(a, size)), size)
    lags = np.arange(-max_lag, max_lag + 1, dtype=np.int64)
    dots = np.asarray([corr[d if d >= 0 else size + d] for d in lags])

    prefix_energy = np.concatenate(([0.0], np.cumsum(a * a)))
    shifted_energy = np.empty(lags.size, dtype=np.float64)
    nonnegative = lags >= 0
    shifted_energy[nonnegative] = prefix_energy[n - lags[nonnegative]]
    shifted_energy[~nonnegative] = prefix_energy[n] - prefix_energy[-lags[~nonnegative]]
    target_energy = float(np.dot(b, b))
    tiny = np.finfo(np.float64).tiny
    if target_energy <= tiny or not np.any(shifted_energy > tiny):
        raise ValueError("alignment is undefined for a silent capture")
    projection = np.abs(dots) / np.sqrt(np.maximum(shifted_energy * target_energy, tiny))
    index = int(np.argmax(projection))
    r = min(float(projection[index]), 1.0)  # clip floating overshoot for exact copies
    return r, float(np.sqrt(max(0.0, 1.0 - r * r))), int(lags[index])


def shifted_copy(x, lag):
    """Zero-padded, non-wrapped shift with the same length as x."""
    out = np.zeros_like(x)
    if lag >= 0:
        if lag < len(x):
            out[lag:] = x[: len(x) - lag]
    else:
        offset = -lag
        if offset < len(x):
            out[: len(x) - offset] = x[offset:]
    return out


def injection_controls(x, sr):
    """A6 controls on one real source, including both search-window boundaries."""
    limit = min(int(sr * LAG_MS / 1000.0), len(x) - 1)
    shifts = sorted({-limit, -1, 1, limit} | {lag for lag in (-120, 120) if abs(lag) <= limit})
    controls = {
        "exact_copy": best_alignment(x, x.copy(), sr)[1],
        "gain_scaled_copy": best_alignment(x, 0.4 * x, sr)[1],
    }
    for lag in shifts:
        controls[f"shifted_copy_{lag:+d}"] = best_alignment(x, shifted_copy(x, lag), sr)[1]
    controls[f"gain_scaled_shifted_copy_{limit:+d}"] = best_alignment(
        x, 0.4 * shifted_copy(x, limit), sr
    )[1]
    controls[f"gain_scaled_shifted_copy_{-limit:+d}"] = best_alignment(
        x, 0.4 * shifted_copy(x, -limit), sr
    )[1]
    return controls, shifts


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

    rows, dup_hash = [], 0
    injection_maxima = {}
    tested_shifts = set()
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
            controls, shifts = injection_controls(xa, sr)
            tested_shifts.update(shifts)
            for name, value in controls.items():
                injection_maxima[name] = max(injection_maxima.get(name, 0.0), value)
            aligned, resid, best_lag = best_alignment(xa, ya, sr)
            rows.append({
                "speaker": spk["speaker"], "event": event, "frames": n,
                "corr_zero_lag": float(np.corrcoef(xa, ya)[0, 1]),
                "corr_max_lag": aligned, "residual_after_alignment": resid,
                "best_lag_samples": best_lag,
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
        "schema": "exp205-posthoc-channel-distinctness-v3",
        "manifest_sha256": sha256(MANIFEST),
        "frozen_manifest_sha256": FROZEN_MANIFEST_SHA256,
        "capture_census_sha256": capture_census_sha256(speakers),
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
        "injection_controls": {
            "n_source_captures": len(rows),
            "tested_shift_samples": sorted(tested_shifts),
            "max_residual_by_transform": {
                name: round(float(value), 10)
                for name, value in sorted(injection_maxima.items())
            },
            "max_residual": round(float(max(injection_maxima.values())), 10),
        },
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
    if inj["n_source_captures"] != EXPECTED_PAIRS:
        problems.append("injection controls did not cover every source capture")
    if inj["max_residual"] > DUPLICATE_BAR:
        problems.append(f"injected duplicates not detected: {inj} — the test cannot fail")
    return problems


def main():
    tmp = None
    try:
        out = measure()
        problems = assertions(out)
        if problems:
            raise RuntimeError("; ".join(problems))
        # A5: publish only after every assertion has passed, and atomically.
        fd, tmp_name = tempfile.mkstemp(prefix=f".{OUT.name}.", suffix=".tmp", dir=OUT.parent)
        tmp = Path(tmp_name)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(out, indent=1) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, OUT)
        tmp = None
    except BaseException as exc:
        # A failed recomputation must not leave a stale result that looks current.
        OUT.unlink(missing_ok=True)
        if tmp is not None:
            tmp.unlink(missing_ok=True)
        if isinstance(exc, KeyboardInterrupt):
            raise
        print(f"FAIL {exc}", file=sys.stderr)
        return 1
    print(json.dumps(out, indent=1))
    print(f"\nOK: {out['n_event_captures']} authenticated pairs, none duplicated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
