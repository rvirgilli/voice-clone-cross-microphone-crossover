#!/usr/bin/env python3
"""Prove that load-bearing F2 checker guards actually fail under mutation."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
TEX = HERE / "main.tex"
BACKUP = HERE / ".main.tex.mutation-backup"


def run_checker() -> bool:
    result = subprocess.run(
        [sys.executable, str(HERE / "check_numbers.py")],
        capture_output=True,
        text=True,
        cwd=HERE,
    )
    return result.returncode == 0


CASES = [
    ("corrupt abstract primary ECAPA", ".896 [.869,.921]", ".896 [.769,.921]"),
    ("corrupt abstract reverse WavLM", ".620 [.589,.653]", ".620 [.489,.653]"),
    ("corrupt pooled primary ECAPA", "\\textbf{.896 [.869,.921]}", "\\textbf{.996 [.869,.921]}"),
    ("corrupt pooled primary WavLM", "\\textbf{.622 [.593,.652]}", "\\textbf{.722 [.593,.652]}"),
    ("corrupt pooled reverse ECAPA", "\\textbf{.907 [.881,.931]}", "\\textbf{.807 [.881,.931]}"),
    ("corrupt pooled reverse WavLM", "\\textbf{.620 [.589,.653]}", "\\textbf{.720 [.589,.653]}"),
    ("swap ECAPA primary system columns", "ECAPA, P & .942 & .833", "ECAPA, P & .833 & .942"),
    ("corrupt WavLM reverse Seed-VC", "WavLM, R & .657 & .574 & .657 & .593", "WavLM, R & .657 & .574 & .657 & .693"),
    ("corrupt loose-pool range", "9.7--16.4", "8.7--16.4"),
    ("corrupt EXP-204 interval", ".806 [.757,.854]", ".806 [.657,.854]"),
    ("corrupt minDCF range", ".79--1.00", ".69--1.00"),
    ("corrupt realized duration gap", "worst duration gap is .112", "worst duration gap is .212"),
    ("corrupt realized duration-per-byte gap", "duration-per-byte gap is 2.57", "duration-per-byte gap is 3.57"),
    ("corrupt paired-capture count", "all 108 selected", "all 107 selected"),
    ("corrupt ancestry pass stratum", "Pass, not retained & 36 & .908/.907", "Pass, not retained & 36 & .808/.907"),
    ("corrupt ancestry fail stratum", "Fail                & 15 & .879/.915", "Fail                & 15 & .779/.915"),
    ("corrupt arm-separated ECAPA", "ECAPA is\n.889/.903", "ECAPA is\n.789/.903"),
    ("corrupt minimum arm cell", "minimum .523", "minimum .423"),
    ("corrupt frozen E bar", "E   & primary ECAPA LCB $>.50$ & .869 & PASS", "E   & primary ECAPA LCB $>.50$ & .769 & PASS"),
    ("drop F5 package version", "F5-TTS 1.1.22", "F5-TTS"),
    ("drop portable verifier", "a verifier for the 3,456-row grid", "a listing for the 3,456-row grid"),
    ("drop no-outcome selection", "never inspects an embedding, score or clone outcome", "inspects clone outcomes"),
    ("drop roster ancestry", "not\nembedding-naive or population-sampled", "fully\nembedding-naive and population-sampled"),
    ("drop fixed-roster interval scope", "not population-coverage confidence", "a confidence interval"),
    ("drop non-rescue rule", "cannot rescue any conjunct", "may rescue a conjunct"),
    ("drop recovery disclosure", "one-field repair is documented in the same internal history", "one-field repair is undocumented"),
    ("drop closed-set boundary", "two-alternative question known to contain the source", "attribution question"),
    ("drop VCTK exposure limitation", "VCTK exposure cannot be\nexcluded", "training exposure is\ncontrolled"),
    ("drop release boundary", "scores and aggregate/per-speaker results, but not model weights", "scores and all model weights"),
    ("revive superseded title", "Which Conditioning Recording Does a Voice Clone Follow?", "Reference-Recording Leakage"),
    ("revive operational overclaim", "meets the predeclared large-effect criterion", "is operationally large"),
    ("revive simultaneity overclaim", "paired VCTK", "simultaneous VCTK"),
    ("revive population inference", "not population-coverage confidence", "population confidence interval"),
]


def main() -> int:
    shutil.copyfile(TEX, BACKUP)
    original = BACKUP.read_text(encoding="utf-8")
    if not run_checker():
        print("BASELINE FAILS — fix paper/checker before mutation testing")
        BACKUP.unlink()
        return 1
    uncaught, invalid = [], []
    try:
        for label, find, replacement in CASES:
            mutated = original.replace(find, replacement, 1)
            if mutated == original:
                invalid.append(label)
                continue
            TEX.write_text(mutated, encoding="utf-8")
            if run_checker():
                uncaught.append(label)
            TEX.write_text(original, encoding="utf-8")
    finally:
        TEX.write_text(original, encoding="utf-8")
        BACKUP.unlink(missing_ok=True)
    caught = len(CASES) - len(uncaught) - len(invalid)
    print(f"mutations: {len(CASES)}  caught: {caught}  UNCAUGHT: {len(uncaught)}  INVALID: {len(invalid)}")
    for label in uncaught:
        print(f"  UNCAUGHT {label}")
    for label in invalid:
        print(f"  INVALID {label}")
    return 1 if uncaught or invalid else 0


if __name__ == "__main__":
    raise SystemExit(main())
