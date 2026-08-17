#!/usr/bin/env python3
"""Render the fixed-roster speaker heterogeneity plot used in F2."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


HERE = Path(__file__).resolve().parent
RESULT = HERE.parents[1] / "experiments/EXP-205-f2-crossmic-crossover/audit_package/exp205/result.json"
OUT = HERE / "fig_exp205_speakers.pdf"


def main() -> int:
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    groups = [
        ("ECAPA\nP", "primary_mic1_to_mic2", "ecapa"),
        ("ECAPA\nR", "reverse_mic2_to_mic1", "ecapa"),
        ("WavLM\nP", "primary_mic1_to_mic2", "wavlm"),
        ("WavLM\nR", "reverse_mic2_to_mic1", "wavlm"),
    ]
    fig, ax = plt.subplots(figsize=(3.35, 1.72))
    offsets = np.linspace(-0.16, 0.16, 54)
    # Deterministic permutation prevents stacked discrete speaker means without
    # implying an additional stochastic sample.
    offsets = offsets[np.random.default_rng(2052027).permutation(54)]
    for x, (label, direction, encoder) in enumerate(groups, start=1):
        node = result["directions"][direction][encoder]
        values = np.asarray(node["speaker_means"], dtype=float)
        ax.scatter(
            x + offsets,
            values,
            s=8,
            facecolors="white" if encoder == "wavlm" else "0.65",
            edgecolors="0.2",
            linewidths=0.35,
            zorder=2,
        )
        lo, hi = node["stability_interval_95"]
        point = node["point"]
        ax.errorbar(
            x,
            point,
            yerr=[[point - lo], [hi - point]],
            fmt="D",
            color="black",
            markersize=3.2,
            capsize=2.2,
            linewidth=1.0,
            zorder=4,
        )
    ax.axhline(0.5, color="0.25", linestyle="--", linewidth=0.7, zorder=1)
    ax.set_xlim(0.55, 4.45)
    ax.set_ylim(0.34, 1.025)
    ax.set_xticks(range(1, 5), [item[0] for item in groups])
    ax.set_yticks([0.4, 0.5, 0.6, 0.8, 1.0])
    ax.set_ylabel("follow rate", labelpad=1)
    ax.tick_params(axis="both", labelsize=7, length=2)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_linewidth(0.6)
    ax.spines["bottom"].set_linewidth(0.6)
    fig.tight_layout(pad=0.25)
    fig.savefig(OUT, bbox_inches="tight")
    print(OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
