#!/usr/bin/env python3
"""Key-anchored checks for the Branch-A F2 manuscript.

Every result literal is tied to a named artifact field and a local sentence or
table row.  The checker also guards the design constants and load-bearing scope
language.  It intentionally does not use a bag-of-values search.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXP205 = ROOT / "experiments/EXP-205-f2-crossmic-crossover"
PACKAGE = EXP205 / "audit_package/exp205"
EXP202 = ROOT / "experiments/EXP-202-f2-campaign"
EXP204 = ROOT / "experiments/EXP-204-f2-seed-crossover"
TEX_RAW = (HERE / "main.tex").read_text(encoding="utf-8")
PROSE = re.sub(r"\s+", " ", TEX_RAW)
TEX = PROSE.replace("$", "")

RESULT = json.loads((PACKAGE / "result.json").read_text(encoding="utf-8"))
AUDIT = json.loads((PACKAGE / "audit.json").read_text(encoding="utf-8"))
MANIFEST = json.loads((PACKAGE / "manifest.portable.json").read_text(encoding="utf-8"))
EXECUTION = json.loads((PACKAGE / "execution-config.portable.json").read_text(encoding="utf-8"))
OLD_RESULTS = json.loads((EXP202 / "results.json").read_text(encoding="utf-8"))
OLD_CHAPTER = json.loads((EXP202 / "crossed_controls.json").read_text(encoding="utf-8"))
OLD_OPEN = json.loads((EXP202 / "open_set_intervals.json").read_text(encoding="utf-8"))
OLD_CROSSOVER = json.loads((EXP204 / "crossover.json").read_text(encoding="utf-8"))
ANCESTRY = json.loads((EXP205 / "roster_ancestry_sensitivity.json").read_text(encoding="utf-8"))
ARM = json.loads((EXP205 / "arm_pairing_sensitivity.json").read_text(encoding="utf-8"))


def f3(value: float) -> str:
    return f"{value:.3f}".lstrip("0")


def standalone(haystack: str, needle: str) -> bool:
    start = 0
    while True:
        pos = haystack.find(needle, start)
        if pos < 0:
            return False
        before = haystack[pos - 1 : pos]
        after = haystack[pos + len(needle) : pos + len(needle) + 1]
        if not (before and before.isdigit()) and not (after and after.isdigit()):
            return True
        start = pos + 1


def unique_window(anchor: str, chars: int = 900) -> str:
    count = TEX.count(anchor)
    if count != 1:
        raise AssertionError(f"anchor {anchor!r} occurs {count} times")
    pos = TEX.index(anchor)
    return TEX[pos : pos + chars]


def require_window(fails: list[str], name: str, anchor: str, values: list[str], chars: int = 900) -> None:
    try:
        text = unique_window(anchor, chars)
    except AssertionError as exc:
        fails.append(f"WINDOW {name}: {exc}")
        return
    for value in values:
        if not standalone(text, value):
            fails.append(f"WINDOW {name}: {value!r} absent near {anchor!r}")


def require_row(fails: list[str], name: str, marker: str, values: list[str]) -> None:
    lines = [line.replace("$", "") for line in TEX_RAW.splitlines() if marker in line]
    if len(lines) != 1:
        fails.append(f"ROW {name}: marker {marker!r} matched {len(lines)} lines")
        return
    line, cursor = lines[0], 0
    for value in values:
        pos = line.find(value, cursor)
        if pos < 0:
            fails.append(f"ROW {name}: {value!r} absent or out of order")
            return
        cursor = pos + len(value)


def main() -> int:
    fails: list[str] = []
    checks = 0

    # Trust roots and result branch must match the package exposed to readers.
    expected_roots = {
        "execution_config": "c82586074c5e7ec6aad9b21a101968f213f4849f35624492400041be1d2bd294",
        "manifest": "4b879491f02badf252365aa4d2b3caa22402c04301c60ed5e02bd06d43f19b2d",
        "receipt": "aae8c2d3dda873f4f72ae2458d8983b5e9a3b2328c4d28b452530e7ead8e55ad",
        "scores": "fe3633f063ab8be6716553ad6bda3d311e97c6351d25dfd5735600367fb9c54e",
        "result": "fc51fc71625a44c18a2b566d81ef85a15ff8b318b07eda4f59ddf37753884e0b",
    }
    for key, value in expected_roots.items():
        checks += 1
        if AUDIT["original_trust_roots"][key] != value:
            fails.append(f"TRUST ROOT {key} changed")
    if RESULT["headline_permission"] != "BIDIRECTIONAL_HEADLINE_PERMITTED":
        fails.append("sealed result no longer permits bidirectional headline")
    checks += 1

    p = RESULT["directions"]["primary_mic1_to_mic2"]
    r = RESULT["directions"]["reverse_mic2_to_mic1"]
    same = RESULT["same_microphone_diagnostics_no_verdict"]

    # Abstract: all four predeclared cross-microphone estimates and intervals.
    abstract_values = []
    for node in (p["ecapa"], p["wavlm"], r["ecapa"], r["wavlm"]):
        abstract_values.extend([f3(node["point"]), f3(node["stability_interval_95"][0]), f3(node["stability_interval_95"][1])])
    require_window(fails, "abstract exact result", "With fixed speaker-verification readouts", abstract_values, 900)
    checks += len(abstract_values)

    # Pooled table rows, including diagnostics in fixed column order.
    pooled_rows = [
        ("mic1$\\rightarrow$mic2 & ECAPA", p["ecapa"], same["same_mic1"]["ecapa"]),
        ("mic2$\\rightarrow$mic1 & ECAPA", r["ecapa"], same["same_mic2"]["ecapa"]),
    ]
    # The WavLM marker occurs twice; bind the reverse row by its exact leading whitespace.
    for marker, cross, diagnostic in pooled_rows:
        require_row(
            fails,
            f"pooled {marker}",
            marker,
            [f3(cross["point"]), f3(cross["stability_interval_95"][0]), f3(cross["stability_interval_95"][1]),
             f3(diagnostic["point"]), f3(diagnostic["stability_interval_95"][0]), f3(diagnostic["stability_interval_95"][1])],
        )
        checks += 6
    primary_wavlm_line = [line.replace("$", "") for line in TEX_RAW.splitlines() if "& WavLM & \\textbf{.622" in line]
    if len(primary_wavlm_line) != 1:
        fails.append("ROW primary WavLM missing or ambiguous")
    else:
        node, diag = p["wavlm"], same["same_mic1"]["wavlm"]
        for value in (f3(node["point"]), f3(node["stability_interval_95"][0]), f3(node["stability_interval_95"][1]),
                      f3(diag["point"]), f3(diag["stability_interval_95"][0]), f3(diag["stability_interval_95"][1])):
            if value not in primary_wavlm_line[0]:
                fails.append(f"ROW primary WavLM missing {value}")
    checks += 6
    reverse_wavlm_line = [line.replace("$", "") for line in TEX_RAW.splitlines() if "& WavLM & \\textbf{.620" in line]
    if len(reverse_wavlm_line) != 1:
        fails.append("ROW reverse WavLM missing or ambiguous")
    else:
        node, diag = r["wavlm"], same["same_mic2"]["wavlm"]
        for value in (f3(node["point"]), f3(node["stability_interval_95"][0]), f3(node["stability_interval_95"][1]),
                      f3(diag["point"]), f3(diag["stability_interval_95"][0]), f3(diag["stability_interval_95"][1])):
            if value not in reverse_wavlm_line[0]:
                fails.append(f"ROW reverse WavLM missing {value}")
    checks += 6

    # Descriptive system table: exact ordering and exact source keys.
    system_rows = (
        ("ECAPA, P &", "primary_mic1_to_mic2", "ecapa"),
        ("ECAPA, R &", "reverse_mic2_to_mic1", "ecapa"),
        ("WavLM, P &", "primary_mic1_to_mic2", "wavlm"),
        ("WavLM, R &", "reverse_mic2_to_mic1", "wavlm"),
    )
    for marker, direction, encoder in system_rows:
        values = [f3(RESULT["system_points_no_intervals"][direction][encoder][system]) for system in ("f5", "xtts", "cosy", "seedvc")]
        require_row(fails, f"systems {marker}", marker, values)
        checks += 4

    bar_rows = (
        ("E   & primary ECAPA", [f3(p["ecapa"]["stability_interval_95"][0])]),
        ("O   & ECAPA point", [f3(p["ecapa"]["point"]), f3(p["ecapa"]["stability_interval_95"][0])]),
        ("R   & primary WavLM", [f3(p["wavlm"]["stability_interval_95"][0])]),
        ("$D_E$ & reverse ECAPA", [f3(r["ecapa"]["stability_interval_95"][0])]),
        ("$D_R$ & reverse WavLM", [f3(r["wavlm"]["stability_interval_95"][0])]),
    )
    for marker, values in bar_rows:
        require_row(fails, f"frozen bar {marker}", marker, values + ["PASS"])
        checks += len(values) + 1

    observed_duration_gap = max(
        value
        for speaker in MANIFEST["speakers"]
        for value in (speaker["match"]["duration_gap_mic1_s"], speaker["match"]["duration_gap_mic2_s"])
    )
    observed_rate_gap = max(
        value
        for speaker in MANIFEST["speakers"]
        for value in (speaker["match"]["relative_rate_gap_mic1"], speaker["match"]["relative_rate_gap_mic2"])
    )
    require_window(fails, "realized metadata match", "selected roster is substantially tighter", [f"{observed_duration_gap:.3f}".lstrip("0"), f"{100 * observed_rate_gap:.2f}"], 300)
    checks += 2

    paired = ARM["paired_capture_integrity"]
    require_window(
        fails,
        "paired-capture integrity",
        "frozen manifest independently supports the pairing",
        [str(paired["selected_event_pairs"]), "identical frame counts", "sampling rates"],
        320,
    )
    checks += 3

    speaker_counts = []
    for direction, encoder in (("primary_mic1_to_mic2", "ecapa"), ("reverse_mic2_to_mic1", "ecapa"),
                               ("primary_mic1_to_mic2", "wavlm"), ("reverse_mic2_to_mic1", "wavlm")):
        values = RESULT["directions"][direction][encoder]["speaker_means"]
        speaker_counts.append(sum(value > .5 for value in values))
    require_window(
        fails,
        "speaker heterogeneity",
        "exposes the heterogeneity hidden by pooled points",
        [str(value) for value in speaker_counts] + [".50", "1,728"],
        700,
    )
    checks += 6

    ancestry_rows = (
        ("Pass, not retained", "tier1_gate_pass_not_selected"),
        ("Fail                &", "tier1_gate_fail"),
        ("Not evaluated", "tier1_gate_not_evaluated"),
    )
    for marker, key in ancestry_rows:
        node = ANCESTRY["strata"][key]
        values = [str(node["directions"]["primary_mic1_to_mic2"]["ecapa"]["n_speakers"])]
        for encoder in ("ecapa", "wavlm"):
            values.extend(
                f3(node["directions"][direction][encoder]["point"])
                for direction in ("primary_mic1_to_mic2", "reverse_mic2_to_mic1")
            )
        require_row(fails, f"ancestry {key}", marker, values)
        checks += len(values)

    arm_values = []
    for encoder in ("ecapa", "wavlm"):
        for direction in ("primary_mic1_to_mic2", "reverse_mic2_to_mic1"):
            arm_values.extend(f3(ARM["arm_results"][direction][encoder][arm]["follow_rate"]) for arm in ("A", "B"))
    require_window(
        fails,
        "arm-separated sensitivity",
        "Arm-separated points also oppose a one-candidate shortcut",
        arm_values + [str(ARM["cells_above_chance"]), f3(ARM["minimum_cell"]["follow_rate"])],
        700,
    )
    checks += len(arm_values) + 2

    # Historical evidence is tied to its original, named artifacts.
    loose = [OLD_RESULTS[f"{system}|lt_native|clean"]["fused"]["rank1_x_chance"] for system in ("f5", "xtts", "cosy", "seedvc")]
    chapter = [OLD_CHAPTER["systems"][system]["honest_within_chapter"]["fused"]["auc"] for system in ("f5", "xtts", "cosy", "seedvc")]
    require_window(fails, "antecedent evidence", "Earlier same-speaker mixed-chapter pools", [f"{min(loose):.1f}", f"{max(loose):.1f}", f3(min(chapter)), f3(max(chapter))], 520)
    old = OLD_CROSSOVER["POOLED"]
    require_window(fails, "EXP-204 correction", "A subsequent two-seed crossover", [f3(old["follow_rate"]), f3(old["ci95"][0]), f3(old["ci95"][1]), ".90"], 430)
    checks += 8

    ndcf = [OLD_OPEN[system]["ndcf"] for system in ("f5", "xtts", "cosy", "seedvc")]
    non_touching = sum(not OLD_OPEN[system]["ndcf_ci_touches_reject_all"] for system in ("f5", "xtts", "cosy", "seedvc"))
    count_word = {0: "zero", 1: "one", 2: "two", 3: "three", 4: "four"}[non_touching]
    require_window(fails, "open-set boundary", "In our earlier paired present/absent evaluation", [f"{min(ndcf):.2f}".lstrip("0"), f"{max(ndcf):.2f}", ".01", "1.00", count_word], 500)
    checks += 5

    # Design constants and frozen decision rule.
    required_statements = {
        "title": "Which Conditioning Recording Does a Voice Clone Follow?",
        "cross-microphone title": "A Balanced Cross-Microphone Crossover",
        "complete clone census": "=3{,}456$ clones",
        "fixed roster": "all 54 speakers remain",
        "roster ancestry": "not embedding-naive or population-sampled",
        "no outcome selection": "never inspects an embedding, score or clone outcome",
        "closed-set threat model": "exactly one of which supplied the prompt",
        "no training on evaluation cells": "does not train on these speakers or systems",
        "candidate-bias control": "both signed arms",
        "opposite-mic control": "opposite microphone",
        "duration match": "at most .25\\,s",
        "rate match": "at most 5\\%",
        "speaker unit": "32 dependent comparisons per direction",
        "bootstrap": "100,000 times with a frozen seed",
        "F5 package version": f"F5-TTS {EXECUTION['generators']['f5']['package_version']}",
        "XTTS package version": f"XTTS-v2 {EXECUTION['generators']['xtts']['package_version']}",
        "fixed-roster interval scope": "not population-coverage confidence",
        "non-rescue": "Diagnostics cannot rescue any conjunct",
        "chronology boundary": "not presented as public preregistration",
        "recovery disclosure": "one-field repair is documented in the same internal history",
        "separate recomputation": "reconstructs every result field to $2\\times10^{-15}$",
        "portable score artifact": "portable artifact contains sanitized scores",
        "portable verifier": "a verifier for the 3,456-row grid",
        "score-level boundary": "score-level reproducibility, not proof of audio-to-score ancestry",
        "descriptive systems": "system-specific points are descriptive",
        "old miss not repaired": "does not retroactively change the earlier",
        "closed set": "closed-set two-alternative question known to contain the source",
        "mechanism boundary": "The carrier is unidentified",
        "transcript remains bundled": "transcript is not a complete explanation",
        "second SV readout scope": "do not establish representation independence",
        "heldout scope": "not globally unseen or population-sampled",
        "training exposure": "VCTK exposure cannot be excluded",
        "no audio release": "not model weights, source audio or playable impersonations",
    }
    for name, literal in required_statements.items():
        checks += 1
        if literal not in PROSE:
            fails.append(f"MISSING STATEMENT {name}: {literal!r}")

    retired = {
        "Reference-Recording Leakage": "superseded catalogue title",
        "Transcript-Disjoint Identifiability Floor": "superseded catalogue subtitle",
        "the target does follow the seed": "old selected crossover phrasing",
        "full confirmation": "EXP-204 did not pass its .90 point bar",
        "no session cue remains": "cross-microphone design does not remove event acoustics",
        "population confidence interval": "intervals are fixed-roster composition summaries",
        "open-set confirmation": "closed-set result cannot establish presence detection",
        "operationally large": "author-declared effect-size bar is not deployment utility",
        "simultaneous VCTK": "VCTK documentation does not establish timing",
        "representation replication": "second encoder is corroboration",
        "bidirectionally replicated": "direction reversal is not an independent replication",
        "pre-registered": "internal chronology has no public timestamp",
        "independently audited": "public release does not establish institutional independence",
        "exact frozen source bytes": "three public source files are portability adaptations",
        "ECAPA-only artifact": "second SV readout is not mechanistically independent",
        "is reproduced after": "microphone reversal is within-design persistence",
    }
    for literal, reason in retired.items():
        if literal in TEX_RAW:
            fails.append(f"RETIRED {literal!r}: {reason}")

    print(f"key-anchored checks: {checks}; retired guards: {len(retired)}")
    for fail in fails:
        print("  " + fail)
    print("RESULT:", "FAIL" if fails else "PASS")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
