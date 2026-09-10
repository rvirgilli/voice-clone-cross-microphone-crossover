"""Verify the public this work result from its released comparison-level scores and speaker summaries."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from types import ModuleType

import numpy as np


HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
RESULT = DATA / "extension_result.json"
INPUT_MANIFEST = DATA / "extension_input_manifest.json"
SCORES = DATA / "clone_to_clone_scores.tsv"
ANALYZER = HERE / "extension_analyze.py"
VERDICT = HERE / "extension_verdict.py"
# The result binds the hashes of the analyzer and verdict modules as they ran; the released
# copies carry documented portability adaptations, so they are pinned here separately.
RELEASED_ANALYZER_SHA256 = "462c48f524b74d83c1e5c8c3e5cd85b221e00b76aaccafa2108b7eb44a22c7eb"
RELEASED_VERDICT_SHA256 = "1ff13c9ad22d2cca41a1746ce21347b0e83e722c1870956f4cdbbc22b8a70854"
EXPECTED_RESULT_SHA256 = "cda08b8b68aea8cc8f45c22e7270a25cf42ddbc360626ffdddaeece285e4005a"
BOOTSTRAPS = 100_000
BOOTSTRAP_SEED = 2062027
DIRECTIONS = ("primary_mic1_to_mic2", "reverse_mic2_to_mic1")
READOUTS = ("ecapa", "wavlm")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def authenticated_verdict(expected_hash: str) -> ModuleType:
    source = VERDICT.read_bytes()
    require(hashlib.sha256(source).hexdigest() == expected_hash, "verdict hash")
    name = "_exp206_public_verdict"
    module = ModuleType(name)
    module.__file__ = str(VERDICT)
    previous = sys.modules.get(name)
    sys.modules[name] = module
    try:
        exec(compile(source, str(VERDICT), "exec"), module.__dict__)
    finally:
        if previous is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous
    require(callable(getattr(module, "decide", None)), "verdict namespace")
    return module


def main() -> int:
    require(sha256(RESULT) == EXPECTED_RESULT_SHA256, "result hash")
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    input_manifest = json.loads(INPUT_MANIFEST.read_text(encoding="utf-8"))
    require(result.get("schema") == "exp206-scientific-result-v1", "schema")
    require(result.get("status") == "SCIENTIFIC_RESULT", "status")
    expected_counts = {
        "speakers": 54,
        "systems": 4,
        "texts": 4,
        "prompt_microphones": 2,
        "seed_arms": 2,
        "clones": 3456,
        "comparisons_per_speaker_per_direction": 288,
        "bootstrap_replicates": BOOTSTRAPS,
        "bootstrap_seed": BOOTSTRAP_SEED,
    }
    require(result.get("counts") == expected_counts, "counts")
    require(result.get("embedding_dimensions") == {"ecapa": 192, "wavlm": 512}, "dimensions")
    roots = result.get("artifact_hashes", {})
    require(sha256(ANALYZER) == RELEASED_ANALYZER_SHA256, "released analyzer hash")
    require(sha256(VERDICT) == RELEASED_VERDICT_SHA256, "released verdict hash")
    print(f"historical analyzer/verdict roots as run: {roots.get('analyzer')[:12]}… / {roots.get('verdict')[:12]}… (recorded, not the released copies)")
    require(roots.get("input_manifest") == sha256(INPUT_MANIFEST), "input root")
    require(
        roots.get("execution_config") == input_manifest.get("execution_config_sha256"),
        "execution config root",
    )
    require(
        roots.get("execution_receipt") == input_manifest.get("execution_receipt_sha256"),
        "execution receipt root",
    )

    # Comparison-level scores: every clone identity resolves in the frozen input manifest and
    # the per-speaker follow rates and margins rebuild from the cosines before aggregation.
    speakers = input_manifest["speakers"]
    clone_hashes = {
        (r["speaker"], r["system"], str(r["text_index"]), r["prompt_mic"], r["seed_arm"]): r["clone_sha256"]
        for r in input_manifest["clones"]
    }
    follows: dict[tuple[str, str, str], list[float]] = {}
    margins: dict[tuple[str, str, str], list[float]] = {}
    with SCORES.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    require(len(rows) == 54 * 2 * 288 * 2, "comparison census")
    for row in rows:
        query_mic, candidate_mic = {"primary_mic1_to_mic2": ("mic1", "mic2"), "reverse_mic2_to_mic1": ("mic2", "mic1")}[row["direction"]]
        require(row["query_prompt_mic"] == query_mic and row["candidate_prompt_mic"] == candidate_mic, "direction microphones")
        require(row["candidate_system"] != row["query_system"] and row["candidate_text_index"] != row["query_text_index"], "cross-generator cross-text")
        require(row["own_seed_arm"] == row["query_seed_arm"] and row["other_seed_arm"] != row["query_seed_arm"], "candidate arms")
        for clone, system, text, mic, arm in (
            ("query_clone_sha256", "query_system", "query_text_index", "query_prompt_mic", "query_seed_arm"),
            ("own_clone_sha256", "candidate_system", "candidate_text_index", "candidate_prompt_mic", "own_seed_arm"),
            ("other_clone_sha256", "candidate_system", "candidate_text_index", "candidate_prompt_mic", "other_seed_arm"),
        ):
            key = (row["speaker"], row[system], row[text], row[mic], row[arm])
            require(key in clone_hashes and clone_hashes[key].startswith(row[clone]) and len(row[clone]) >= 12, f"clone identity {key}")
        own, other = float(row["cos_own_event"]), float(row["cos_other_event"])
        cell_key = (row["direction"], row["readout"], row["speaker"])
        follows.setdefault(cell_key, []).append(1.0 if own > other else 0.0 if own < other else 0.5)
        margins.setdefault(cell_key, []).append(own - other)
    require(len(follows) == 2 * 2 * 54 and all(len(v) == 288 for v in follows.values()), "speaker census")

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(0, 54, size=(BOOTSTRAPS, 54), dtype=np.int32)
    verdict_cells = {}
    for direction in DIRECTIONS:
        require(set(result["directions"].get(direction, {})) == set(READOUTS), "direction grid")
        verdict_cells[direction] = {}
        for readout in READOUTS:
            cell = result["directions"][direction][readout]
            require(
                set(cell)
                == {
                    "point",
                    "stability_interval_95",
                    "speaker_means",
                    "speaker_mean_margins",
                },
                "cell schema",
            )
            vector = np.asarray(cell["speaker_means"], dtype=np.float64)
            margins_vector = np.asarray(cell["speaker_mean_margins"], dtype=np.float64)
            require(vector.shape == (54,) and margins_vector.shape == (54,), "speaker arrays")
            require(np.isfinite(vector).all() and np.isfinite(margins_vector).all(), "finite arrays")
            require(bool(((0.0 <= vector) & (vector <= 1.0)).all()), "follow range")
            rebuilt = np.asarray([np.mean(follows[(direction, readout, s)]) for s in speakers])
            rebuilt_margins = np.asarray([np.mean(margins[(direction, readout, s)]) for s in speakers])
            require(np.allclose(rebuilt, vector, rtol=0.0, atol=1e-9), "speaker means from comparison scores")
            require(np.allclose(rebuilt_margins, margins_vector, rtol=0.0, atol=1e-9), "speaker margins from comparison scores")
            print(f"{direction} {readout}: {float(rebuilt.mean()):.3f} recomputed from {len(speakers)} speakers × 288 comparisons")
            point = float(vector.mean())
            distribution = vector[indices].mean(axis=1)
            lo, hi = np.quantile(distribution, (0.025, 0.975))
            observed_interval = np.asarray(cell["stability_interval_95"], dtype=np.float64)
            require(math.isclose(point, float(cell["point"]), abs_tol=2e-15), "point")
            require(
                np.allclose(observed_interval, np.asarray([lo, hi]), rtol=0.0, atol=2e-15),
                "interval",
            )
            verdict_cells[direction][readout] = {"point": point, "lcb": float(lo)}

    verdict = authenticated_verdict(RELEASED_VERDICT_SHA256).decide(verdict_cells).as_dict()
    require(verdict == result.get("verdict"), "verdict")
    require(verdict["material_event_signal"] is True, "material verdict")
    require(
        verdict["manuscript_permission"]
        == "ABSTRACT_AND_CONCLUSION_UPGRADE_PERMITTED",
        "manuscript permission",
    )
    print("PASS — clone-to-clone per-speaker accuracies and margins rebuild from the released comparison-level cosine scores, and the aggregate estimates, intervals and verdict reproduce from the speaker summaries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
