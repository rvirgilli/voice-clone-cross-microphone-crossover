"""Verify the public EXP-206 result from its released speaker summaries."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from types import ModuleType

import numpy as np


HERE = Path(__file__).resolve().parent
RESULT = HERE / "result.json"
INPUT_MANIFEST = HERE / "input-manifest.json"
ANALYZER = HERE / "analyze.py"
VERDICT = HERE / "verdict.py"
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
    require(roots.get("analyzer") == sha256(ANALYZER), "analyzer root")
    require(roots.get("verdict") == sha256(VERDICT), "verdict root")
    require(roots.get("input_manifest") == sha256(INPUT_MANIFEST), "input root")
    require(
        roots.get("execution_config") == input_manifest.get("execution_config_sha256"),
        "execution config root",
    )
    require(
        roots.get("execution_receipt") == input_manifest.get("execution_receipt_sha256"),
        "execution receipt root",
    )

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
            margins = np.asarray(cell["speaker_mean_margins"], dtype=np.float64)
            require(vector.shape == (54,) and margins.shape == (54,), "speaker arrays")
            require(np.isfinite(vector).all() and np.isfinite(margins).all(), "finite arrays")
            require(bool(((0.0 <= vector) & (vector <= 1.0)).all()), "follow range")
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

    verdict = authenticated_verdict(roots["verdict"]).decide(verdict_cells).as_dict()
    require(verdict == result.get("verdict"), "verdict")
    require(verdict["material_event_signal"] is True, "material verdict")
    require(
        verdict["manuscript_permission"]
        == "ABSTRACT_AND_CONCLUSION_UPGRADE_PERMITTED",
        "manuscript permission",
    )
    print("PASS — EXP-206 public result, intervals and verdict verify")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
