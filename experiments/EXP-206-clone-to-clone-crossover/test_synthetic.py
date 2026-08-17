"""Synthetic protocol and estimator tests for EXP-206."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


HERE = Path(__file__).resolve().parent


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ANALYZE = load_module("exp206_analyze_test", HERE / "analyze.py")
VERDICT = load_module("exp206_verdict_test", HERE / "verdict.py")


def unit(value: np.ndarray) -> np.ndarray:
    return value / np.linalg.norm(value)


def synthetic_embeddings(mode: str = "positive"):
    values = {}
    for system_index, system in enumerate(ANALYZE.SYSTEMS):
        for text in ANALYZE.TEXTS:
            nuisance = 0.001 * (system_index * 4 + text)
            for mic in ANALYZE.MICS:
                for arm in ANALYZE.ARMS:
                    sign = 1.0 if arm == "A" else -1.0
                    if mode == "inverted_mic2" and mic == "mic2":
                        sign *= -1.0
                    vector = unit(np.asarray([sign, nuisance], dtype=np.float64))
                    if mode == "ties":
                        vector = np.asarray([1.0, 0.0], dtype=np.float64)
                    values[("p001", system, text, mic, arm)] = {
                        "ecapa": vector,
                        "wavlm": vector.copy(),
                    }
    return values


def valid_manifest() -> dict:
    speakers = [f"p{index:03d}" for index in range(54)]
    rows = []
    for speaker in speakers:
        for system in ANALYZE.SYSTEMS:
            for text in ANALYZE.TEXTS:
                for mic in ANALYZE.MICS:
                    for arm in ANALYZE.ARMS:
                        relative = f"clones/{system}__{mic}__seed{arm}/{speaker}_t{text}.wav"
                        token = hashlib.sha256(relative.encode()).hexdigest()
                        rows.append(
                            {
                                "speaker": speaker,
                                "system": system,
                                "text_index": text,
                                "prompt_mic": mic,
                                "seed_arm": arm,
                                "clone_relative_path": relative,
                                "clone_sha256": token,
                                "ledger_relative_path": relative + ".ledger.json",
                                "ledger_sha256": token[::-1],
                                "cache_file": token[:24] + ".npz",
                                "cache_sha256": hashlib.sha256(token.encode()).hexdigest(),
                            }
                        )
    return {
        "schema": "exp206-input-manifest-v1",
        "status": "FROZEN_BEFORE_EXP206_ANALYSIS",
        "source_experiment": "EXP-205",
        "execution_config_sha256": "0" * 64,
        "execution_receipt_sha256": "1" * 64,
        "selection_manifest_sha256": "2" * 64,
        "exp205_scores_sha256_not_opened": "3" * 64,
        "readout_cache_pin": "4" * 64,
        "counts": ANALYZE.EXPECTED_COUNTS.copy(),
        "speakers": speakers,
        "clones": rows,
    }


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def contract_fixture(root: Path, manifest: dict) -> tuple[Path, Path]:
    run_root = root / "run"
    config_path = root / "execution-config.json"
    receipt_path = root / "receipt.json"
    generators = {system: {"pin": system} for system in ANALYZE.SYSTEMS}
    readouts = {readout: {"pin": readout} for readout in ANALYZE.READOUTS}
    source_pins = {"score": {"path": "score.py", "sha256": "5" * 64}}
    config = {
        "schema": "exp205-execution-config-v1",
        "run_root": str(run_root),
        "manifest": {"path": "selection.json", "sha256": "2" * 64},
        "expected_counts": {**ANALYZE.EXPECTED_COUNTS, "real_candidates": 216},
        "generators": generators,
        "readouts": readouts,
        "source_pins": source_pins,
    }
    # EXP-205 uses `texts`; EXP-206 adds only the derived comparison count.
    config["expected_counts"].pop("comparisons_per_speaker_per_direction")
    config_path.write_text(json.dumps(config), encoding="utf-8")
    manifest["execution_config_sha256"] = file_hash(config_path)
    manifest["selection_manifest_sha256"] = "2" * 64
    manifest["exp205_scores_sha256_not_opened"] = "3" * 64
    manifest["readout_cache_pin"] = hashlib.sha256(
        json.dumps(
            {"readouts": readouts, "score_source": source_pins["score"]},
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    receipt_rows = []
    for row in manifest["clones"]:
        receipt_rows.append(
            {
                "speaker": row["speaker"],
                "system": row["system"],
                "text_index": row["text_index"],
                "prompt_mic": row["prompt_mic"],
                "seed_arm": row["seed_arm"],
                "path": str(run_root / row["clone_relative_path"]),
                "sha256": row["clone_sha256"],
                "ledger_path": str(run_root / row["ledger_relative_path"]),
                "ledger_sha256": row["ledger_sha256"],
            }
        )
    receipt = {
        "schema": "exp205-execution-receipt-v1",
        "execution_config_sha256": file_hash(config_path),
        "manifest_sha256": "2" * 64,
        "scores_sha256": "3" * 64,
        "clone_count": 3456,
        "generator_pins": generators,
        "readout_pins": readouts,
        "source_pins": source_pins,
        "clones": receipt_rows,
    }
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    manifest["execution_receipt_sha256"] = file_hash(receipt_path)
    return config_path, receipt_path


class EstimatorTests(unittest.TestCase):
    def test_positive_grid_has_288_following_comparisons(self):
        result = ANALYZE.speaker_direction_values(
            synthetic_embeddings(), "p001", "mic1", "mic2"
        )
        self.assertEqual(result["ecapa"][0], 1.0)
        self.assertEqual(result["wavlm"][0], 1.0)

    def test_inverted_candidate_microphone_reverses_label(self):
        result = ANALYZE.speaker_direction_values(
            synthetic_embeddings("inverted_mic2"), "p001", "mic1", "mic2"
        )
        self.assertEqual(result["ecapa"][0], 0.0)
        self.assertEqual(result["wavlm"][0], 0.0)

    def test_exact_ties_score_half(self):
        result = ANALYZE.speaker_direction_values(
            synthetic_embeddings("ties"), "p001", "mic1", "mic2"
        )
        self.assertEqual(result["ecapa"][0], 0.5)
        self.assertEqual(result["wavlm"][0], 0.5)


class ManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = valid_manifest()

    def test_full_fixed_manifest_is_accepted(self):
        speakers, indexed = ANALYZE.validate_input_manifest(self.manifest)
        self.assertEqual(len(speakers), 54)
        self.assertEqual(len(indexed), 3456)

    def test_skeleton_manifest_is_rejected(self):
        with self.assertRaises(ANALYZE.InfrastructureError):
            ANALYZE.validate_input_manifest(
                {
                    "schema": "exp206-input-manifest-v1",
                    "status": "FROZEN_BEFORE_EXP206_ANALYSIS",
                }
            )

    def test_duplicate_identity_is_rejected(self):
        mutated = copy.deepcopy(self.manifest)
        mutated["clones"][1] = copy.deepcopy(mutated["clones"][0])
        with self.assertRaises(ANALYZE.InfrastructureError):
            ANALYZE.validate_input_manifest(mutated)

    def test_unfrozen_manifest_is_rejected(self):
        mutated = copy.deepcopy(self.manifest)
        mutated["status"] = "OUTCOME_BLIND_INPUT_FREEZE_CANDIDATE"
        with self.assertRaises(ANALYZE.InfrastructureError):
            ANALYZE.validate_input_manifest(mutated)

    def test_manifest_clone_substitution_against_receipt_is_rejected(self):
        manifest = copy.deepcopy(self.manifest)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path, receipt_path = contract_fixture(root, manifest)
            _, indexed = ANALYZE.validate_input_manifest(manifest)
            ANALYZE.authenticate_contract(
                manifest, config_path, receipt_path, indexed, root / "run"
            )
            manifest["clones"][0]["clone_sha256"] = "f" * 64
            _, substituted = ANALYZE.validate_input_manifest(manifest)
            with self.assertRaisesRegex(
                ANALYZE.InfrastructureError, "RECEIPT_CLONE_HASH_MISMATCH"
            ):
                ANALYZE.authenticate_contract(
                    manifest,
                    config_path,
                    receipt_path,
                    substituted,
                    root / "run",
                )


def cells(ecapa_point: float, ecapa_lcb: float, wavlm_point: float, wavlm_lcb: float):
    return {
        direction: {
            "ecapa": {"point": ecapa_point, "lcb": ecapa_lcb},
            "wavlm": {"point": wavlm_point, "lcb": wavlm_lcb},
        }
        for direction in VERDICT.DIRECTIONS
    }


class VerdictTests(unittest.TestCase):
    def test_material_state(self):
        result = VERDICT.decide(cells(0.65, 0.55, 0.58, 0.51))
        self.assertTrue(result.material_event_signal)
        self.assertEqual(
            result.manuscript_permission,
            "ABSTRACT_AND_CONCLUSION_UPGRADE_PERMITTED",
        )

    def test_positive_below_bar_has_limited_permission(self):
        result = VERDICT.decide(cells(0.59, 0.53, 0.54, 0.51))
        self.assertTrue(result.event_signal_present)
        self.assertFalse(result.material_event_signal)
        self.assertEqual(result.manuscript_permission, "LIMITED_SUPPORT_ONLY")

    def test_one_failed_cell_prevents_confirmation(self):
        value = cells(0.65, 0.55, 0.58, 0.51)
        value[VERDICT.DIRECTIONS[1]]["wavlm"]["lcb"] = 0.50
        result = VERDICT.decide(value)
        self.assertFalse(result.event_signal_present)
        self.assertEqual(result.manuscript_permission, "NO_HEADLINE_UPGRADE")

    def test_invalid_interval_is_rejected(self):
        with self.assertRaises(ValueError):
            VERDICT.decide(cells(0.60, 0.61, 0.58, 0.51))


if __name__ == "__main__":
    unittest.main()
