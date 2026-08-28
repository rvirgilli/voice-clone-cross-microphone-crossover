"""Synthetic full-census and mutation tests for the analyzer."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("exp205_analyze", HERE / "analyze.py")
assert SPEC is not None and SPEC.loader is not None
ANALYZE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ANALYZE
SPEC.loader.exec_module(ANALYZE)
ANALYZE.BOOTSTRAPS = 1_000


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class SyntheticPackage:
    def __init__(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.manifest_path = self.root / "manifest.json"
        self.scores_path = self.root / "scores.tsv"
        self.receipt_path = self.root / "receipt.json"
        self.execution_config_path = self.root / "execution-config.json"
        self.config_path = self.root / "config.json"
        self.output_path = self.root / "result.json"
        self.speakers = [f"p{i:03d}" for i in range(54)]
        self.manifest = self._build_manifest()
        self.rows, self.clones = self._build_rows()
        self.write_all()

    def close(self) -> None:
        self.temp.cleanup()

    def _write_file(self, relative: str, payload: bytes) -> tuple[str, str]:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        return str(path), file_hash(path)

    def _build_manifest(self) -> dict:
        speaker_rows = []
        for speaker in self.speakers:
            audio = {}
            for key in ("A_mic1", "A_mic2", "B_mic1", "B_mic2"):
                path, digest = self._write_file(
                    f"candidates/{speaker}_{key}.wav", f"{speaker}:{key}".encode()
                )
                audio[key] = {
                    "path": path,
                    "sha256": digest,
                    "frames": 80_000,
                    "samplerate_hz": 16_000,
                    "duration_s": 5.0,
                }
            speaker_rows.append({"speaker": speaker, "audio": audio})
        return {
            "schema": "exp205-selection-manifest-v1",
            "counts": {
                "speakers": 54,
                "seed_arms": 2,
                "prompt_microphones": 2,
                "generated_texts": 4,
                "systems": 4,
                "expected_clones": 3456,
                "comparisons_per_speaker_per_direction": 32,
            },
            "speakers": speaker_rows,
        }

    def _build_rows(self) -> tuple[list[dict[str, str]], list[dict[str, str | int]]]:
        by_speaker = {row["speaker"]: row for row in self.manifest["speakers"]}
        rows = []
        clones = []
        for speaker in self.speakers:
            for system in ANALYZE.SYSTEMS:
                for text_index in ANALYZE.TEXT_INDICES:
                    for prompt_mic in ANALYZE.PROMPT_MICS:
                        candidate_mic = "mic2" if prompt_mic == "mic1" else "mic1"
                        for arm in ANALYZE.ARMS:
                            relative = (
                                f"clones/{speaker}_{system}_{text_index}_{prompt_mic}_{arm}.wav"
                            )
                            clone_path, clone_hash = self._write_file(
                                relative, relative.encode()
                            )
                            candidate_a = by_speaker[speaker]["audio"][f"A_{candidate_mic}"]
                            candidate_b = by_speaker[speaker]["audio"][f"B_{candidate_mic}"]
                            own_a = arm == "A"
                            row = {
                                "speaker": speaker,
                                "system": system,
                                "text_index": str(text_index),
                                "prompt_mic": prompt_mic,
                                "seed_arm": arm,
                                "clone_path": clone_path,
                                "clone_sha256": clone_hash,
                                "candidate_A_path": candidate_a["path"],
                                "candidate_A_sha256": candidate_a["sha256"],
                                "candidate_B_path": candidate_b["path"],
                                "candidate_B_sha256": candidate_b["sha256"],
                                "same_candidate_A_path": by_speaker[speaker]["audio"][f"A_{prompt_mic}"]["path"],
                                "same_candidate_A_sha256": by_speaker[speaker]["audio"][f"A_{prompt_mic}"]["sha256"],
                                "same_candidate_B_path": by_speaker[speaker]["audio"][f"B_{prompt_mic}"]["path"],
                                "same_candidate_B_sha256": by_speaker[speaker]["audio"][f"B_{prompt_mic}"]["sha256"],
                                "ecapa_A": "0.9" if own_a else "0.1",
                                "ecapa_B": "0.1" if own_a else "0.9",
                                "wavlm_A": "0.8" if own_a else "0.2",
                                "wavlm_B": "0.2" if own_a else "0.8",
                                "ecapa_same_A": "0.9" if own_a else "0.1",
                                "ecapa_same_B": "0.1" if own_a else "0.9",
                                "wavlm_same_A": "0.8" if own_a else "0.2",
                                "wavlm_same_B": "0.2" if own_a else "0.8",
                            }
                            rows.append(row)
                            clones.append(
                                {
                                    "speaker": speaker,
                                    "system": system,
                                    "text_index": text_index,
                                    "prompt_mic": prompt_mic,
                                    "seed_arm": arm,
                                    "path": clone_path,
                                    "sha256": clone_hash,
                                }
                            )
        return rows, clones

    def write_scores(self) -> None:
        with self.scores_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=ANALYZE.EXPECTED_HEADER, delimiter="\t"
            )
            writer.writeheader()
            writer.writerows(self.rows)

    def write_all(self) -> None:
        self.manifest_path.write_text(json.dumps(self.manifest), encoding="utf-8")
        self.write_scores()
        for clone in self.clones:
            ledger_path = Path(clone["path"] + ".ledger.json")
            ledger = {
                "schema": "exp205-clone-ledger-v1",
                "speaker": clone["speaker"],
                "system": clone["system"],
                "text_index": clone["text_index"],
                "prompt_mic": clone["prompt_mic"],
                "seed_arm": clone["seed_arm"],
                "output_path": clone["path"],
                "clone_sha256": clone["sha256"],
            }
            ledger_path.write_text(json.dumps(ledger), encoding="utf-8")
            clone["ledger_path"] = str(ledger_path)
            clone["ledger_sha256"] = file_hash(ledger_path)
        expected_counts = {
            "speakers": 54,
            "systems": 4,
            "prompt_microphones": 2,
            "seed_arms": 2,
            "texts": 4,
            "clones": 3456,
            "real_candidates": 216,
        }
        source_pins = {
            "analyze": {
                "path": str(HERE / "analyze.py"),
                "sha256": file_hash(HERE / "analyze.py"),
            },
            "verdict": {
                "path": str(HERE / "verdict.py"),
                "sha256": file_hash(HERE / "verdict.py"),
            },
        }
        generators = {name: {"test_pin": name} for name in ANALYZE.SYSTEMS}
        readouts = {"ecapa": {"test_pin": "e"}, "wavlm": {"test_pin": "w"}}
        execution_config = {
            "schema": "exp205-execution-config-v1",
            "manifest": {
                "path": str(self.manifest_path),
                "sha256": file_hash(self.manifest_path),
            },
            "execution_receipt_path": str(self.receipt_path),
            "scores_path": str(self.scores_path),
            "scientific_result_path": str(self.output_path),
            "expected_counts": expected_counts,
            "generators": generators,
            "readouts": readouts,
            "source_pins": source_pins,
        }
        self.execution_config_path.write_text(
            json.dumps(execution_config), encoding="utf-8"
        )
        self.expected_execution_config_sha256 = file_hash(
            self.execution_config_path
        )
        receipt = {
            "schema": "exp205-execution-receipt-v1",
            "manifest_sha256": file_hash(self.manifest_path),
            "scores_sha256": file_hash(self.scores_path),
            "execution_config_sha256": self.expected_execution_config_sha256,
            "clone_count": 3456,
            "candidate_count": 216,
            "expected_counts": expected_counts,
            "generator_pins": generators,
            "readout_pins": readouts,
            "source_pins": source_pins,
            "clones": self.clones,
        }
        self.receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
        config = {
            "execution_config": {
                "path": str(self.execution_config_path),
                "sha256": self.expected_execution_config_sha256,
            },
            "manifest": {
                "path": str(self.manifest_path),
                "sha256": file_hash(self.manifest_path),
            },
            "execution_receipt": {
                "path": str(self.receipt_path),
                "sha256": file_hash(self.receipt_path),
            },
            "scores": {
                "path": str(self.scores_path),
                "sha256": file_hash(self.scores_path),
            },
            "verdict": {
                "path": str(HERE / "verdict.py"),
                "sha256": file_hash(HERE / "verdict.py"),
            },
            "output": str(self.output_path),
        }
        self.config_path.write_text(json.dumps(config), encoding="utf-8")

    def resign_scores_and_receipt(self) -> None:
        self.write_scores()
        receipt = json.loads(self.receipt_path.read_text(encoding="utf-8"))
        receipt["scores_sha256"] = file_hash(self.scores_path)
        self.receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
        config = json.loads(self.config_path.read_text(encoding="utf-8"))
        config["scores"]["sha256"] = file_hash(self.scores_path)
        config["execution_receipt"]["sha256"] = file_hash(self.receipt_path)
        self.config_path.write_text(json.dumps(config), encoding="utf-8")

    def resign_receipt(self) -> None:
        config = json.loads(self.config_path.read_text(encoding="utf-8"))
        config["execution_receipt"]["sha256"] = file_hash(self.receipt_path)
        self.config_path.write_text(json.dumps(config), encoding="utf-8")

    def set_primary_pattern(self, encoder: str, mode: str) -> None:
        if mode not in {"strong", "weak", "chance"}:
            raise ValueError(mode)
        for row in self.rows:
            if row["prompt_mic"] != "mic1":
                continue
            correct = mode == "strong" or (
                mode == "weak" and row["system"] in {"f5", "xtts"}
            )
            arm = row["seed_arm"]
            other = "B" if arm == "A" else "A"
            if correct:
                row[f"{encoder}_{arm}"] = "0.9"
                row[f"{encoder}_{other}"] = "0.1"
            else:
                row[f"{encoder}_A"] = "0.5"
                row[f"{encoder}_B"] = "0.5"


class AnalyzerIntegration(unittest.TestCase):
    def setUp(self) -> None:
        self.package = SyntheticPackage()

    def tearDown(self) -> None:
        self.package.close()

    def assert_infrastructure_failure(self) -> None:
        self.assertEqual(
            ANALYZE.run(
                self.package.config_path,
                self.package.expected_execution_config_sha256,
            ),
            2,
        )
        result = json.loads(self.package.output_path.read_text(encoding="utf-8"))
        self.assertEqual(result["status"], "INFRASTRUCTURE_FAILURE")
        self.assertNotIn("directions", result)

    def test_exact_census_operational_bidirectional(self) -> None:
        self.assertEqual(
            ANALYZE.run(
                self.package.config_path,
                self.package.expected_execution_config_sha256,
            ),
            0,
        )
        result = json.loads(self.package.output_path.read_text(encoding="utf-8"))
        self.assertEqual(result["counts"]["score_rows"], 3456)
        self.assertEqual(
            result["primary"]["verdict"], "OPERATIONAL_CROSSMIC_CONFIRMATION"
        )
        self.assertEqual(
            result["reverse"]["verdict"], "BIDIRECTIONAL_REPLICATION"
        )
        self.assertEqual(
            result["headline_permission"], "BIDIRECTIONAL_HEADLINE_PERMITTED"
        )

    def test_all_six_reachable_primary_branches_through_analyzer(self) -> None:
        cases = (
            ("strong", "strong", "OPERATIONAL_CROSSMIC_CONFIRMATION"),
            ("strong", "chance", "PRIMARY_ONLY"),
            ("weak", "strong", "REPLICATED_WEAK_POSITIVE"),
            ("weak", "chance", "WEAK_PRIMARY_ONLY"),
            ("chance", "strong", "SECONDARY_ONLY_NOT_CONFIRMED"),
            ("chance", "chance", "NOT_CONFIRMED"),
        )
        for ecapa, wavlm, expected in cases:
            with self.subTest(expected=expected):
                self.package.set_primary_pattern("ecapa", ecapa)
                self.package.set_primary_pattern("wavlm", wavlm)
                self.package.resign_scores_and_receipt()
                self.assertEqual(
                    ANALYZE.run(
                        self.package.config_path,
                        self.package.expected_execution_config_sha256,
                    ),
                    0,
                )
                result = json.loads(
                    self.package.output_path.read_text(encoding="utf-8")
                )
                self.assertEqual(result["primary"]["verdict"], expected)

    def test_missing_row_is_not_partial_science(self) -> None:
        self.package.rows.pop()
        self.package.resign_scores_and_receipt()
        self.assert_infrastructure_failure()

    def test_duplicate_row_is_not_partial_science(self) -> None:
        self.package.rows.append(dict(self.package.rows[0]))
        self.package.resign_scores_and_receipt()
        self.assert_infrastructure_failure()

    def test_wrong_candidate_is_not_partial_science(self) -> None:
        self.package.rows[0]["candidate_A_path"] = self.package.rows[0][
            "candidate_B_path"
        ]
        self.package.resign_scores_and_receipt()
        self.assert_infrastructure_failure()

    def test_wrong_speaker_is_not_partial_science(self) -> None:
        self.package.rows[0]["speaker"] = "p999"
        self.package.resign_scores_and_receipt()
        self.assert_infrastructure_failure()

    def test_wrong_microphone_is_not_partial_science(self) -> None:
        self.package.rows[0]["prompt_mic"] = "mic3"
        self.package.resign_scores_and_receipt()
        self.assert_infrastructure_failure()

    def test_swapped_receipt_clone_is_not_partial_science(self) -> None:
        receipt = json.loads(self.package.receipt_path.read_text(encoding="utf-8"))
        receipt["clones"][0]["path"] = receipt["clones"][1]["path"]
        self.package.receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
        self.package.resign_receipt()
        self.assert_infrastructure_failure()

    def test_tampered_clone_is_not_partial_science(self) -> None:
        Path(self.package.clones[0]["path"]).write_bytes(b"tampered")
        self.assert_infrastructure_failure()

    def test_resigned_wrong_clone_ledger_is_not_partial_science(self) -> None:
        ledger_path = Path(self.package.clones[0]["ledger_path"])
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        ledger["prompt_mic"] = "mic2" if ledger["prompt_mic"] == "mic1" else "mic1"
        ledger_path.write_text(json.dumps(ledger), encoding="utf-8")
        receipt = json.loads(self.package.receipt_path.read_text(encoding="utf-8"))
        receipt["clones"][0]["ledger_sha256"] = file_hash(ledger_path)
        self.package.receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
        self.package.resign_receipt()
        self.assert_infrastructure_failure()

    def test_wrong_verdict_pin_is_not_partial_science(self) -> None:
        config = json.loads(self.package.config_path.read_text(encoding="utf-8"))
        config["verdict"]["sha256"] = "0" * 64
        self.package.config_path.write_text(json.dumps(config), encoding="utf-8")
        self.assert_infrastructure_failure()

    def test_receipt_cannot_reassert_mutated_execution_contract(self) -> None:
        for field, value in (
            ("execution_config_sha256", "0" * 64),
            ("clone_count", 3455),
            ("candidate_count", 215),
            ("expected_counts", {}),
            ("generator_pins", {}),
            ("readout_pins", {}),
            ("source_pins", {}),
        ):
            with self.subTest(field=field):
                original = self.package.receipt_path.read_text(encoding="utf-8")
                receipt = json.loads(original)
                receipt[field] = value
                self.package.receipt_path.write_text(
                    json.dumps(receipt), encoding="utf-8"
                )
                self.package.resign_receipt()
                self.assert_infrastructure_failure()
                self.package.receipt_path.write_text(original, encoding="utf-8")
                self.package.resign_receipt()


if __name__ == "__main__":
    unittest.main()
