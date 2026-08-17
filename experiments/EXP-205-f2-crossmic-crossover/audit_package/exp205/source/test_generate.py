"""Outcome-blind ancestry/resume tests for EXP-205 clone generation."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
import wave


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("exp205_generate", HERE / "generate.py")
assert SPEC is not None and SPEC.loader is not None
GENERATE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = GENERATE
SPEC.loader.exec_module(GENERATE)
SEED_SPEC = importlib.util.spec_from_file_location(
    "exp205_generate_seedvc", HERE / "generate_seedvc.py"
)
assert SEED_SPEC is not None and SEED_SPEC.loader is not None
GENERATE_SEEDVC = importlib.util.module_from_spec(SEED_SPEC)
sys.modules[SEED_SPEC.name] = GENERATE_SEEDVC
SEED_SPEC.loader.exec_module(GENERATE_SEEDVC)
BUILD_SPEC = importlib.util.spec_from_file_location(
    "exp205_build_execution_config", HERE / "build_execution_config.py"
)
assert BUILD_SPEC is not None and BUILD_SPEC.loader is not None
BUILD_CONFIG = importlib.util.module_from_spec(BUILD_SPEC)
sys.modules[BUILD_SPEC.name] = BUILD_CONFIG
BUILD_SPEC.loader.exec_module(BUILD_CONFIG)
VERIFY_SPEC = importlib.util.spec_from_file_location(
    "exp205_verify_pins", HERE / "verify_pins.py"
)
assert VERIFY_SPEC is not None and VERIFY_SPEC.loader is not None
VERIFY_PINS = importlib.util.module_from_spec(VERIFY_SPEC)
sys.modules[VERIFY_SPEC.name] = VERIFY_PINS
VERIFY_SPEC.loader.exec_module(VERIFY_PINS)


def write_wav(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16_000)
        handle.writeframes(b"\x00\x00" * 16_000)


class AuthenticatedResume(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        reference = root / "reference.wav"
        output = root / "clone.wav"
        write_wav(reference)
        write_wav(output)
        self.job = {
            "speaker": "p001",
            "prompt_mic": "mic1",
            "arm": "A",
            "reference": reference,
            "reference_sha256": GENERATE.sha256(reference),
            "reference_text_sha256": "1" * 64,
            "text_index": 0,
            "generated_text_sha256": "2" * 64,
            "seed": 42,
            "out": output,
        }
        self.context = {
            "execution_config_sha256": "3" * 64,
            "manifest_sha256": "4" * 64,
            "generate_source_sha256": "5" * 64,
            "generator_pin_sha256": "6" * 64,
        }

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_valid_wav_without_ledger_is_not_resumable(self) -> None:
        self.assertFalse(GENERATE.resumable(self.job, "f5", self.context))

    def test_exact_ledger_is_resumable_and_tampering_is_not(self) -> None:
        GENERATE.seal_clone(self.job, "f5", self.context)
        self.assertTrue(GENERATE.resumable(self.job, "f5", self.context))
        sidecar = GENERATE.ledger_path(self.job["out"])
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
        payload["prompt_mic"] = "mic2"
        sidecar.write_text(json.dumps(payload), encoding="utf-8")
        self.assertFalse(GENERATE.resumable(self.job, "f5", self.context))

    def test_reference_hash_is_authenticated_before_generation(self) -> None:
        self.job["reference_sha256"] = "0" * 64
        with self.assertRaisesRegex(RuntimeError, "GENERATION_REFERENCE hash mismatch"):
            GENERATE.authenticate_reference(self.job, set())


class AuthenticatedSeedVCResume(unittest.TestCase):
    def test_orphan_seedvc_wav_is_rejected_then_exact_ledger_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            reference = root / "reference.wav"
            source = root / "content.wav"
            output = root / "clone.wav"
            for path in (reference, source, output):
                write_wav(path)
            job = {
                "speaker": "p001",
                "prompt_mic": "mic2",
                "arm": "B",
                "text_index": 2,
                "source": source,
                "source_sha256": GENERATE_SEEDVC.sha256(source),
                "source_transcript_sha256": "7" * 64,
                "generated_text_sha256": "8" * 64,
                "reference": reference,
                "reference_sha256": GENERATE_SEEDVC.sha256(reference),
                "seed": 44,
                "out": output,
            }
            context = {
                "execution_config_sha256": "3" * 64,
                "manifest_sha256": "4" * 64,
                "generate_source_sha256": "5" * 64,
                "generator_pin_sha256": "6" * 64,
            }
            self.assertFalse(GENERATE_SEEDVC.resumable(job, context))
            GENERATE_SEEDVC.seal(job, context)
            self.assertTrue(GENERATE_SEEDVC.resumable(job, context))


class LogicalSnapshotPins(unittest.TestCase):
    def test_symlink_retarget_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            good = root / "good.bin"
            bad = root / "bad.bin"
            logical = root / "snapshot" / "model.bin"
            logical.parent.mkdir()
            good.write_bytes(b"good-model")
            bad.write_bytes(b"bad-model")
            logical.symlink_to(good)
            record = BUILD_CONFIG.pin(logical)
            self.assertEqual(record["path"], str(logical.absolute()))
            VERIFY_PINS.verify(record, "SYNTHETIC_MODEL")
            logical.unlink()
            logical.symlink_to(bad)
            with self.assertRaisesRegex(RuntimeError, "resolved target mismatch"):
                VERIFY_PINS.verify(record, "SYNTHETIC_MODEL")

    def test_f5_loader_consumes_the_pinned_logical_safetensors_path(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            blob = root / "blob-without-suffix"
            logical = root / "snapshot" / "model.safetensors"
            logical.parent.mkdir()
            blob.write_bytes(b"model")
            logical.symlink_to(blob)
            record = BUILD_CONFIG.pin(logical)
            pins = {"checkpoint_path": record["path"], "files": [record]}
            observed = GENERATE.require_pinned_loader_file(
                pins, "checkpoint_path", "F5_CHECKPOINT", suffix=".safetensors"
            )
            self.assertEqual(observed, logical.absolute())
            pins["checkpoint_path"] = record["resolved_path"]
            with self.assertRaisesRegex(RuntimeError, "not the unique pinned logical"):
                GENERATE.require_pinned_loader_file(
                    pins, "checkpoint_path", "F5_CHECKPOINT", suffix=".safetensors"
                )


if __name__ == "__main__":
    unittest.main()
