"""Synthetic failure injections for the load-bearing channel-distinctness probe."""

from __future__ import annotations

import copy
import contextlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import soundfile as sf


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("channel_distinctness", HERE / "channel_distinctness.py")
channel = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(channel)


class ChannelDistinctnessMutations(unittest.TestCase):
    def test_gain_and_every_lag_in_synthetic_window_score_as_duplicates(self) -> None:
        rng = np.random.default_rng(205)
        x = rng.normal(size=257)
        sr = 400  # 50 ms = 20 samples, small enough to exhaust every integer lag
        limit = int(sr * channel.LAG_MS / 1000.0)
        for lag in range(-limit, limit + 1):
            with self.subTest(lag=lag):
                y = 0.4 * channel.shifted_copy(x, lag)
                _, residual, recovered = channel.best_alignment(x, y, sr)
                self.assertLessEqual(residual, 1e-7)
                self.assertEqual(recovered, lag)

    def test_real_capture_controls_include_both_window_boundaries(self) -> None:
        x = np.random.default_rng(206).normal(size=2000)
        controls, shifts = channel.injection_controls(x, 4000)
        self.assertEqual(shifts, [-200, -120, -1, 1, 120, 200])
        self.assertIn("shifted_copy_+200", controls)
        self.assertIn("shifted_copy_-200", controls)
        self.assertLessEqual(max(controls.values()), channel.DUPLICATE_BAR)

    def test_each_assertion_failure_mode_can_fail(self) -> None:
        base = {
            "n_event_captures": 108,
            "byte_identical_pairs": 0,
            "residual_after_alignment": {"min": 0.21, "closest_pair": "p274/A"},
            "injection_controls": {"n_source_captures": 108, "max_residual": 0.0},
        }
        mutations = (("n_event_captures", 107), ("byte_identical_pairs", 1))
        for key, value in mutations:
            changed = copy.deepcopy(base)
            changed[key] = value
            self.assertTrue(channel.assertions(changed), key)
        changed = copy.deepcopy(base)
        changed["residual_after_alignment"]["min"] = channel.DUPLICATE_BAR
        self.assertTrue(channel.assertions(changed))
        changed = copy.deepcopy(base)
        changed["injection_controls"]["n_source_captures"] = 107
        self.assertTrue(channel.assertions(changed))
        changed = copy.deepcopy(base)
        changed["injection_controls"]["max_residual"] = 2 * channel.DUPLICATE_BAR
        self.assertTrue(channel.assertions(changed))

    def test_manifest_hash_mismatch_fails(self) -> None:
        with tempfile.TemporaryDirectory(prefix="f2-channel-hash-") as raw:
            root = Path(raw)
            a, b = root / "a.wav", root / "b.wav"
            sf.write(a, np.arange(64, dtype=np.float64) / 64, 16000)
            sf.write(b, np.arange(64, dtype=np.float64) / 64, 16000)
            bad = "0" * 64
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                channel.read_pair({"path": str(a), "sha256": bad},
                                  {"path": str(b), "sha256": bad})

    def test_failed_census_removes_stale_output(self) -> None:
        with tempfile.TemporaryDirectory(prefix="f2-channel-stale-") as raw:
            root = Path(raw)
            manifest = root / "selection-manifest.json"
            output = root / "channel_distinctness.json"
            output.write_text('{"looks": "current"}\n', encoding="utf-8")
            speakers = [{"speaker": f"p{i:03d}", "audio": {}} for i in range(54)]
            manifest.write_text(json.dumps({"speakers": speakers}), encoding="utf-8")
            with mock.patch.object(channel, "MANIFEST", manifest), mock.patch.object(channel, "OUT", output):
                with contextlib.redirect_stderr(io.StringIO()):
                    self.assertEqual(channel.main(), 1)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
