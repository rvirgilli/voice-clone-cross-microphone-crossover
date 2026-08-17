"""Outcome-blind cache/authentication tests for EXP-205 scoring."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest
from unittest import mock

import numpy as np


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("exp205_score", HERE / "score.py")
assert SPEC is not None and SPEC.loader is not None
SCORE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SCORE
SPEC.loader.exec_module(SCORE)


def package(name: str) -> ModuleType:
    module = ModuleType(name)
    module.__path__ = []
    return module


class CacheCannotBypassLoaderAuthentication(unittest.TestCase):
    def test_wrong_loader_pin_fails_before_cache_is_read(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            audio = root / "audio.wav"
            audio.write_bytes(b"audio")
            model = root / "model.bin"
            model.write_bytes(b"model")
            source = root / "loader.py"
            source.write_text("# synthetic loader\n", encoding="utf-8")
            fake_modules = {
                "speechbrain": package("speechbrain"),
                "speechbrain.inference": package("speechbrain.inference"),
                "speechbrain.inference.speaker": ModuleType(
                    "speechbrain.inference.speaker"
                ),
                "transformers": package("transformers"),
                "transformers.models": package("transformers.models"),
                "transformers.models.wav2vec2": package(
                    "transformers.models.wav2vec2"
                ),
                "transformers.models.wav2vec2.feature_extraction_wav2vec2": ModuleType(
                    "transformers.models.wav2vec2.feature_extraction_wav2vec2"
                ),
                "transformers.models.wavlm": package("transformers.models.wavlm"),
                "transformers.models.wavlm.modeling_wavlm": ModuleType(
                    "transformers.models.wavlm.modeling_wavlm"
                ),
            }
            for name in (
                "speechbrain.inference.speaker",
                "transformers.models.wav2vec2.feature_extraction_wav2vec2",
                "transformers.models.wavlm.modeling_wavlm",
            ):
                fake_modules[name].__file__ = str(source)
            record = {
                "path": str(model),
                "sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
            }
            config = {
                "readouts": {
                    "ecapa": {
                        "files": [record],
                        "loader_sha256": {"ecapa_speaker": "0" * 64},
                    },
                    "wavlm": {
                        "files": [record],
                        "loader_sha256": {
                            "wavlm_model": "0" * 64,
                            "wavlm_feature_extractor": "0" * 64,
                        },
                    },
                },
                "source_pins": {"score": {"path": str(HERE / "score.py"), "sha256": "x"}},
            }
            with mock.patch.dict(sys.modules, fake_modules), mock.patch.object(
                SCORE,
                "load_cached",
                return_value=(np.ones(2, dtype=np.float32), np.ones(2, dtype=np.float32)),
            ) as cached:
                with self.assertRaisesRegex(RuntimeError, "readout loader mismatch"):
                    SCORE.extract_embeddings(
                        [audio],
                        {str(audio): hashlib.sha256(audio.read_bytes()).hexdigest()},
                        root / "cache",
                        config,
                    )
                cached.assert_not_called()


if __name__ == "__main__":
    unittest.main()
