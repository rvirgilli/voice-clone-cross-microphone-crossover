#!/usr/bin/env python3
"""Offline checks for the release's pinned licensing evidence and boundaries."""

from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
LICENCES = ROOT / "licenses"


class LicenceSnapshotsTest(unittest.TestCase):
    def test_manifest_is_complete_and_authentic(self):
        manifest = json.loads((LICENCES / "SNAPSHOT-MANIFEST.json").read_text())
        self.assertEqual(manifest["schema"], "f2-license-snapshots-v1")
        records = manifest["files"]
        self.assertEqual(len(records), 8)
        expected = {record["path"] for record in records} | {"SNAPSHOT-MANIFEST.json"}
        self.assertEqual({path.name for path in LICENCES.iterdir() if path.is_file()}, expected)
        for record in records:
            observed = hashlib.sha256((LICENCES / record["path"]).read_bytes()).hexdigest()
            self.assertEqual(observed, record["sha256"], record["path"])

    def test_model_cards_name_the_snapshotted_terms(self):
        self.assertIn("license: cc-by-nc-4.0", (LICENCES / "f5-model-card-84e5a410.md").read_text())
        self.assertIn("license: \"apache-2.0\"", (LICENCES / "ecapa-model-card-0f99f2d0.md").read_text())
        self.assertIn("Coqui Public Model License", (LICENCES / "xtts-v2-model-card-6c2b0d75.md").read_text())
        self.assertIn("github.com/microsoft/UniSpeech/blob/main/LICENSE", (LICENCES / "wavlm-model-card-feb593a6.md").read_text())

    def test_xtts_terms_explicitly_cover_outputs(self):
        cpml = (LICENCES / "xtts-v2-cpml-1.0.0.txt").read_text()
        self.assertIn("only non-commercial use of a machine learning model and its outputs", cpml)
        self.assertIn("anyone who gets a copy", cpml)
        self.assertIn("their output", cpml)

    def test_release_contains_no_audio_or_model_payloads(self):
        forbidden = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aac", ".opus",
                     ".pt", ".pth", ".ckpt", ".safetensors", ".bin"}
        ignored = {".git", ".venv", ".pytest_cache", "__pycache__", "regenerated"}
        offenders = [
            path for path in ROOT.rglob("*")
            if path.is_file()
            and not any(part in ignored for part in path.relative_to(ROOT).parts)
            and path.suffix.lower() in forbidden
        ]
        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
