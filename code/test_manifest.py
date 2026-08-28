"""Verify the four Seed-VC content sources against the fixed text contract."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest


HERE = Path(__file__).resolve().parent


@unittest.skip(
    "Reads LibriTTS-R source transcripts, which this release does not redistribute. "
    "The mapping it checks is recorded in data/trials.json and data/generation_ledger.json."
)
class SeedVCTextMapping(unittest.TestCase):
    def test_all_four_sources_match_the_same_indexed_generated_text(self) -> None:
        manifest = json.loads(
            (HERE.parent / "data" / "selection_manifest.json").read_text(encoding="utf-8")
        )
        sources = manifest["generation"]["seedvc_sources"]
        texts = manifest["generation"]["generated_texts"]
        self.assertEqual([row["index"] for row in sources], list(range(4)))
        self.assertEqual([row["index"] for row in texts], list(range(4)))
        for source, text in zip(sources, texts, strict=True):
            transcript_path = Path(source["transcript_path"])
            transcript = transcript_path.read_text(encoding="utf-8").strip()
            self.assertEqual(transcript, source["transcript"])
            self.assertEqual(transcript, text["text"])
            self.assertEqual(
                hashlib.sha256(transcript_path.read_bytes()).hexdigest(),
                source["transcript_sha256"],
            )
            self.assertEqual(
                hashlib.sha256(transcript.encode("utf-8")).hexdigest(),
                text["sha256_utf8"],
            )


if __name__ == "__main__":
    unittest.main()
