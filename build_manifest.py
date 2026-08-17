#!/usr/bin/env python3
"""Build the deterministic SHA-256 release manifest."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "MANIFEST.json"
IGNORED_PARTS = {".git", ".venv", "__pycache__"}
IGNORED_SUFFIXES = {".pyc", ".pyo", ".aux", ".bbl", ".blg", ".fdb_latexmk", ".fls", ".log"}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


records = {}
for path in sorted(ROOT.rglob("*")):
    if not path.is_file() or path == OUT:
        continue
    relative = path.relative_to(ROOT)
    if any(part in IGNORED_PARTS for part in relative.parts) or path.suffix in IGNORED_SUFFIXES:
        continue
    records[relative.as_posix()] = {"sha256": digest(path), "bytes": path.stat().st_size}
OUT.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(f"wrote {OUT.name}: {len(records)} files")
