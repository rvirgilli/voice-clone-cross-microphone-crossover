#!/usr/bin/env python3
"""Refresh public package hashes while preserving the sealed trust roots."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
AUDIT = HERE / "audit.json"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> int:
    previous = json.loads(AUDIT.read_text(encoding="utf-8"))
    records = {}
    for path in sorted(HERE.rglob("*")):
        if not path.is_file() or path == AUDIT or "__pycache__" in path.parts:
            continue
        relative = path.relative_to(HERE).as_posix()
        records[relative] = {"sha256": digest(path), "size": path.stat().st_size}

    source_records = previous["source_records"]
    for record in source_records.values():
        record["sha256"] = digest(HERE / record["path"])

    output = {
        "artifacts": records,
        "counts": previous["counts"],
        "original_trust_roots": previous["original_trust_roots"],
        "schema": previous["schema"],
        "scope": previous["scope"],
        "source_records": source_records,
    }
    AUDIT.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {AUDIT.name}: {len(records)} artifacts, sealed trust roots preserved")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
