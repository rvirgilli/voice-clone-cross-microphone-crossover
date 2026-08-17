#!/usr/bin/env python3
"""Create a path-portable copy of the historical EXP-205 feasibility census."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace_prefix(value, prefix: str):
    if isinstance(value, str):
        return value.replace(prefix, "inputs/")
    if isinstance(value, list):
        return [replace_prefix(item, prefix) for item in value]
    if isinstance(value, dict):
        return {key: replace_prefix(item, prefix) for key, item in value.items()}
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--strip-prefix", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source_hash = sha256(args.input)
    source = json.loads(args.input.read_text(encoding="utf-8"))
    portable = replace_prefix(source, args.strip_prefix)
    portable["publication_portability"] = {
        "status": "path-prefix-normalization-after-result",
        "historical_source_sha256": source_hash,
        "scientific_fields_changed": False,
        "transform": "one absolute run-root prefix replaced by inputs/",
    }
    rendered = json.dumps(portable, indent=2, sort_keys=True) + "\n"
    if any(root in rendered for root in ("/home/", "/mnt/", "/media/")):
        raise SystemExit("portable output still contains a machine root")
    args.output.write_text(rendered, encoding="utf-8")
    print(f"wrote {args.output}: historical source {source_hash}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
