#!/usr/bin/env python3
"""Authenticate publication-wording amendments against repository history."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
LEDGER = HERE / "PUBLICATION-AMENDMENTS.json"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def git_blob(commit: str, path: Path) -> bytes:
    relative = path.relative_to(ROOT).as_posix()
    return subprocess.check_output(["git", "show", f"{commit}:{relative}"], cwd=ROOT)


def main() -> int:
    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    if ledger.get("schema_version") != 1 or ledger.get("status") != "publication-wording-only-after-result":
        raise AssertionError("publication amendment ledger schema/status differs")
    if ledger.get("external_timestamp_claimed") is not False:
        raise AssertionError("ledger must not claim an external timestamp")
    commit = ledger["historical_public_commit"]
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "HEAD"], cwd=ROOT, check=False
    )
    if ancestor.returncode:
        raise AssertionError("historical public commit is not an ancestor of HEAD")
    for name, pins in ledger["files"].items():
        path = HERE / name
        if sha256_bytes(git_blob(commit, path)) != pins["original_sha256"]:
            raise AssertionError(f"historical bytes differ: {name}")
        if sha256_bytes(path.read_bytes()) != pins["publication_sha256"]:
            raise AssertionError(f"publication bytes differ: {name}")
    print("PASS — historical protocol bytes and publication-only wording amendments verify")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
