#!/usr/bin/env python3
"""Verify the public release without consulting private project state."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = ROOT / "MANIFEST.json"
IGNORED_PARTS = {".git", ".venv", "__pycache__"}
IGNORED_SUFFIXES = {".pyc", ".pyo", ".aux", ".bbl", ".blg", ".fdb_latexmk", ".fls", ".log"}


def released_files() -> set[str]:
    output = set()
    for path in ROOT.rglob("*"):
        if not path.is_file() or path == MANIFEST_PATH:
            continue
        relative = path.relative_to(ROOT)
        if any(part in IGNORED_PARTS for part in relative.parts) or path.suffix in IGNORED_SUFFIXES:
            continue
        output.add(relative.as_posix())
    return output


def main() -> int:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    failures: list[str] = []
    for name, record in manifest.items():
        path = ROOT / name
        if not path.is_file():
            failures.append(f"missing manifest file: {name}")
            continue
        observed = hashlib.sha256(path.read_bytes()).hexdigest()
        if observed != record["sha256"] or path.stat().st_size != record["bytes"]:
            failures.append(f"manifest mismatch: {name}")

    actual = released_files()
    if actual != set(manifest):
        failures.extend(f"unbound release file: {name}" for name in sorted(actual - set(manifest)))
        failures.extend(f"manifest-only file: {name}" for name in sorted(set(manifest) - actual))

    machine_roots = ("/" + "home" + "/" + "rv", "~" + "/projects")
    for name in sorted(actual):
        path = ROOT / name
        if path.suffix.lower() in {".pdf", ".png", ".jpg", ".pyc"}:
            continue
        value = path.read_text(encoding="utf-8", errors="ignore")
        if any(token in value for token in machine_roots):
            failures.append(f"machine-specific path: {name}")
        if "Anonymous" + " ICASSP" in value:
            failures.append(f"anonymous author placeholder: {name}")

    commands = (
        [sys.executable, "experiments/EXP-205-f2-crossmic-crossover/audit_package/exp205/verify_exp205_package.py"],
        [sys.executable, "experiments/EXP-205-f2-crossmic-crossover/audit_package/exp205/source/selection/verify_selection_provenance.py"],
        [sys.executable, "experiments/EXP-205-f2-crossmic-crossover/audit_package/exp205/verify_public_history.py"],
        [sys.executable, "experiments/EXP-206-clone-to-clone-crossover/verify_result.py"],
        [sys.executable, "paper/F2/check_numbers.py"],
        [sys.executable, "paper/F2/mutation_test.py"],
    )
    for command in commands:
        result = subprocess.run(command, cwd=ROOT, check=False)
        if result.returncode:
            failures.append(f"command failed: {' '.join(command)}")

    if failures:
        print("FAILED — release repository")
        print("\n".join(f"  - {item}" for item in failures))
        return 1
    print("PASS — release repository verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
