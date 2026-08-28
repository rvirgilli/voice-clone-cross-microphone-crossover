#!/usr/bin/env python3
"""Rebuild the complete release checksum inventory.

``data/checksums.sha256`` is the explicit root of trust and therefore cannot hash
itself. Every other payload file is included. Runtime caches and reproducible build
intermediates are excluded; audio, model weights, archives and symlinks fail closed.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "checksums.sha256"
IGNORED_DIRS = {".git", ".venv", ".pytest_cache", "__pycache__", "regenerated",
                "generated-audio", "model-snapshots"}
IGNORED_SUFFIXES = {".pyc", ".aux", ".bbl", ".blg", ".fdb_latexmk", ".fls", ".log"}
FORBIDDEN_RELEASE_SUFFIXES = {
    ".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aac", ".opus",
    ".pt", ".pth", ".ckpt", ".safetensors", ".bin", ".zip", ".tar", ".gz", ".7z",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def release_files() -> list[Path]:
    files = []
    for path in ROOT.rglob("*"):
        rel = path.relative_to(ROOT)
        if any(part in IGNORED_DIRS for part in rel.parts):
            continue
        if path.is_symlink():
            raise RuntimeError(f"symlink forbidden in release: {rel}")
        if not path.is_file() or path == OUT or path.suffix.lower() in IGNORED_SUFFIXES:
            continue
        if path.suffix.lower() in FORBIDDEN_RELEASE_SUFFIXES:
            raise RuntimeError(f"audio/model/archive forbidden in release: {rel}")
        files.append(path)
    return sorted(files, key=lambda path: path.relative_to(ROOT).as_posix())


def main() -> int:
    files = release_files()
    payload = "".join(
        f"{sha256(path)}  {path.relative_to(ROOT).as_posix()}\n" for path in files
    )
    fd, tmp_name = tempfile.mkstemp(prefix=".checksums.", suffix=".tmp", dir=OUT.parent)
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, OUT)
    finally:
        tmp.unlink(missing_ok=True)
    print(f"PASS — recorded {len(files)} release payload files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
