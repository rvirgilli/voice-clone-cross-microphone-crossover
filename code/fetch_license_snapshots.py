#!/usr/bin/env python3
"""Fetch and authenticate the exact model-card/licence evidence for this release."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "licenses"
CHECKED = "2026-08-28"
SNAPSHOTS = (
    {
        "path": "f5-model-card-84e5a410.md",
        "component": "F5-TTS checkpoint",
        "kind": "model-card",
        "url": "https://huggingface.co/SWivid/F5-TTS/resolve/84e5a410d9cead4de2f847e7c9369a6440bdfaca/README.md",
        "sha256": "cf3354feaf4020527b3642b12a5a382de13d0af6bc99e8d37bee84425c1ca224",
    },
    {
        "path": "f5-cc-by-nc-4.0.txt",
        "component": "F5-TTS checkpoint",
        "kind": "licence-text",
        "url": "https://creativecommons.org/licenses/by-nc/4.0/legalcode.txt",
        "sha256": "41003d4a74749c0220e33dd415042164b5a1093ed401f36277234f772d22d3d0",
    },
    {
        "path": "xtts-v2-model-card-6c2b0d75.md",
        "component": "XTTS-v2 checkpoint",
        "kind": "model-card",
        "url": "https://huggingface.co/coqui/XTTS-v2/resolve/6c2b0d75eae4b7047358e3b6bd9325f857d43f77/README.md",
        "sha256": "1cfa85b3293f685b3a6537f8da3d94820fd111270e553589073885dea3facfb7",
    },
    {
        "path": "xtts-v2-cpml-1.0.0.txt",
        "component": "XTTS-v2 checkpoint and outputs",
        "kind": "licence-text",
        "url": "https://huggingface.co/coqui/XTTS-v2/resolve/6c2b0d75eae4b7047358e3b6bd9325f857d43f77/LICENSE.txt",
        "sha256": "190f6d7c19b8984f91b97712b94ce92d2b2e640fc677dacab966e955ece9d043",
    },
    {
        "path": "ecapa-model-card-0f99f2d0.md",
        "component": "SpeechBrain ECAPA readout",
        "kind": "model-card",
        "url": "https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb/resolve/0f99f2d0ebe89ac095bcc5903c4dd8f72b367286/README.md",
        "sha256": "00f58c3cbd7a7510de9374080da0e82a4c4e8f4df567f7338fe6efe108be705a",
    },
    {
        "path": "apache-2.0.txt",
        "component": "SpeechBrain ECAPA readout",
        "kind": "licence-text",
        "url": "https://www.apache.org/licenses/LICENSE-2.0.txt",
        "sha256": "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30",
    },
    {
        "path": "wavlm-model-card-feb593a6.md",
        "component": "Microsoft WavLM readout",
        "kind": "model-card",
        "url": "https://huggingface.co/microsoft/wavlm-base-plus-sv/resolve/feb593a6c23c1cc3d9510425c29b0a14d2b07b1e/README.md",
        "sha256": "97f5513cde351b3adb4e182b60ec23154dadba7ca83e667ceace237177855b8e",
    },
    {
        "path": "wavlm-unispeech-cc-by-sa-3.0-6112826a.txt",
        "component": "Microsoft WavLM readout",
        "kind": "model-card-linked-licence-text",
        "url": "https://raw.githubusercontent.com/microsoft/UniSpeech/6112826ac13a4327f4c9a7afa2a505e35b763514/LICENSE",
        "sha256": "74295d561f60f770a4aa7525b71c0d119ec70422e9f5c601ee3c77e1b7822c91",
    },
)


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def atomic_write(path: Path, payload: bytes) -> None:
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for record in SNAPSHOTS:
        request = urllib.request.Request(
            record["url"], headers={"User-Agent": "F2-reproducibility-snapshot/1.0"}
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = response.read()
        observed = digest(payload)
        if observed != record["sha256"]:
            raise RuntimeError(
                f"snapshot drift for {record['path']}: expected {record['sha256']} observed {observed}"
            )
        atomic_write(OUT / record["path"], payload)
    manifest = {
        "schema": "f2-license-snapshots-v1",
        "checked": CHECKED,
        "files": list(SNAPSHOTS),
    }
    atomic_write(
        OUT / "SNAPSHOT-MANIFEST.json",
        (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    print(f"PASS — authenticated {len(SNAPSHOTS)} model-card/licence snapshots")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
