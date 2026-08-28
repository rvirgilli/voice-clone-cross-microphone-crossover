"""Outcome-blind integrity gate for the complete this work execution config."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(record: dict[str, str], label: str) -> None:
    path = Path(record["path"])
    expected_resolved = record.get("resolved_path")
    if not isinstance(expected_resolved, str) or not expected_resolved:
        raise RuntimeError(f"{label} has no frozen resolved_path: {path}")
    try:
        observed_resolved = str(path.resolve(strict=True))
    except OSError as exc:
        raise RuntimeError(f"{label} logical path is unavailable: {path}") from exc
    if observed_resolved != expected_resolved:
        raise RuntimeError(
            f"{label} resolved target mismatch expected={expected_resolved} "
            f"observed={observed_resolved} path={path}"
        )
    observed = sha256(path)
    if observed != record["sha256"]:
        raise RuntimeError(
            f"{label} hash mismatch expected={record['sha256']} observed={observed} path={path}"
        )


def git_head(path: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def git_status(path: Path) -> list[str]:
    result = subprocess.run(
        ["git", "-C", str(path), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    )
    return sorted(line for line in result.stdout.splitlines() if line)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config.get("schema") != "exp205-execution-config-v1":
        raise RuntimeError("execution config schema mismatch")
    verify(config["manifest"], "MANIFEST")
    if config["expected_counts"] != {
        "speakers": 54,
        "systems": 4,
        "prompt_microphones": 2,
        "seed_arms": 2,
        "texts": 4,
        "clones": 3456,
        "real_candidates": 216,
    }:
        raise RuntimeError("execution count contract mismatch")

    n_model_files = 0
    for name, generator in config["generators"].items():
        for record in generator["files"]:
            verify(record, f"GENERATOR_{name.upper()}")
            n_model_files += 1
        for field in ("inference_source", "dit_checkpoint", "dit_config"):
            if field in generator:
                verify(
                    generator[field],
                    f"GENERATOR_{name.upper()}_{field.upper()}",
                )
    for name, readout in config["readouts"].items():
        for record in readout["files"]:
            verify(record, f"READOUT_{name.upper()}")
            n_model_files += 1
    for name, record in config["source_pins"].items():
        verify(record, f"SOURCE_{name.upper()}")

    for name, repository in config["repositories"].items():
        observed = git_head(Path(repository["path"]))
        if observed != repository["commit"]:
            raise RuntimeError(
                f"{name} commit mismatch expected={repository['commit']} observed={observed}"
            )
        status = git_status(Path(repository["path"]))
        if status != sorted(repository["allowed_status"]):
            raise RuntimeError(
                f"{name} worktree mismatch expected={repository['allowed_status']} observed={status}"
            )
    print(
        json.dumps(
            {
                "status": "EXP205_PINS_OK_NO_OUTCOMES",
                "config_sha256": sha256(args.config),
                "model_files": n_model_files,
                "source_files": len(config["source_pins"]),
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
