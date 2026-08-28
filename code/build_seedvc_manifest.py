"""Build the exact Seed-VC conversion manifest (outcome blind)."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    manifest_path = Path(config["manifest"]["path"])
    if sha256(manifest_path) != config["manifest"]["sha256"]:
        raise RuntimeError("selection manifest hash mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "exp205-selection-manifest-v1":
        raise RuntimeError("selection manifest schema mismatch")
    sources = manifest["generation"]["seedvc_sources"]
    if len(sources) != 4:
        raise RuntimeError("expected four Seed-VC content sources")
    generated_texts = manifest["generation"]["generated_texts"]
    for source in sources:
        if sha256(Path(source["path"])) != source["sha256"]:
            raise RuntimeError(f"Seed-VC source hash mismatch: {source['path']}")
        transcript_path = Path(source["transcript_path"])
        if sha256(transcript_path) != source["transcript_sha256"]:
            raise RuntimeError(f"Seed-VC transcript hash mismatch: {transcript_path}")
        index = int(source["index"])
        transcript = transcript_path.read_text(encoding="utf-8").strip()
        if transcript != source["transcript"] or transcript != generated_texts[index]["text"]:
            raise RuntimeError(f"Seed-VC content mismatch at text index {index}")

    run_root = Path(config["run_root"])
    seed_base = int(manifest["generation"]["rng_seed_base"])
    rows = []
    for speaker in manifest["speakers"]:
        for prompt_mic in ("mic1", "mic2"):
            for arm in ("A", "B"):
                target = speaker["audio"][f"{arm}_{prompt_mic}"]
                if sha256(Path(target["path"])) != target["sha256"]:
                    raise RuntimeError(f"Seed-VC target hash mismatch: {target['path']}")
                out_dir = run_root / "clones" / f"seedvc__{prompt_mic}__seed{arm}"
                for source in sources:
                    index = int(source["index"])
                    rows.append(
                        {
                            "source": source["path"],
                            "target": target["path"],
                            "seed": seed_base + index,
                            "out": str(out_dir / f"{speaker['speaker']}_t{index}.wav"),
                        }
                    )
    if len(rows) != 864 or len({row["out"] for row in rows}) != 864:
        raise RuntimeError("Seed-VC manifest census mismatch")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "SEEDVC_MANIFEST_BUILT_NO_OUTCOMES",
                "conversions": len(rows),
                "sha256": sha256(args.out),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
