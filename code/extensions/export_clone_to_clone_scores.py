#!/usr/bin/env python3
"""Export the EXP-206 clone-to-clone comparison-level scores for the public release.

One row per comparison and readout: the query clone, the two candidate clones and the
cosine of the query to each, so the per-speaker accuracies and margins of
``data/extension_result.json`` recompute from text alone. The embeddings are read from
the EXP-205 feature cache named by the frozen input manifest and authenticated against
it the way the analyzer does; no audio or embeddings are written.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np

RUNS = Path(os.environ.get("ICASSP_RUNS", "/runs"))
RUN205 = RUNS / "EXP-205-f2-crossmic-crossover"
SYSTEMS = ("f5", "xtts", "cosy", "seedvc")
TEXTS = (0, 1, 2, 3)
ARMS = ("A", "B")
DIRECTIONS = {
    "primary_mic1_to_mic2": ("mic1", "mic2"),
    "reverse_mic2_to_mic1": ("mic2", "mic1"),
}
READOUTS = ("ecapa", "wavlm")
PREFIX = 16
HEADER = [
    "speaker", "direction", "readout",
    "query_system", "query_text_index", "query_prompt_mic", "query_seed_arm", "query_clone_sha256",
    "candidate_system", "candidate_text_index", "candidate_prompt_mic",
    "own_seed_arm", "own_clone_sha256", "other_seed_arm", "other_clone_sha256",
    "cos_own_event", "cos_other_event",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_embeddings(manifest: dict) -> dict[tuple, dict]:
    embeddings = {}
    for row in manifest["clones"]:
        key = (row["speaker"], row["system"], row["text_index"], row["prompt_mic"], row["seed_arm"])
        cache = RUN205 / "feature-cache" / row["cache_file"]
        if sha256(cache) != row["cache_sha256"]:
            raise RuntimeError(f"cache hash mismatch: {key}")
        with np.load(cache, allow_pickle=False) as payload:
            if str(payload["source_sha256"]) != row["clone_sha256"]:
                raise RuntimeError(f"cache source mismatch: {key}")
            if str(payload["readout_pin"]) != manifest["readout_cache_pin"]:
                raise RuntimeError(f"cache readout pin mismatch: {key}")
            value = {"sha256": row["clone_sha256"][:PREFIX]}
            for readout in READOUTS:
                vector = np.asarray(payload[readout], dtype=np.float64)
                value[readout] = vector / float(np.linalg.norm(vector))
        embeddings[key] = value
    if len(embeddings) != 3456:
        raise RuntimeError("embedding census")
    return embeddings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    embeddings = load_embeddings(manifest)
    rows = []
    for speaker in manifest["speakers"]:
        for direction, (query_mic, candidate_mic) in DIRECTIONS.items():
            for query_arm in ARMS:
                other_arm = "B" if query_arm == "A" else "A"
                for query_system in SYSTEMS:
                    for query_text in TEXTS:
                        query = embeddings[(speaker, query_system, query_text, query_mic, query_arm)]
                        for candidate_system in SYSTEMS:
                            if candidate_system == query_system:
                                continue
                            for candidate_text in TEXTS:
                                if candidate_text == query_text:
                                    continue
                                own = embeddings[(speaker, candidate_system, candidate_text, candidate_mic, query_arm)]
                                other = embeddings[(speaker, candidate_system, candidate_text, candidate_mic, other_arm)]
                                for readout in READOUTS:
                                    rows.append([
                                        speaker, direction, readout,
                                        query_system, query_text, query_mic, query_arm, query["sha256"],
                                        candidate_system, candidate_text, candidate_mic,
                                        query_arm, own["sha256"], other_arm, other["sha256"],
                                        f"{float(np.dot(query[readout], own[readout])):.17g}",
                                        f"{float(np.dot(query[readout], other[readout])):.17g}",
                                    ])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(HEADER)
        writer.writerows(rows)
    print(args.out, len(rows), "rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
