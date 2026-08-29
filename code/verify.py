#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy==2.4.6"]
# ///
"""Verify the portable release package and independently recompute its result."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path, PurePosixPath

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "data"
SYSTEMS = ("f5", "xtts", "cosy", "seedvc")
DIRECTIONS = ("primary_mic1_to_mic2", "reverse_mic2_to_mic1")
ENCODERS = ("ecapa", "wavlm")
BOOTSTRAPS = 100_000
SEED = 2052027
EXPECTED_MANIFEST_SHA256 = "4b879491f02badf252365aa4d2b3caa22402c04301c60ed5e02bd06d43f19b2d"
EXPECTED_CHANNEL_SCHEMA = "exp205-posthoc-channel-distinctness-v3"
EXPECTED_DUPLICATE_BAR = 1e-5
EXPECTED_LAG_MS = 50.0
IGNORED_DIRS = {".git", ".venv", ".pytest_cache", "__pycache__", "regenerated",
                "generated-audio", "model-snapshots"}
IGNORED_SUFFIXES = {".pyc", ".aux", ".bbl", ".blg", ".fdb_latexmk", ".fls", ".log"}
FORBIDDEN_RELEASE_SUFFIXES = {
    ".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aac", ".opus",
    ".pt", ".pth", ".ckpt", ".safetensors", ".bin", ".zip", ".tar", ".gz", ".7z",
}
EXPECTED_LICENSE_SNAPSHOTS = {
    "f5-model-card-84e5a410.md": "cf3354feaf4020527b3642b12a5a382de13d0af6bc99e8d37bee84425c1ca224",
    "f5-cc-by-nc-4.0.txt": "41003d4a74749c0220e33dd415042164b5a1093ed401f36277234f772d22d3d0",
    "xtts-v2-model-card-6c2b0d75.md": "1cfa85b3293f685b3a6537f8da3d94820fd111270e553589073885dea3facfb7",
    "xtts-v2-cpml-1.0.0.txt": "190f6d7c19b8984f91b97712b94ce92d2b2e640fc677dacab966e955ece9d043",
    "ecapa-model-card-0f99f2d0.md": "00f58c3cbd7a7510de9374080da0e82a4c4e8f4df567f7338fe6efe108be705a",
    "apache-2.0.txt": "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30",
    "wavlm-model-card-feb593a6.md": "97f5513cde351b3adb4e182b60ec23154dadba7ca83e667ceace237177855b8e",
    "wavlm-unispeech-cc-by-sa-3.0-6112826a.txt": "74295d561f60f770a4aa7525b71c0d119ec70422e9f5c601ee3c77e1b7822c91",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(name: str):
    return json.loads((DATA / name).read_text(encoding="utf-8"))


def close(observed, expected, label: str, tolerance: float = 2e-15) -> None:
    if not math.isclose(float(observed), float(expected), rel_tol=0.0, abs_tol=tolerance):
        raise AssertionError(f"{label}: {observed} != {expected}")


def discover_release_files() -> set[str]:
    files = set()
    for path in ROOT.rglob("*"):
        rel = path.relative_to(ROOT)
        if any(part in IGNORED_DIRS for part in rel.parts):
            continue
        if path.is_symlink():
            raise AssertionError(f"symlink forbidden in release: {rel}")
        if not path.is_file() or path.suffix.lower() in IGNORED_SUFFIXES:
            continue
        name = rel.as_posix()
        if name == "data/checksums.sha256":
            continue  # this inventory is the explicit root of trust
        if path.suffix.lower() in FORBIDDEN_RELEASE_SUFFIXES:
            raise AssertionError(f"audio/model/archive forbidden in release: {name}")
        files.add(name)
    return files


def read_checksum_inventory() -> dict[str, str]:
    records = {}
    previous = ""
    for line in (DATA / "checksums.sha256").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, rel = line.split("  ", 1)
        pure = PurePosixPath(rel)
        if pure.is_absolute() or ".." in pure.parts or pure.as_posix() != rel:
            raise AssertionError(f"unsafe checksum path: {rel}")
        if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
            raise AssertionError(f"invalid checksum digest: {rel}")
        if rel in records:
            raise AssertionError(f"duplicate checksum entry: {rel}")
        if previous and rel <= previous:
            raise AssertionError("checksum inventory is not strictly path-sorted")
        records[rel] = digest
        previous = rel
    return records


def verify_license_snapshots() -> int:
    root = ROOT / "licenses"
    manifest = json.loads((root / "SNAPSHOT-MANIFEST.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != "f2-license-snapshots-v1":
        raise AssertionError("licence snapshot schema invalid")
    records = manifest.get("files", [])
    by_path = {record.get("path"): record for record in records}
    if len(by_path) != len(records) or set(by_path) != set(EXPECTED_LICENSE_SNAPSHOTS):
        raise AssertionError("licence snapshot census invalid")
    actual = {path.name for path in root.iterdir() if path.is_file()}
    if actual != set(EXPECTED_LICENSE_SNAPSHOTS) | {"SNAPSHOT-MANIFEST.json"}:
        raise AssertionError("unlisted or missing licence snapshot")
    for name, expected in EXPECTED_LICENSE_SNAPSHOTS.items():
        record = by_path[name]
        if record.get("sha256") != expected or sha256(root / name) != expected:
            raise AssertionError(f"licence snapshot hash mismatch: {name}")
        if not str(record.get("url", "")).startswith("https://"):
            raise AssertionError(f"licence snapshot source is not HTTPS: {name}")
    if "license: cc-by-nc-4.0" not in (root / "f5-model-card-84e5a410.md").read_text():
        raise AssertionError("F5 model-card licence changed")
    cpml = (root / "xtts-v2-cpml-1.0.0.txt").read_text()
    if "only non-commercial use of a machine learning model and its outputs" not in cpml:
        raise AssertionError("XTTS CPML output terms missing")
    if "license: \"apache-2.0\"" not in (root / "ecapa-model-card-0f99f2d0.md").read_text():
        raise AssertionError("ECAPA model-card licence changed")
    if "Attribution-ShareAlike 3.0 Unported" not in (
        root / "wavlm-unispeech-cc-by-sa-3.0-6112826a.txt"
    ).read_text():
        raise AssertionError("WavLM linked licence changed")
    return len(records)


def capture_census_sha256(speakers: list[dict]) -> tuple[str, int]:
    census = []
    for spk in speakers:
        audio = spk.get("audio", {})
        for event in ("A", "B"):
            m1, m2 = audio.get(f"{event}_mic1"), audio.get(f"{event}_mic2")
            if not isinstance(m1, dict) or not isinstance(m2, dict):
                raise AssertionError(f"channel manifest arm missing: {spk.get('speaker')}/{event}")
            if m1.get("sha256") == m2.get("sha256"):
                raise AssertionError(f"channel manifest pair is byte-identical: {spk.get('speaker')}/{event}")
            census.append({
                "speaker": spk["speaker"], "event": event,
                "mic1_sha256": m1["sha256"], "mic2_sha256": m2["sha256"],
            })
    payload = json.dumps(census, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest(), len(census)


def main() -> int:
    # The checksum file is the explicit root of trust; it must enumerate every other
    # release payload file exactly once. Runtime caches and documented generated files are
    # outside the release surface.
    records = read_checksum_inventory()
    actual_files = discover_release_files()
    if set(records) != actual_files:
        raise AssertionError(
            f"release inventory mismatch: missing={sorted(set(records)-actual_files)}, "
            f"unlisted={sorted(actual_files-set(records))}"
        )
    for rel, digest in records.items():
        if sha256(ROOT / rel) != digest:
            raise AssertionError(f"checksum mismatch: {rel}")
    checksum_count = len(records)
    licence_snapshot_count = verify_license_snapshots()

    # Released files must not contain local machine paths.
    for name in ("trials.json", "generation_ledger.json", "generation_config.json",
                 "analysis_config.json", "scores.tsv"):
        text = (DATA / name).read_text(encoding="utf-8")
        if any(root in text for root in ("/home/", "/mnt/", "/media/")):
            raise AssertionError(f"private path in {name}")

    result = load("result.json")
    manifest = load("trials.json")
    selection_manifest = load("selection_manifest.json")
    receipt = load("generation_ledger.json")
    antecedent = load("antecedent_seed_crossover.json")
    old = antecedent.get("POOLED", {})
    if (
        old.get("follow_rate") != 0.806
        or old.get("ci95") != [0.757, 0.854]
        or old.get("bar_point_at_least") != 0.90
        or old.get("VERDICT") != "AMBIGUOUS"
        or old.get("n_conjuncts_declared") != 3
        or old.get("n_conjuncts_evaluated") != 3
        or [item.get("passed") for item in old.get("conjuncts", [])] != [False, True, False]
    ):
        raise AssertionError("antecedent EXP-204 conjunctive correction missing or invalid")
    if result["status"] != "SCIENTIFIC_RESULT" or manifest["counts"]["speakers"] != 54:
        raise AssertionError("result/manifest scope invalid")
    speakers = result["counts"]["speaker_ids"]
    if len(speakers) != 54 or len(set(speakers)) != 54:
        raise AssertionError("speaker census invalid")

    with (DATA / "scores.tsv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if len(rows) != 3456:
        raise AssertionError("score row count invalid")
    identities = [tuple(row[k] for k in ("speaker", "system", "text_index", "prompt_mic", "seed_arm")) for row in rows]
    if len(set(identities)) != 3456:
        raise AssertionError("duplicate score identity")
    expected = {
        (speaker, system, str(text), mic, arm)
        for speaker in speakers
        for system in SYSTEMS
        for text in range(4)
        for mic in ("mic1", "mic2")
        for arm in ("A", "B")
    }
    if set(identities) != expected:
        raise AssertionError("score identity grid invalid")

    receipt_ids = {
        (r["speaker"], r["system"], str(r["text_index"]), r["prompt_mic"], r["seed_arm"]): r
        for r in receipt["clones"]
    }
    if set(receipt_ids) != expected or receipt["clone_count"] != 3456:
        raise AssertionError("receipt identity grid invalid")
    candidate_bindings = {}
    clone_hashes = set()
    for row in rows:
        key = tuple(row[k] for k in ("speaker", "system", "text_index", "prompt_mic", "seed_arm"))
        if receipt_ids[key]["sha256"] != row["clone_sha256"]:
            raise AssertionError(f"clone hash/receipt mismatch: {key}")
        clone_hashes.add(row["clone_sha256"])
        for prefix in ("candidate_A", "candidate_B", "same_candidate_A", "same_candidate_B"):
            path, digest = row[f"{prefix}_path"], row[f"{prefix}_sha256"]
            if path in candidate_bindings and candidate_bindings[path] != digest:
                raise AssertionError(f"candidate hash inconsistency: {path}")
            candidate_bindings[path] = digest
    if len(clone_hashes) != 3456 or len(candidate_bindings) != 216:
        raise AssertionError("clone/candidate census invalid")

    followed = defaultdict(list)
    same_followed = defaultdict(list)
    system_values = defaultdict(list)
    for row in rows:
        speaker, system, _, mic, arm = (row[k] for k in ("speaker", "system", "text_index", "prompt_mic", "seed_arm"))
        direction = DIRECTIONS[0] if mic == "mic1" else DIRECTIONS[1]
        diagnostic = f"same_{mic}"
        other = "B" if arm == "A" else "A"
        for encoder in ENCODERS:
            own, alt = float(row[f"{encoder}_{arm}"]), float(row[f"{encoder}_{other}"])
            same_own = float(row[f"{encoder}_same_{arm}"])
            same_alt = float(row[f"{encoder}_same_{other}"])
            value = 1.0 if own > alt else 0.0 if own < alt else 0.5
            same_value = 1.0 if same_own > same_alt else 0.0 if same_own < same_alt else 0.5
            followed[(direction, encoder, speaker)].append(value)
            same_followed[(diagnostic, encoder, speaker)].append(same_value)
            system_values[(direction, encoder, system)].append(value)

    vectors = {}
    same_vectors = {}
    for direction in DIRECTIONS:
        for encoder in ENCODERS:
            vectors[(direction, encoder)] = np.asarray(
                [np.mean(followed[(direction, encoder, speaker)]) for speaker in speakers], dtype=np.float64
            )
    for diagnostic in ("same_mic1", "same_mic2"):
        for encoder in ENCODERS:
            same_vectors[(diagnostic, encoder)] = np.asarray(
                [np.mean(same_followed[(diagnostic, encoder, speaker)]) for speaker in speakers], dtype=np.float64
            )
    if any(len(values) != 32 for values in followed.values()) or any(len(values) != 32 for values in same_followed.values()):
        raise AssertionError("speaker cell does not contain 32 comparisons")

    rng = np.random.default_rng(SEED)
    indices = rng.integers(0, 54, size=(BOOTSTRAPS, 54), dtype=np.int32)
    for (direction, encoder), vector in vectors.items():
        node = result["directions"][direction][encoder]
        close(vector.mean(), node["point"], f"{direction}/{encoder}/point")
        if not np.array_equal(vector, np.asarray(node["speaker_means"], dtype=np.float64)):
            raise AssertionError(f"speaker vector mismatch: {direction}/{encoder}")
        lo, hi = np.quantile(vector[indices].mean(axis=1), (0.025, 0.975))
        close(lo, node["stability_interval_95"][0], f"{direction}/{encoder}/lo")
        close(hi, node["stability_interval_95"][1], f"{direction}/{encoder}/hi")
    for (diagnostic, encoder), vector in same_vectors.items():
        node = result["same_microphone_diagnostics_no_verdict"][diagnostic][encoder]
        close(vector.mean(), node["point"], f"{diagnostic}/{encoder}/point")
        lo, hi = np.quantile(vector[indices].mean(axis=1), (0.025, 0.975))
        close(lo, node["stability_interval_95"][0], f"{diagnostic}/{encoder}/lo")
        close(hi, node["stability_interval_95"][1], f"{diagnostic}/{encoder}/hi")
    for direction in DIRECTIONS:
        for encoder in ENCODERS:
            for system in SYSTEMS:
                close(
                    np.mean(system_values[(direction, encoder, system)]),
                    result["system_points_no_intervals"][direction][encoder][system],
                    f"system/{direction}/{encoder}/{system}",
                )

    p_ecapa = result["directions"][DIRECTIONS[0]]["ecapa"]
    p_wavlm = result["directions"][DIRECTIONS[0]]["wavlm"]
    r_ecapa = result["directions"][DIRECTIONS[1]]["ecapa"]
    r_wavlm = result["directions"][DIRECTIONS[1]]["wavlm"]
    bars = (
        p_ecapa["stability_interval_95"][0] > .5,
        p_ecapa["point"] >= .8 and p_ecapa["stability_interval_95"][0] > .7,
        p_wavlm["stability_interval_95"][0] > .5,
        r_ecapa["stability_interval_95"][0] > .5,
        r_wavlm["stability_interval_95"][0] > .5,
    )
    if bars != (True, True, True, True, True):
        raise AssertionError(f"frozen bars did not all pass: {bars}")
    if result["headline_permission"] != "BIDIRECTIONAL_HEADLINE_PERMITTED":
        raise AssertionError("headline permission mismatch")

    # Channel distinctness is load-bearing for the crossover control (Gate 2, A1/A2):
    # authenticate the measurement's own contract, not merely the file's hash.
    ch = load("channel_distinctness.json")
    if ch.get("schema") != EXPECTED_CHANNEL_SCHEMA:
        raise AssertionError("channel probe schema invalid")
    manifest_root = result["artifact_hashes"].get("manifest")
    portable_manifest_hash = sha256(DATA / "selection_manifest.json")
    if manifest_root != EXPECTED_MANIFEST_SHA256:
        raise AssertionError("frozen manifest trust root changed")
    if ch.get("frozen_manifest_sha256") != manifest_root:
        raise AssertionError("channel result's frozen manifest trust root changed")
    if ch.get("manifest_sha256") not in {manifest_root, portable_manifest_hash}:
        raise AssertionError("channel result is not bound to the frozen manifest")
    census_digest, census_count = capture_census_sha256(selection_manifest["speakers"])
    if census_count != 108 or ch.get("capture_census_sha256") != census_digest:
        raise AssertionError("channel result is not bound to the released 108-pair hash census")
    if ch["n_speakers"] != 54 or ch["n_event_captures"] != 108:
        raise AssertionError(f"channel census incomplete: {ch['n_speakers']}x2 != 108")
    if ch["byte_identical_pairs"] != 0:
        raise AssertionError("a mic1/mic2 pair is byte-identical: crossover control vacuous")
    bar = ch["duplicate_bar_alignment_residual"]
    if bar != EXPECTED_DUPLICATE_BAR or ch.get("lag_window_ms") != EXPECTED_LAG_MS:
        raise AssertionError("channel transform contract changed")
    if ch["residual_after_alignment"]["min"] <= bar:
        raise AssertionError("a capture pair is at or below the duplicate bar")
    controls = ch["injection_controls"]
    if controls.get("n_source_captures") != 108:
        raise AssertionError("injection controls do not cover all 108 source captures")
    if controls.get("tested_shift_samples") != [-2400, -120, -1, 1, 120, 2400]:
        raise AssertionError("injection controls do not cover both 50 ms boundaries")
    by_transform = controls.get("max_residual_by_transform", {})
    required_transforms = {
        "exact_copy", "gain_scaled_copy", "shifted_copy_+2400", "shifted_copy_-2400",
        "gain_scaled_shifted_copy_+2400", "gain_scaled_shifted_copy_-2400",
    }
    if not required_transforms.issubset(by_transform):
        raise AssertionError("injection transform coverage incomplete")
    if not by_transform or controls.get("max_residual") != max(by_transform.values()):
        raise AssertionError("injection-control maximum is internally inconsistent")
    if controls["max_residual"] > bar:
        raise AssertionError("injected duplicates not detected: the probe cannot fail")

    # Bind the released manuscript source to the recomputed headline and channel evidence.
    tex_raw = (ROOT / "paper" / "F2" / "main.tex").read_text(encoding="utf-8")
    tex = " ".join(tex_raw.split())
    required_manuscript = {
        "title": "Which Conditioning Recording Does a Voice Clone Follow?",
        "primary ECAPA": ".896 [.869,.921]",
        "primary WavLM": ".622 [.593,.652]",
        "reverse ECAPA": ".907 [.881,.931]",
        "reverse WavLM": ".620 [.589,.653]",
        "channel census": "Across all 108 event captures",
        "channel aligned median": "reach $.92$ median",
        "channel residual range": "is $.38$ median (range $.21$--$.70$)",
        "channel injection maximum": "at most $.000000026$",
        "channel byte result": "no pair is byte-identical",
        "duplicate tolerance boundary": "not a universal perceptual threshold",
        "artifact locator": "github.com/rvirgilli/voice-clone-cross-microphone-crossover/tree/f2-icassp2027-final",
        "pre-specified plan wording": "pre-specified complete crossover",
        "known-positive triage scope": "known-positive two-recording set for human provenance review",
        "no arbitrary presence decision": "cannot decide whether an arbitrary queried recording was present",
        "prosody-cloning positioning": "\\cite{prosodyclone2022}",
        "source-voiceprint positioning": "\\cite{sourcevoiceprint2023}",
        "rank-disclosure positioning": "\\cite{rankdisclosure2026,sterns2026}",
        "prior-intervention distinction": "None intervenes on which of two same-speaker recording events conditions a clone",
        "open-set boundary": "neither identifies the carrier nor solves open-set recording-presence detection",
    }
    for label, phrase in required_manuscript.items():
        if phrase not in tex:
            raise AssertionError(f"released manuscript source lost {label}: {phrase!r}")
    retired_claims = (
        "is reproduced after", "ECAPA-only artifact", "exact frozen source bytes",
        "population confidence interval", "open-set confirmation", "operationally large",
        "simultaneous VCTK", "bidirectionally replicated", "pre-registered",
    )
    present = [phrase for phrase in retired_claims if phrase in tex_raw]
    if present:
        raise AssertionError(f"retired manuscript claims restored: {present}")
    print(
        f"PASS — {checksum_count} payload files match their checksums; "
        f"{licence_snapshot_count} licence snapshots authenticate; census, generation ledger, "
        f"statistics, channel control and frozen result all verify"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
