"""Build the hash-pinned, outcome-blind EXP-205 execution config.

This intentionally hashes several gigabytes of generator/readout weights. Run
it only when the shared GPU and storage path are not serving another workload.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


EXP = Path(__file__).resolve().parent
MANIFEST = EXP / "selection-manifest.json"
RUN = Path(os.environ.get("EXP205_RUN", "runtime/exp205"))


def configured_path(name: str, default: str) -> Path:
    return Path(os.environ.get(name, default))

F5_SNAPSHOT = configured_path("EXP205_F5_SNAPSHOT", "external/models/f5-tts")
VOCOS_SNAPSHOT = configured_path("EXP205_VOCOS_SNAPSHOT", "external/models/vocos")
XTTS_DIR = configured_path("EXP205_XTTS_MODEL", "external/models/xtts-v2")
COSY_ROOT = configured_path("EXP205_COSY_ROOT", "external/CosyVoice")
COSY_SNAPSHOT = configured_path("EXP205_COSY_SNAPSHOT", "external/models/cosyvoice2")
SEEDVC_ROOT = configured_path("EXP205_SEEDVC_ROOT", "external/seed-vc")
SEEDVC_SNAPSHOTS = (
    SEEDVC_ROOT
    / "checkpoints/models--Plachta--Seed-VC/snapshots/257283f9f41585055e8f858fba4fd044e5caed6e",
    SEEDVC_ROOT
    / "checkpoints/hf_cache/models--openai--whisper-small/snapshots/973afd24965f72e36ca33b3055d56a652f456b4d",
    SEEDVC_ROOT
    / "checkpoints/hf_cache/models--nvidia--bigvgan_v2_22khz_80band_256x/snapshots/633ff708ed5b74903e86ff1298cf4a98e921c513",
    SEEDVC_ROOT
    / "checkpoints/models--funasr--campplus/snapshots/e4b6ede7ce16997aff4ae69fbca1f0175e2afede",
)
SEEDVC_REFS = (
    SEEDVC_ROOT / "checkpoints/models--Plachta--Seed-VC/refs/main",
    SEEDVC_ROOT / "checkpoints/hf_cache/models--openai--whisper-small/refs/main",
    SEEDVC_ROOT
    / "checkpoints/hf_cache/models--nvidia--bigvgan_v2_22khz_80band_256x/refs/main",
    SEEDVC_ROOT / "checkpoints/models--funasr--campplus/refs/main",
)
ECAPA_SNAPSHOT = configured_path("EXP205_ECAPA_SNAPSHOT", "external/models/ecapa-voxceleb")
WAVLM_SNAPSHOT = configured_path("EXP205_WAVLM_SNAPSHOT", "external/models/wavlm-base-plus-sv")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pin(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    # Preserve the loader-visible logical path.  Hugging Face snapshots are
    # symlink forests; recording only ``resolve()`` would authenticate an old
    # blob even after the snapshot symlink was retargeted.
    logical = path.absolute()
    return {
        "path": str(logical),
        "resolved_path": str(logical.resolve(strict=True)),
        "sha256": sha256(logical),
    }


def pin_tree(path: Path) -> list[dict[str, str]]:
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        raise RuntimeError(f"empty model snapshot: {path}")
    return [pin(item) for item in files]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=EXP / "execution-config.json")
    args = parser.parse_args()

    f5_checkpoint = F5_SNAPSHOT / "F5TTS_v1_Base/model_1250000.safetensors"
    seedvc_dit_checkpoint = (
        SEEDVC_SNAPSHOTS[0]
        / "DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned.pth"
    )
    seedvc_dit_config = (
        SEEDVC_SNAPSHOTS[0]
        / "config_dit_mel_seed_uvit_whisper_small_wavenet.yml"
    )
    f5_checkpoint_record = pin(f5_checkpoint)
    source_names = (
        "build_execution_config.py",
        "build_seedvc_manifest.py",
        "generate.py",
        "generate_seedvc.py",
        "score.py",
        "analyze.py",
        "verdict.py",
        "verify_pins.py",
        "test_generate.py",
        "test_manifest.py",
        "test_score.py",
        "test_analyze.py",
        "test_verdict.py",
        "run.sh",
        "LICENSES.md",
    )
    config = {
        "schema": "exp205-execution-config-v1",
        "purpose": "prospective_crossmic_crossover_no_discovery_inputs",
        "manifest": pin(MANIFEST),
        "run_root": str(RUN),
        "scores_path": str(RUN / "scores.tsv"),
        "execution_receipt_path": str(RUN / "execution-receipt.json"),
        "analysis_config_path": str(RUN / "analysis-config.json"),
        "scientific_result_path": str(RUN / "sealed/scientific-result.json"),
        "expected_counts": {
            "speakers": 54,
            "systems": 4,
            "prompt_microphones": 2,
            "seed_arms": 2,
            "texts": 4,
            "clones": 3456,
            "real_candidates": 216,
        },
        "environments": {
            "f5": {"python": "3.10", "f5-tts": "1.1.22"},
            "xtts": {"venv": os.environ.get("EXP205_XTTS_VENV", "external/venvs/xtts"), "coqui-tts": "0.25.3"},
            "cosy": {"venv": str(COSY_ROOT / "venv")},
            "seedvc": {"venv": str(SEEDVC_ROOT / "venv")},
            "readouts": {
                "python": "3.11",
                "speechbrain": "1.1.0",
                "transformers": "5.15.0",
                "torch": "2.13.0",
                "librosa": "0.11.0",
                "numpy": "2.4.6",
            },
        },
        "generators": {
            "f5": {
                "package_version": "1.1.22",
                "api_source_sha256": "3a1fe090e70f4c7edab5a3fb27727a93ac8f71d664768a54b7bc308933af43a3",
                "infer_source_sha256": "b05b7b159ceabe07f318b68c082970ec14ba8481fe2812e3774f8246628267c8",
                # F5 dispatches on the suffix. Use the same logical snapshot
                # path authenticated in ``files``, never its suffixless blob.
                "checkpoint_path": f5_checkpoint_record["path"],
                "vocoder_path": str(VOCOS_SNAPSHOT.resolve()),
                "files": [f5_checkpoint_record, *pin_tree(VOCOS_SNAPSHOT)],
            },
            "xtts": {
                "package_version": "0.25.3",
                "api_source_sha256": "9fd85c29d0cb461c8bfb0f4f428ec4e5cb2df50321ffc3f404255163867da161",
                "files": pin_tree(XTTS_DIR),
            },
            "cosy": {
                "api_source_sha256": "4b2a605f129d172cd031286a14a591396c2e8ed0bf53258acc7445525fdaa65b",
                "snapshot_path": str(COSY_SNAPSHOT.resolve()),
                "snapshot_revision": "eec1ae6c79877dbd9379285cf8789c9e0879293d",
                "files": pin_tree(COSY_SNAPSHOT),
            },
            "seedvc": {
                "inference_source": pin(SEEDVC_ROOT / "inference.py"),
                "dit_checkpoint": pin(seedvc_dit_checkpoint),
                "dit_config": pin(seedvc_dit_config),
                "files": [
                    *[
                        record
                        for tree in SEEDVC_SNAPSHOTS
                        for record in pin_tree(tree)
                    ],
                    *[pin(path) for path in SEEDVC_REFS],
                ],
            },
        },
        "readouts": {
            "ecapa": {
                "snapshot_path": str(ECAPA_SNAPSHOT.resolve()),
                "snapshot_revision": "0f99f2d0ebe89ac095bcc5903c4dd8f72b367286",
                "files": pin_tree(ECAPA_SNAPSHOT),
                "loader_sha256": {
                    "ecapa_speaker": "8d9b3da17b363353c754ff0f2ebd66606545c74af6b0e36d6db13bd2f6a49ee3"
                },
            },
            "wavlm": {
                "snapshot_path": str(WAVLM_SNAPSHOT.resolve()),
                "snapshot_revision": "feb593a6c23c1cc3d9510425c29b0a14d2b07b1e",
                "files": pin_tree(WAVLM_SNAPSHOT),
                "loader_sha256": {
                    "wavlm_model": "b9607c8c9c94d6d8567f13246d1eb5b793e219419032cd8ee8cc9926ce80f145",
                    "wavlm_feature_extractor": "e5e9a0baf70716fee503f4f66a7a61312a132be989b2d7e2649e057ccbefa2cc",
                },
            },
        },
        "repositories": {
            "cosyvoice": {
                "path": str(COSY_ROOT),
                "commit": "074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc",
                "allowed_status": ["?? smoke_cosyvoice2.py"],
            },
            "seedvc": {
                "path": str(SEEDVC_ROOT),
                "commit": "51383efd921027683c89e5348211d93ff12ac2a8",
                "allowed_status": ["?? batch_convert.py"],
            },
        },
        "source_pins": {name.removesuffix(".py").replace(".", "_"): pin(EXP / name) for name in source_names},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "EXP205_EXECUTION_CONFIG_BUILT_NO_OUTCOMES",
                "sha256": sha256(args.out),
                "generator_model_files": sum(
                    len(value["files"]) for value in config["generators"].values()
                ),
                "readout_model_files": sum(
                    len(value["files"]) for value in config["readouts"].values()
                ),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
