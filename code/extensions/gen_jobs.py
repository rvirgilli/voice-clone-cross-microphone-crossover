#!/usr/bin/env python3
"""Run one system's second-generation jobs with the EXP-205 generators, pins and ledgers.

The EXP-205 runners are imported from the frozen generate.py / generate_seedvc.py so
that model pins, reference authentication, atomic writes, resumability and per-clone
ledgers are exactly those of the original campaign. Only the job list differs.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import tempfile
from pathlib import Path

import os

# Historical run layout: these scripts document what ran. Point the two roots at copies of
# the private run directories and the experiment tree; nothing here reads the release itself.
RUNS = Path(os.environ.get("ICASSP_RUNS", "/runs"))
EXPERIMENTS = Path(os.environ.get("ICASSP_EXPERIMENTS", "/experiments"))
HF = Path(os.environ.get("HF_HUB_CACHE", "~/.cache/huggingface/hub")).expanduser()
EXP205 = EXPERIMENTS / "EXP-205-f2-crossmic-crossover"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=EXP205 / "execution-config.json")
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--system", choices=("f5", "xtts", "cosy", "seedvc"), required=True)
    parser.add_argument("--dry", action="store_true", help="authenticate inputs and count jobs; no synthesis")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    payload = json.loads(args.jobs.read_text(encoding="utf-8"))
    todo = [dict(job) for job in payload["jobs"] if job["system"] == args.system]
    for job in todo:
        job["reference"] = Path(job["reference"])
        job["out"] = Path(job["out"])
        if "source" in job:
            job["source"] = Path(job["source"])
    if not todo:
        raise RuntimeError("no jobs for this system")
    pins = config["generators"][args.system]
    source_key = "generate_seedvc" if args.system == "seedvc" else "generate"
    context = {
        "execution_config_sha256": sha256(args.config),
        "manifest_sha256": sha256(args.jobs),
        "generate_source_sha256": config["source_pins"][source_key]["sha256"],
        "generator_pin_sha256": hashlib.sha256(json.dumps(pins, sort_keys=True).encode("utf-8")).hexdigest(),
    }
    print(f"{args.system.upper()}_START total={len(todo)}", flush=True)
    if args.dry:
        gen = load_module("exp205_generate", EXP205 / "generate.py")
        for job in todo:
            gen.require_hash(job["reference"], job["reference_sha256"], "GENERATION_REFERENCE")
        print(f"{args.system.upper()}_DRY_OK total={len(todo)} resumable={sum(1 for j in todo if (EXP205 / 'generate.py').exists() and gen.resumable(j, args.system, context))}")
        return 0

    if args.system == "seedvc":
        sv = load_module("exp205_generate_seedvc", EXP205 / "generate_seedvc.py")
        sv.authenticate_seedvc_runtime(pins)
        import inference
        sv.require_hash(Path(inference.__file__).resolve(), pins["inference_source"]["sha256"], "SEEDVC_IMPORTED_INFERENCE")
        models = None
        original_load = inference.load_models

        def cached_load(namespace):
            nonlocal models
            if models is None:
                models = original_load(namespace)
                sv.authenticate_seedvc_runtime(pins)
            return models

        inference.load_models = cached_load
        for index, job in enumerate(todo, start=1):
            sv.require_hash(job["source"], job["source_sha256"], "SEEDVC_CONTENT")
            sv.require_hash(job["reference"], job["reference_sha256"], "SEEDVC_REFERENCE")
            if sv.resumable(job, context):
                continue
            job["out"].parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(dir=job["out"].parent, prefix=f".{job['out'].stem}.partial.") as temporary_dir:
                sv.set_rng(job["seed"])
                inference.main(argparse.Namespace(
                    source=str(job["source"]), target=str(job["reference"]), output=str(temporary_dir),
                    diffusion_steps=25, length_adjust=1.0, inference_cfg_rate=0.7, f0_condition=False,
                    auto_f0_adjust=False, semi_tone_shift=0, checkpoint=pins["dit_checkpoint"]["path"],
                    config=pins["dit_config"]["path"], fp16=True))
                produced = sorted(Path(temporary_dir).glob("*.wav"))
                if len(produced) != 1 or not sv.valid_audio(produced[0]):
                    raise RuntimeError(f"Seed-VC produced {len(produced)} valid candidates for {job['out']}")
                os.replace(produced[0], job["out"])
            sv.seal(job, context)
            if index % 50 == 0:
                print(f"SEEDVC_PROGRESS {index}/{len(todo)}", flush=True)
        missing = [str(job["out"]) for job in todo if not sv.resumable(job, context)]
    else:
        gen = load_module("exp205_generate", EXP205 / "generate.py")
        runner = {"f5": gen.run_f5, "xtts": gen.run_xtts, "cosy": gen.run_cosy}[args.system]
        runner(todo, pins, context)
        missing = [str(job["out"]) for job in todo if not gen.resumable(job, args.system, context)]
    if missing:
        raise RuntimeError(f"{args.system} generation incomplete: {len(missing)} missing")
    print(f"{args.system.upper()}_COMPLETE total={len(todo)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
