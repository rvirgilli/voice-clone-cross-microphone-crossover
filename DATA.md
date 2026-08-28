# Data and model provenance

## Public corpora

This work uses VCTK v0.92 candidate/reference speech and one fixed LibriTTS-R
source utterance per generated text for Seed-VC. The repository does not
redistribute audio. Obtain the corpora from their official releases and verify
the exact selected-file hashes recorded in the trial roster and execution
config before generation.

- VCTK v0.92: <https://doi.org/10.7488/ds/2645>, CC BY 4.0.
- LibriTTS-R SLR141: <https://www.openslr.org/141>, CC BY 4.0.

Keep corpus files outside the Git checkout. A convenient local convention is:

```text
data/
  vctk-0.92/
  libritts-r/
```

For a Level-3 rerun, expose those files below the logical `inputs/` paths in
the released selection manifest (symlinks are sufficient). The runner hashes
every selected input before use, so a wrong corpus version or file mapping is
rejected.

## Systems and fixed readouts

The sealed execution records exact repository states, package versions,
snapshot revisions, and file hashes for F5-TTS, XTTS-v2, CosyVoice2, Seed-VC,
SpeechBrain ECAPA, and Microsoft WavLM. The complete licence inventory is at
`LICENSES.md`.
Notably, the fixed readouts are pinned to:

- `speechbrain/spkrec-ecapa-voxceleb`, revision
  `0f99f2d0ebe89ac095bcc5903c4dd8f72b367286`, Apache-2.0.
- `microsoft/wavlm-base-plus-sv`, revision
  `feb593a6c23c1cc3d9510425c29b0a14d2b07b1e`, MIT.

Model weights are not committed. Never replace a pinned snapshot with a moving
branch name when claiming reproduction of the sealed execution.

## Level-3 environment variables

Run the commands in `REPRODUCE.md` from the repository root. For a Level-3 generation
rerun, the runner requires these six variables:

- `EXP205_ROOT`: the absolute repository root (normally `$PWD`).
- `EXP205_RUN`: a writable, fast work directory outside the Git checkout; it
  receives generated audio, scores, caches, and execution receipts.
- `EXP205_DETECTOR_ROOT`: checkout of the detector/readout uv project used to
  run pin verification and manifest construction.
- `EXP205_XTTS_VENV`: virtual environment containing `coqui-tts==0.25.3`.
- `EXP205_COSY_ROOT`: pinned CosyVoice checkout containing its `venv/`.
- `EXP205_SEEDVC_ROOT`: pinned Seed-VC checkout containing `inference.py`, its
  `venv/`, and the snapshot/cache layout authenticated by the config builder.

For example, replace the placeholders below with local absolute paths:

```bash
export EXP205_ROOT="$PWD"
export EXP205_RUN=/fast/work/exp205-run
export EXP205_DETECTOR_ROOT=/path/to/detector-project
export EXP205_XTTS_VENV=/path/to/xtts-venv
export EXP205_COSY_ROOT=/path/to/CosyVoice
export EXP205_SEEDVC_ROOT=/path/to/seed-vc
```

The execution-config builder also accepts optional model-location overrides.
If omitted, the paths shown here are resolved from the `source/` directory:

| Variable | Default |
|---|---|
| `EXP205_F5_SNAPSHOT` | `external/models/f5-tts` |
| `EXP205_VOCOS_SNAPSHOT` | `external/models/vocos` |
| `EXP205_XTTS_MODEL` | `external/models/xtts-v2` |
| `EXP205_COSY_SNAPSHOT` | `external/models/cosyvoice2` |
| `EXP205_ECAPA_SNAPSHOT` | `external/models/ecapa-voxceleb` |
| `EXP205_WAVLM_SNAPSHOT` | `external/models/wavlm-base-plus-sv` |

`EXP205_COSY_ROOT`, `EXP205_SEEDVC_ROOT`, `EXP205_XTTS_VENV`, and
`EXP205_RUN` are consumed by both the builder and runner as described above.
The builder hashes the loader-visible files and records resolved targets; run
`verify_pins.py` before generation, as required by `REPRODUCE.md`.

## Generated data and publication boundary

The paper-facing artifact distributes sanitized scores, per-speaker summaries,
logical manifests, receipts, hashes, and aggregate results. It excludes source
audio, generated impersonations, model weights, and large feature caches.
Playable generated speech requires a separate human release review regardless
of the upstream licences.

Generation writes must target a fast work volume outside both this repository
and the read-mostly corpus library. Preserve only hashes and milestone outputs
needed for provenance.

## Selection provenance

The public selection package records the complete 30 + 24 + 54 roster
ancestry. Tier 1 and Tier 2 used ECAPA geometry during earlier development;
The roster used here is their disjoint complement and is therefore not embedding-naive.
Within that fixed complement, A/B pair selection used only duration,
transcript-byte rate, and deterministic lexical tie-breaking. The published
feasibility census replaces only a historical absolute run-root prefix with
`inputs/`; its scientific fields and original source hash are preserved.
