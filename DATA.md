# Data and model provenance

## Public corpora

EXP-205 uses VCTK v0.92 candidate/reference speech and one fixed LibriTTS-R
source utterance per generated text for Seed-VC. The repository does not
redistribute audio. Obtain the corpora from their official releases and verify
the exact selected-file hashes recorded in the portable manifest and execution
config before generation.

- VCTK v0.92: <https://doi.org/10.7488/ds/2645>, CC BY 4.0.
- LibriTTS-R SLR141: <https://www.openslr.org/141>, CC BY 4.0.

Keep corpus files outside the Git checkout. A convenient local convention is:

```text
data/
  vctk-0.92/
  libritts-r/
```

## Systems and fixed readouts

The sealed execution records exact repository states, package versions,
snapshot revisions, and file hashes for F5-TTS, XTTS-v2, CosyVoice2, Seed-VC,
SpeechBrain ECAPA, and Microsoft WavLM. The complete licence inventory is at
`experiments/EXP-205-f2-crossmic-crossover/audit_package/exp205/LICENSES.md`.
Notably, the fixed readouts are pinned to:

- `speechbrain/spkrec-ecapa-voxceleb`, revision
  `0f99f2d0ebe89ac095bcc5903c4dd8f72b367286`, Apache-2.0.
- `microsoft/wavlm-base-plus-sv`, revision
  `feb593a6c23c1cc3d9510425c29b0a14d2b07b1e`, MIT.

Model weights are not committed. Never replace a pinned snapshot with a moving
branch name when claiming reproduction of the sealed execution.

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
EXP-205 is their disjoint complement and is therefore not embedding-naive.
Within that fixed complement, EXP-205 A/B pair selection used only duration,
transcript-byte rate, and deterministic lexical tie-breaking. The portable
feasibility census replaces only a historical absolute run-root prefix with
`inputs/`; its scientific fields and original source hash are preserved.
