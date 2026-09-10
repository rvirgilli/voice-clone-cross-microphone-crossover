# Reproducing this work

Python 3.11 or newer and [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync --frozen
```

## 1. Verify the released results (CPU, seconds)

```bash
uv run --frozen python code/verify.py
```

Treats `data/checksums.sha256` as the explicit root of trust, requires it to list
every other release payload file exactly once, and then recomputes from
`data/scores.tsv`:

- the 3,456-row score census and the 54-speaker × 2-event roster;
- the attribution accuracies and their 100,000-draw whole-speaker bootstrap intervals,
  for both directions and both speaker-verification readouts;
- the generation ledger, covering all 3,456 clones;
- the stored microphone-channel control summary, which records that the two
  captures of each event are distinct signals rather than copies of one another
  (re-measuring it from audio is section 4);
- the within-speaker presence-detection points of the paper's §5 (EER and
  normalized minimum DCF at two priors);
- all eight pinned model-card/licence snapshots and XTTS's explicit output terms;
- the registered decision rule applied to those statistics.

The inventory check rejects unlisted payloads, symlinks, audio, model weights and
archives. Runtime caches and reproducible LaTeX intermediates are outside the release
surface.

## 2. Run the tests (CPU, seconds)

```bash
uv run --frozen --with pytest --with soundfile python -m pytest -q code/
```

One test is skipped: it reads LibriTTS-R source transcripts, which this release
does not redistribute. The mapping it would check is recorded in
`data/trials.json` and `data/generation_ledger.json`.

## 3. Sensitivity analyses (CPU, seconds)

```bash
uv run --frozen python code/roster_ancestry_sensitivity.py
uv run --frozen python code/arm_pairing_sensitivity.py
uv run --frozen python code/presence_detection_sensitivity.py
```

All three rewrite their result files in `data/`; the recomputed values must match the
committed ones. The third one is the within-speaker presence-detection boundary of
the paper's §5: EER and normalized minimum DCF under one global threshold, with
whole-speaker bootstrap intervals (about a minute on CPU).

## 4b. Post-hoc extensions (§4.3 and §5; audio and GPU needed to re-run)

`code/extensions/` documents the N-candidate scaling, second-generation cloning and
prompt-intervention analyses (see its README). Their comparison-level cosine tables
are released (`data/ncandidate_scores.tsv`, `data/second_generation_scores.tsv`,
`data/intervention_scores.tsv`) together with the summaries the paper prints;
`verify.py` recomputes every printed point from those tables by speaker-weighted
averaging. The clone-to-clone comparison of §4.1 is `data/extension_result.json`, with
its comparison-level cosine scores in `data/clone_to_clone_scores.tsv` (one row per
comparison and readout, query and candidate clones identified); `code/extension_verify.py`
rebuilds the per-speaker accuracies and margins from that table and reproduces the
aggregate estimates, intervals and decision from the speaker summaries.
The fresh-pair replication (the same crossover on a second metadata-selected A/B pair for
the 53 speakers that admit one; same systems, texts, readouts and decision rule) is
`data/fresh_pair_result.json`, with its comparison-level cosines in
`data/fresh_pair_scores.tsv` and the outcome-blind pair manifest in
`data/selection_manifest_fresh.json`; `verify.py` recomputes its four points from that
table under the 53 × 32 census and checks the stored replication reading.
As for the primary result, `input_hashes` in the fresh-pair result record the private
files as they ran; the released manifest has its audio paths relativized to `inputs/`,
so it does not hash to the recorded manifest value.
The full paired-capture roster (the same crossover, systems, texts, readouts and decision
rule on the 54 speakers whose pairs passed the earlier ECAPA screen, and on the complete
108-speaker paired-capture roster obtained by adding them to the primary 54) is
`data/full_roster_result.json`, with its comparison-level cosines in
`data/full_roster_scores.tsv` and the outcome-blind pair manifest in
`data/selection_manifest_213.json`; `verify.py` recomputes the four points of the added
54 from that table under the 54 × 32 census, recomputes the four 108-speaker points by
concatenating the released per-speaker means of the primary result with those, and checks
the stored reading for both cohorts. Here too, `input_hashes` in the result record the
private files as they ran, and the released manifest has its audio paths relativized to
`inputs/`, so it does not hash to the recorded manifest value.
The readout roster of §5 (fixed non-verification readouts on the primary grid) is
`data/readout_roster.json`, produced by `code/extensions/extract_readouts.py` and
`analyze_readouts.py`; its features are not released, so `verify.py` checks that its
pipeline-control cells equal the main recomputation and that its predeclared reading
holds on the stored intervals.

## 4a. Output-content audit (needs the clone audio, which is not released)

`code/content_audit.py` transcribes every clone and its conditioning utterance with
faster-whisper large-v3-turbo (CPU, int8; `EXP205_RUN` and `FASTER_WHISPER_TURBO`
point at the run directory and the model) and records, per clone, the word error rate
against the requested text and the longest run of consecutive recognized words shared
with the conditioning transcript **after excluding sequences present in the requested
text**. A clone is flagged when that filtered run reaches four words.
`data/transcripts.jsonl` holds the recognized transcripts and `data/content_audit.json`
the per-clone values and summary (0 of 3,456 flagged, median WER .04); `verify.py`
recomputes every WER and overlap value from the released transcripts. This is a
screen for copied text, not proof that no clone repeats reference audio.

## 4. Re-measure the microphone-channel control (needs VCTK audio)

`code/channel_distinctness.py` re-derives the channel control from the audio
itself rather than from the released result. It reads the file paths in
`data/selection_manifest.json`, which are relative to `inputs/`, so it needs a
local VCTK v0.92 copy placed as `inputs/vctk/wav48_silence_trimmed/...`. The
released `data/channel_distinctness.json` records what it produced.

```bash
uv run --frozen --with soundfile python code/channel_distinctness.py
```

## 5. Regenerate the clones (GPU, hours)

Not required to check any number in the paper. It needs the third-party cloning
systems and the pinned model revisions recorded in `data/generation_config.json`
and described in `DATA.md`, plus a local copy of VCTK. See `code/run.sh`.

## 6. Verify manuscript source/PDF coherence (CPU, seconds)

With `pdflatex`, `bibtex`, `pdfinfo` and `pdftotext` installed:

```bash
uv run --frozen python code/verify_manuscript.py
```

This builds in a temporary directory, accepts four pages or five with references
only on the fifth, rejects unresolved
references/citations and overfull boxes, and compares the clean build's extracted text
with the released PDF.

## What is not redistributed

Source and generated speech, model weights, and large environments. The
score-based statistics and decision criteria can be re-derived from the released
score-level files; the microphone-channel control needs the identified VCTK audio
(section 4), and release consistency does not authenticate historical execution.

## Refresh licensing evidence (network, not needed for verification)

`licenses/` already contains the authenticated snapshots. To prove that the
pinned upstream URLs still return the exact recorded bytes:

```bash
python3 code/fetch_license_snapshots.py
```

The fetcher refuses upstream drift before replacing any snapshot. Offline
verification never downloads anything.
