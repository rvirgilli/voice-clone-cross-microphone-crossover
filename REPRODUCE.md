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
- the follow rates and their 100,000-draw whole-speaker bootstrap intervals,
  for both directions and both speaker-verification readouts;
- the generation ledger, covering all 3,456 clones;
- the microphone-channel control, which confirms that the two captures of each
  event are distinct signals rather than copies of one another;
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
```

Both rewrite their result files in `data/`; the recomputed values must match the
committed ones.

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

This builds in a temporary directory, requires exactly four pages, rejects unresolved
references/citations and overfull boxes, and compares the clean build's extracted text
with the released PDF.

## What is not redistributed

Source and generated speech, model weights, and large environments. Everything
the paper asserts can be re-derived from the released score-level files.

## Refresh licensing evidence (network, not needed for verification)

`licenses/` already contains the authenticated snapshots. To prove that the
pinned upstream URLs still return the exact recorded bytes:

```bash
python3 code/fetch_license_snapshots.py
```

The fetcher refuses upstream drift before replacing any snapshot. Offline
verification never downloads anything.
