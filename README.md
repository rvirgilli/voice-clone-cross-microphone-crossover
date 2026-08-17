# Cross-microphone voice-clone attribution

This repository accompanies **“Which Conditioning Recording Does a Voice
Clone Follow? A Balanced Cross-Microphone Crossover.”** It contains the exact
release-candidate manuscript, frozen score-level artifacts, contemporaneous
protocol records and amendments, released analysis source, and executable
checks that bind the paper's claims to those records.

## Verify the release

Python 3.11 or newer and [`uv`](https://docs.astral.sh/uv/) are required.

```bash
uv sync --frozen
uv run --frozen python verify_release.py
```

The release verifier authenticates every versioned file, recomputes the
3,456-row EXP-205 census and its 100,000-bootstrap summaries, verifies the full
108-speaker ancestry and all 54 selected A/B pairs, reproduces the frozen
EXP-206 clone-to-clone points, intervals and verdict, checks the manuscript
against EXP-202/204/205/206 records, and runs mutation tests against the
paper-facing checker. It does not generate or score audio.

## Repository map

- `paper/F2/`: exact PDF, LaTeX source, bibliography, figure source, and checks.
- `CLAIM-SCOPE.md`: fixed scientific thesis, provenance boundary, exclusions,
  and stop rule.
- `experiments/EXP-202-f2-campaign/`: antecedent same-speaker results used in the paper.
- `experiments/EXP-204-f2-seed-crossover/`: earlier two-seed crossover result.
- `experiments/EXP-205-f2-crossmic-crossover/`: final crossover, sensitivities, and portable audit package.
- `experiments/EXP-206-clone-to-clone-crossover/`: frozen cross-generator,
  cross-text extension, result, source, and public verifier.
- `DATA.md`: data, model, licence, and redistribution boundaries.
- `REPRODUCE.md`: reproduction levels and exact commands.
- `MANIFEST.json`: SHA-256 and byte size of every released file.

## Reproduction boundary

All headline points, intervals, tables, sensitivities, selection checks, and
claim-to-artifact bindings reproduce from committed files on CPU. Source and
generated speech, model weights, and large environments are not redistributed.
Regenerating the 3,456 clones requires the third-party assets and pinned model
revisions described in `DATA.md`; it is not required to verify the paper.

## Licence

Repository code is MIT licensed. The paper, datasets, models, checkpoints, and
third-party implementations retain their original terms. See `DATA.md` and the
EXP-205 package's `LICENSES.md`.
