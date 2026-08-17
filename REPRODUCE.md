# Reproduction guide

## Level 1 — paper from frozen public artifacts

This level is CPU-only and uses only versioned repository files.

```bash
uv sync --frozen
uv run --frozen python verify_release.py
```

For individual checks:

```bash
uv run --frozen python experiments/EXP-205-f2-crossmic-crossover/audit_package/exp205/verify_exp205_package.py
uv run --frozen python experiments/EXP-205-f2-crossmic-crossover/audit_package/exp205/source/selection/verify_selection_provenance.py
uv run --frozen python experiments/EXP-205-f2-crossmic-crossover/audit_package/exp205/verify_public_history.py
uv run --frozen python paper/F2/check_numbers.py
uv run --frozen python paper/F2/mutation_test.py
```

Expected final lines are `PASS` from both the result and selection packages,
`PASS` from the paper checker, and `UNCAUGHT: 0  INVALID: 0` from mutation
testing.

## Level 2 — post-result sensitivities and figure

The sensitivity scripts authenticate their inputs before recomputing outputs:

```bash
uv run --frozen python experiments/EXP-205-f2-crossmic-crossover/audit_package/exp205/roster_ancestry_sensitivity.py
uv run --frozen python experiments/EXP-205-f2-crossmic-crossover/audit_package/exp205/arm_pairing_sensitivity.py
uv run --frozen python paper/F2/make_exp205_figure.py
```

Run them in a disposable clone when comparing regenerated files with the
canonical release. Neither sensitivity is allowed to alter the sealed verdict.

## Level 3 — generation and scoring

Audio generation and scoring require the third-party corpora, repositories,
model snapshots, and environments described in `DATA.md`. Set the documented
`EXP205_*` environment variables, construct a fresh execution config, verify
its pins, and only then invoke the released runner. Generated audio must remain
outside this repository.

The public package retains the original result trust roots while using logical
paths and environment variables in its released source. A new inference run is
a new execution with new hashes; it must not be presented as the sealed run.
The runnable source directory contains the logical `selection-manifest.json`,
`LICENSES.md`, and `run.sh` names expected by the builder and runner. From that
directory, after configuring every `EXP205_*` path described in `DATA.md`:

```bash
uv run python build_execution_config.py --out execution-config.json
uv run python verify_pins.py --config execution-config.json
sha256sum execution-config.json
bash run.sh THE_PRINTED_64_HEX_SHA256
```

This full command regenerates and scores 3,456 clones and is intentionally not
part of the CPU-only release gate. Use a writable fast work volume for
`EXP205_RUN`; never write generated audio into this Git checkout.

## Manuscript build

With a TeX installation that provides `pdflatex`, `bibtex`, and `pdfinfo`:

```bash
bash paper/F2/build.sh
```

The submitted PDF is already committed and authenticated by `MANIFEST.json`.
