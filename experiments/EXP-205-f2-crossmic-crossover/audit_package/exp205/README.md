# EXP-205 portable audit package

This package reproduces the paper-facing EXP-205 score census, pooled and
per-system statistics, 100,000 whole-speaker bootstrap intervals, and frozen
verdict. It contains no source audio, generated audio, model weights, or
playable impersonations.

Run the verifier from the repository root with the locked environment:

```bash
uv run --frozen python experiments/EXP-205-f2-crossmic-crossover/audit_package/exp205/verify_exp205_package.py
```

The script declares and pins its sole third-party dependency, NumPy 2.4.6,
using PEP 723 inline metadata. Expected final output:

```text
PASS — portable EXP-205 census, provenance, statistics and verdict verify
```

`scores.portable.tsv`, the `*.portable.json` records, and `result.json` use
logical paths and are the intended portable public records. Files under
`source/` are hash-pinned release copies of the analysis and execution source.
Machine-specific defaults were replaced by environment variables, and
historical roster paths were mapped to logical paths; these public adaptations
do not alter the sealed result or the original trust-root hashes. The portable
verifier does not invoke generation or scoring code.

The post-review `roster_ancestry_sensitivity.*` and
`arm_pairing_sensitivity.*` files are descriptive checks added after the frozen
result. Their status fields explicitly prohibit changing or rescuing the
pre-registered verdict. They can be reproduced in place with:

```bash
uv run --frozen python experiments/EXP-205-f2-crossmic-crossover/audit_package/exp205/roster_ancestry_sensitivity.py
uv run --frozen python experiments/EXP-205-f2-crossmic-crossover/audit_package/exp205/arm_pairing_sensitivity.py
```

The exact hash-pinned historical roster records needed by the first command are
under `source/history/`.

`audit.json` authenticates every package component. The score and source-file
licenses are summarized in `LICENSES.md`; model code and weights are not
redistributed here.
