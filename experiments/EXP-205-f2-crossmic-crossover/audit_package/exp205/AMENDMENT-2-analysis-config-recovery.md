# Amendment 2 — analysis-config structural recovery

Date: 2026-08-17 (America/Sao_Paulo), after scoring and before opening any
scientific result.

## Observed failure

The one-job campaign authenticated and scored the complete frozen census, then
the analyzer stopped fail-closed with exit code 2 and
`ANALYSIS_MANIFEST_PIN_MISMATCH`.  The sealed result contained only:

```json
{
  "status": "INFRASTRUCTURE_FAILURE",
  "reason": "ANALYSIS_MANIFEST_PIN_MISMATCH"
}
```

No scientific value was printed or inspected.  The relevant frozen artifacts
at failure were:

- execution config: `c82586074c5e7ec6aad9b21a101968f213f4849f35624492400041be1d2bd294`;
- manifest: `4b879491f02badf252365aa4d2b3caa22402c04301c60ed5e02bd06d43f19b2d`;
- execution receipt: `aae8c2d3dda873f4f72ae2458d8983b5e9a3b2328c4d28b452530e7ead8e55ad`;
- score table: `fe3633f063ab8be6716553ad6bda3d311e97c6351d25dfd5735600367fb9c54e`;
- generated analysis config: `b7289bd5742831c9cbcb328fd72e64e2416a7d5cbec54dea07dd42bc023caf46`;
- infrastructure-only sealed result: `e72cdc8c50de466fcf725c1f8ba25fd0ddf49d17067272eaf5e27801c77b4299`;
- execution status: `edeae8c84694fea650bb5002407d6c4c131575a3ed7f1ed064d45b58c937ce5c`.

## Root cause

`score.py` constructed `analysis_config["manifest"]` from only `path` and
`sha256`.  The already-frozen execution config represents the same regular
manifest with `path`, identical `resolved_path`, and `sha256`.  The analyzer
correctly requires exact object equality, so the two authenticated hashes were
identical but the structural objects differed by the redundant
`resolved_path` field.

This is a transport/configuration defect.  It does not change the manifest,
the census, audio, clone ledgers, embeddings, scores, statistical procedure,
bars, random seed, or branch logic.

## Recovery authorized by this amendment

Run the already-pinned analyzer exactly once using
`analysis-config-recovery.json`.  That file differs from the generated config
only by copying the frozen execution config's manifest object verbatim; all
artifact paths and hashes are unchanged.  The analyzer must continue to
authenticate the execution config, its own source, the verdict source,
manifest, receipt, score table, complete census, and receipt contract before
parsing scientific values.

The recovery is CPU-only.  It must not rerun generation, feature extraction,
or scoring.  Before execution, an independent read-only auditor must verify
the exact diff, hashes, and that no scientific result has been used to choose
this repair.  After execution, a separate result audit must independently
authenticate and recompute the sealed result before anyone interprets it.

For future campaigns, `score.py` should copy `config["manifest"]` verbatim
when constructing its analysis config, with a regression test.  That source
change is deliberately deferred until after this frozen run is recovered so
the current source pins remain valid.
