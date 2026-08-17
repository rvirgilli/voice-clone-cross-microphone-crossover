# EXP-205 amendment 1 — F5 logical checkpoint path

Frozen **2026-08-17**, after job 4 failed in F5 model construction and before
any clone, feature, score or scientific result existed.

## Outcome-blind failure

The first audited execution verified all pins and built the Seed-VC job
manifest, then stopped at `F5TTS(...)` before generating its first clone. The
durable status is `INFRASTRUCTURE_FAILURE` at `generate_f5`; the clone-file
census is exactly zero.

The corrected pin builder preserved the logical `.safetensors` path in the F5
model-file record, but the separate `checkpoint_path` field consumed by the F5
API still used `f5_checkpoint.resolve()`. That produced a suffixless Hugging
Face blob path. F5 dispatches checkpoint loading by suffix, so it treated the
safetensors bytes as a torch pickle and failed closed.

## Delimited correction

This amendment authorizes only:

1. construct the F5 model pin once and set `checkpoint_path` to that exact
   pin's logical `path`;
2. require at runtime that `checkpoint_path` equals one unique model-file pin
   and retains the `.safetensors` suffix before `F5TTS` construction;
3. add a symlink fixture proving the logical path passes and the resolved blob
   target is rejected.

No manifest, clone roster, RNG, system, text, prompt arm, readout, statistic,
bar or verdict branch changes.

## Corrected pins

- pin builder:
  `b8fb3a88cd39ed85f9bd74d5b689f63fc6013fe90de5746e645a296ab1439b64`;
- three-backend generator:
  `261384c30b5398818865893aaaf76247cf6b9dca2ff2dc3c323a6a8fef567e8f`;
- generation tests:
  `688515ddb9f628cb8e7deb53e58268dab58eb14b9ea895c7b6130d0866c70c31`;
- successor execution config:
  `c82586074c5e7ec6aad9b21a101968f213f4849f35624492400041be1d2bd294`.

The full outcome-blind suite passes **26/26** and `verify_pins.py` rehashes 49
model/ref files plus 15 source files under the successor config. Execution
remains prohibited until the same independent auditor reproduces this exact
closure and authorizes the successor trust root.
