# Selection provenance

This directory closes the public history of the EXP-205 roster and pair
selection. It was assembled for publication after the result; it is not an
external timestamp or a retrospective preregistration.

- `build_tier1_gate.historical.py` is the source that selected the
  lexicographically first 30 of 105 speakers passing the earlier ECAPA gate.
- `build_tier2_pairs.historical.py` is the source that searched the remaining
  speakers with ECAPA geometry. Only 24 speakers met its low/high-dose and
  duration criteria, below its cap of 25, so all qualifiers were consumed. The
  original Tier-2 output hash remains in the selection manifest; the copy under
  `source/history/` has only portable logical paths.
- `feasibility.historical.py` is byte-identical to the source hash already
  recorded in `manifest.portable.json`. It defines EXP-205 as the 54-speaker
  complement and selects A/B pairs using only duration, transcript-byte rate,
  and lexicographic tie-breaking within that complement.
- `feasibility.portable.json` is the historical feasibility output with only
  its absolute run-root prefix replaced by `inputs/`. Its embedded ledger binds
  the historical source hash; it does not pretend the portable rendering was
  frozen before the result.

Run `verify_selection_provenance.py` to confirm the complete 30 + 24 + 54
ancestry, the absence of roster overlap, and all 54 selected A/B pairs. Raw
VCTK and the historical EXP-203 feature cache are needed only to rerun the
builders, not to verify the released records.
