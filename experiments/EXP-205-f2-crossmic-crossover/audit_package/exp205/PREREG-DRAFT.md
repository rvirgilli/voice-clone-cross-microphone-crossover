# EXP-205 frozen preregistration — held-out cross-microphone seed crossover

Date drafted: 2026-08-17

Status: **FROZEN / AMENDMENT 1 CLOSED FOR THE EXACT SUCCESSOR JOB.** The
outcome-blind selection builder, manifest, generators,
per-clone ancestry ledgers, authenticated readouts/scorer, receipt-bound
analyzer, verdict truth table, licence/release boundary and synthetic
census/mutation suite are frozen by execution config SHA-256
`c82586074c5e7ec6aad9b21a101968f213f4849f35624492400041be1d2bd294`.
Read-only integrity checks closed the logical-snapshot/ref blocker. The first
execution then exposed the outcome-blind F5 path failure recorded in
`AMENDMENT-1-f5-logical-checkpoint.md`; zero clones existed. The suffixless-blob
failure was reproduced and the successor trust root was closed before rerun.

Only the exact trust root above satisfied the technical execution gate. Any
source, manifest, model mapping, ref or config change required a new integrity
verification.

## Question and paper-changing claim

When two same-speaker recordings are both plausible prompts, does a clone
preferentially identify the conditioning recording event after the real
candidate is replaced by a simultaneous recording from another microphone?

This is stronger than placing a clone near its selected seed in a closed pool.
It combines an intervention (change the seed within speaker) with a pickup-path
control (generation consumes mic 1; retrieval sees mic 2). A positive result
cannot be explained by speaker identity, generated text, exact waveform, or an
exact microphone fingerprint shared by prompt and candidate.

“Recording event” is deliberate. A positive result may be carried by prosody,
speaking rate, background/session acoustics or another recording-specific
trace. The primary result alone does not identify which mechanism carried it.

The scoped claim is about four fixed open systems and the eligible VCTK roster.
It is not a claim about all voice-cloning systems or natural conversational
speech.

## Independence boundary

EXP-203 used 108 VCTK speakers with one generation seed each. EXP-204 crossover
tiers used 54 of those speakers with new two-seed designs. The remaining 54 are
therefore **crossover-held-out, not globally unseen speakers**.

For each of the 54:

- no EXP-204 clone exists (filesystem census is zero);
- neither member of the new pair may be the exact utterance that seeded that
  speaker's EXP-203 clone;
- neither member may be selected with an embedding, score, clone outcome, or
  downstream retrieval statistic;
- all 54 speakers remain in the analysis. There is no post-selection exclusion
  for difficult or weakly separated pairs.

`feasibility.py` currently verifies this boundary using only filenames, real
audio headers, transcripts and prior experiment rosters. It finds 54/54
feasible speakers after excluding the earlier EXP-203 seed. Under the
bidirectional selector, the worst selected-pair gap across either microphone is
0.112 s and 2.571% in seconds per transcript byte.

## Draft selection rule

For each fixed crossover-held-out speaker:

1. enumerate paired VCTK mic1/mic2 utterances lasting 4--10 s;
2. exclude the exact EXP-203 generation seed;
3. retain pairs whose absolute within-microphone duration difference is at most
   0.25 s and whose relative difference in seconds per UTF-8 transcript byte is
   at most 5%, on both mic1 and mic2;
4. choose deterministically by the smallest maximum normalized gap
   across both microphones, then the smallest sum of all four normalized gaps,
   maximum rate gap, maximum duration gap, path A and path B.

The transcript-rate control is fixed because the F5 harness exposes a known
duration-per-text cue. Choosing the closest metadata pair is intentionally
conservative. No SV-distinguishability gate is allowed: the result must be
unconditional over the complete 54-speaker roster, including hard pairs.

The frozen `selection-manifest.json` records both mic paths,
transcripts, durations, rate gaps, the excluded prior seed, corpus/version
identity and SHA-256 hashes. It contains 54 speakers and binds 216 real VCTK
files plus the four fixed LibriTTS-R Seed-VC source files. Its current SHA-256
is `4b879491f02badf252365aa4d2b3caa22402c04301c60ed5e02bd06d43f19b2d`;
the builder SHA-256 is
`23c545e5ddafe4034251f016052a7a57bf77f8d3111a012a074debd03941141b`.
These are immutable execution pins under the audited config above.

## Generation and comparison

- Systems: F5-TTS, XTTS-v2, CosyVoice 2 and Seed-VC, with the exact pinned
  environments/checkpoints used by EXP-203/204.
- F5-TTS must pin package `1.1.22` and the inference source that computes output
  duration from `ref_audio_frames / len(ref_text.encode("utf-8"))`; the source
  hash must accompany the generation receipt. A floating `--with f5-tts`
  environment is not admissible.
- Prompt arms: seed A and seed B from mic1, plus the same two performances from
  mic2. The primary pickup direction is mic1 prompt to mic2 candidates; the
  reverse mic2 prompt to mic1 candidates is a predeclared directional
  replication and cannot rescue the primary.
- VCTK v0.92 documents mic1/mic2 as two physically different microphones in
  the same fixed recording setup (DPA 4035 omni and Sennheiser MKH 800
  condenser, recorded in a hemi-anechoic chamber; DOI
  [10.7488/ds/2645](https://doi.org/10.7488/ds/2645)). The paired-file and
  channel identities must be bound in the manifest rather than inferred only
  from filename replacement. The corpus record flags MKH 800 technical issues
  for p280 and p315; neither speaker belongs to the 54-speaker holdout.
- Generated texts: the same four fixed, seed-transcript-disjoint LibriTTS-R
  texts used by EXP-202--204. Keeping them fixed makes the new result directly
  comparable while generated content remains identical across A/B.
- Generation RNG: one fixed per-text seed shared across A/B and speakers for
  every backend that exposes it; deterministic settings and exceptions logged.
- Workload: 54 speakers x 2 seeds x 2 prompt microphones x 4 texts x 4 systems
  = **3,456 clones**.
- Primary comparison: for a clone generated from mic1 seed X, compare the mic2
  counterparts of seed A and seed B and score whether X ranks above the other.
- Tie: 0.5, declared before outcomes.

Within each pickup direction the design is a complete two-arm crossover. For
every fixed system/text cell it includes both signed comparisons, shown here
for the primary direction,

`I[s(clone_A, mic2_A) > s(clone_A, mic2_B)]` and
`I[s(clone_B, mic2_B) > s(clone_B, mic2_A)]`,

with 0.5 for ties. Averaging the two seed arms prevents a candidate that is
intrinsically favoured by an encoder from creating seed following by itself.
The analyzer will also report the continuous antisymmetric similarity margin
as a diagnostic, but it cannot change the binary primary verdict.

Every speaker contributes 32 dependent comparisons per pickup direction, 64 in
total. The analysis unit is the speaker, not the clone. Systems, texts, prompt
microphones and arms are fixed crossed factors.

## Fixed readouts under consideration

The protocol fixes this roster unchanged unless a scientific revision is
recorded before freeze:

1. **Primary:** cosine similarity from the same fixed ECAPA-TDNN SV encoder used
   throughout F2. This was named primary before the cross-mic discovery screen
   was opened because the claim concerns a same-speaker recording contrast.
2. **Second fixed SV readout:** the SV-fine-tuned WavLM encoder already used
   throughout F2.
3. **Diagnostics only:** the predeclared four-family fusion, five-family
   fusion, handcrafted no-SV families and same-mic candidate counterparts.

No method is chosen by whichever produces the best held-out result. Diagnostics
cannot rescue a failed primary or second-readout conjunct.

## Candidate estimand and intervals

For encoder `e` and pickup direction `d`, compute each speaker's mean follow
value over that direction's 32 fixed comparisons, then average the 54 speaker
means with equal speaker weight. Directions are never pooled for a verdict.

- 95% speaker-resampling stability interval: percentile bootstrap of speakers,
  `B=100,000`, fixed RNG,
  resampling whole speaker vectors so encoders/systems/texts remain paired.
- Report the 54 per-speaker means and all four system-specific point estimates.
- System intervals are descriptive and simultaneous (max-t speaker bootstrap)
  or omitted; no uncorrected system-by-system significance claims.
- Report same-microphone candidates beside each cross-microphone direction as
  diagnostics, but they cannot enter either verdict axis.

Because the 54 speakers are a fixed crossover-held-out census rather than a
fresh probability sample, the interval quantifies stability to speaker
composition in this eligible roster; it is not presented as design-based
coverage for a universal speaker population.

The analyzer must verify the exact 3,456-file census, 54 speakers, 64
comparisons/speaker, two seed arms, two prompt microphones, four texts, four
systems, unique paths, input hashes and feature completeness before it reads
scores. Any gate failure yields
`INFRASTRUCTURE_FAILURE`, never a partial scientific result.

## Frozen scientific bars

These are the frozen primary mic1-to-mic2 bars:

- `E`: ECAPA 95% speaker-resampling stability lower endpoint strictly above
  `.50` (the positive direction is robust to speaker composition in the fixed
  roster).
- `O`: ECAPA point estimate at least `.80` **and** lower bound above `.70`
  (at least a 4:1 point ratio correct:incorrect, with the earlier `.70`
  smallest-operational-effect floor retained).
- `R`: WavLM 95% speaker-resampling stability lower endpoint strictly above
  `.50` (representation replication in an independently trained
  representation).

The operational headline requires the conjunction `E & O & R`: every component
must pass and no component can rescue another. This is a frozen decision rule,
not a formal population-level intersection-union test, because the bootstrap
endpoints are explicitly composition-stability summaries on a fixed eligible
roster rather than claimed-coverage confidence bounds. Exploratory
system-specific and diagnostic readouts receive simultaneous intervals or
explicit exploratory labels and never enter that conjunction.

The reverse mic2-to-mic1 direction has a separate, non-rescuing status:

- `D_E`: reverse ECAPA lower bound strictly above `.50`;
- `D_R`: reverse WavLM lower bound strictly above `.50`;
- `(1,1)` = `BIDIRECTIONAL_REPLICATION`, exactly one =
  `REVERSE_REPRESENTATION_DEPENDENT`, `(0,0)` = `REVERSE_NOT_CONFIRMED`.

Only a primary `OPERATIONAL_CROSSMIC_CONFIRMATION` together with
`BIDIRECTIONAL_REPLICATION` permits a bidirectional headline. Every other
combination retains the complete primary truth-table reading and reports the
reverse status without averaging directions.

The `.80` point bar is tied to a four-in-five operational reading, not to a
rounded discovery estimate. It is different from EXP-204's missed `.90` fused
bar because the primary representation and the unconditional cross-microphone
design are different. The manuscript must report both experiments' bars and
must not present the successor as retroactively repairing the old verdict.

## Complete draft truth table

| E | O | R | Draft reading | Allowed claim |
|---|---|---|---|---|
| 1 | 1 | 1 | `OPERATIONAL_CROSSMIC_CONFIRMATION` | Conditioning-recording identity is recoverable across pickup paths in the fixed four-system VCTK design; the carrier mechanism remains open. |
| 1 | 1 | 0 | `PRIMARY_ONLY` | Large ECAPA result, but encoder-general mechanism not confirmed. |
| 1 | 0 | 1 | `REPLICATED_WEAK_POSITIVE` | Cross-encoder effect exists but is smaller than operational bar. |
| 1 | 0 | 0 | `WEAK_PRIMARY_ONLY` | ECAPA-only positive; narrow to representation-dependent evidence. |
| 0 | 0 | 1 | `SECONDARY_ONLY_NOT_CONFIRMED` | Primary fails; no confirmation regardless of WavLM. |
| 0 | 0 | 0 | `NOT_CONFIRMED` | Preserve current F2 and withdraw cross-mic crossover mechanism claim. |
| 0 | 1 | 0 | `INVALID_STATE` | Impossible because `O` includes `E`; analyzer/test failure. |
| 0 | 1 | 1 | `INVALID_STATE` | Impossible because `O` includes `E`; analyzer/test failure. |

No branch is named `REFUTED`: failure to clear a finite-sample bar is not proof
of the null. No pooled discovery+confirmation estimate is allowed.

The current outcome-independent implementation is `verdict.py` (SHA-256
`8b12a6b94366c1f7d08436397b83b1c64fadcb4085ca490cb9b7a593802380be`).
`test_verdict.py` (SHA-256
`69d38dc85386a2f88ea02e128045c781928a9be22dbd13e913d218e530687391`)
passes six tests covering all six reachable primary states, all four reverse
states, strict interval boundaries, the closed `.80` point bar, invalid
probabilities and the non-rescue rule. These are frozen pins; the analyzer
must authenticate the exact copy before using it.

The frozen `analyze.py` (SHA-256
`6ff83b3c9657b373a437446413cc704bf4c99ffa92074b274375e12b7d8174d6`)
performs a full identity-only pass before parsing any similarity, authenticates
the externally supplied execution-config hash, manifest, score table, receipt,
verdict bytes and every clone ledger, rehashes all 3,456 clone files and 216
candidate files, enforces the exact crossed census, then applies the speaker
bootstrap and truth table without printing outcomes. Its synthetic suite
`test_analyze.py` (SHA-256
`d5aa366cee2d97592f291c813beee46907197c6dd44196d8449e8b85ca931f7e`)
passes twelve tests, including all six reachable scientific branches through
the full analyzer and infrastructure mutations for missing/duplicate rows,
wrong candidates, speakers and microphones, receipt/config/pin/count swaps,
clone/ledger tampering and verdict-pin tampering. The full outcome-blind suite
passes 26 tests, additionally requiring exact four-text Seed-VC mapping, rejecting
orphan F5/Seed-VC WAVs, rejecting a reference-audio hash mismatch before resume
or generation, rejecting a retargeted logical snapshot symlink, and proving
that F5 consumes the pinned logical `.safetensors` path rather than its
suffixless blob target, and proving that a full feature cache cannot bypass
loader authentication. Seed-VC's exact
snapshot refs are pinned and its runner is offline. Every infrastructure mutation returns
`INFRASTRUCTURE_FAILURE` without a partial scientific payload. These pins and
the runner received independent PASS under the exact trust root above.

## Discovery-based risk screen

The post-hoc screen opens only the 54 EXP-204-consumed speakers and uses the
cross-microphone candidates already extracted by EXP-203. It exactly
reconstructs tier 1's fused mic1 result and obtains:

- fused mic2 `.790 [.743,.835]` in tier 1 and `.777 [.746,.807]` over both tiers;
- ECAPA mic2 `.917 [.893,.938]` over both tiers;
- WavLM-SV mic2 `.657 [.629,.684]` over both tiers;
- positive ECAPA direction in every system, including XTTS `.863`.

These are conditioned/dose-selected discovery pairs, not the new unconditional
pairs. Empirical paired-speaker resampling at `n=54` estimates passage of the
full draft conjunction at 100% under the observed distribution and 92.4% after
25% shrinkage of both encoder effects toward chance. At 50% shrinkage, existence
passes while the operational bar fails. This screen is useful for risk ranking;
it is not prospective power and cannot authorize execution.

No reverse mic2-prompt clone exists in discovery. The reverse direction is
therefore genuinely uncalibrated additional evidence: its success can upgrade
the interpretation to bidirectional, while its failure cannot alter or rescue
the frozen primary direction.

## Compute, storage and calendar

The closest measured end-to-end anchor is EXP-203 job 138: the same 3,456-clone
count, plus 10,800 real pool files, eight clone feature bundles and two pool
bundles, completed in **1 h 22 min 38 s** on the RTX 3090. EXP-205 reuses the
already extracted real pools. Use **1.4 GPU-h as a conservative upper anchor**
and reserve **1.6 h in the queue** including 15% contingency.

Existing EXP-204 artifacts imply about 0.35 MB/clone. Expected new clone audio
is roughly 1.2 GB and features below 0.2 GB; the active filesystem currently has
about 1.1 TB free. No running job may write to bigfour SMR.

After M1 releases the GPU, a valid frozen package can execute and analyze in one
day. Reserve two additional days for artifact audit, paper rewrite and exact-PDF
This is not on the deadline critical path unless a substantive protocol defect
is found.

## Satisfied GO package

Before GPU submission, all existed and passed recorded integrity checks:

1. append-only EXP-204 correction acknowledging its omitted `.90` conjunct;
2. exact hashed selection manifest and four generated-text hashes;
3. pinned generator/checkpoint/environment identities and licences;
4. resumable one-job/one-exit-code generator+extractor with silent outcome
   handling and an exact completion receipt;
5. sealed analyzer and exhaustive verdict truth-table tests;
6. synthetic alternatives that exercise every scientific branch, tie handling,
   missing/duplicate files, wrong speakers, wrong microphones and a swapped
   ancestry receipt;
7. a protocol-only integrity test suite returning PASS;
8. execution within the frozen resource boundary.

The historical execution command was:

```bash
experiments/EXP-205-f2-crossmic-crossover/run.sh \
  c82586074c5e7ec6aad9b21a101968f213f4849f35624492400041be1d2bd294
```
