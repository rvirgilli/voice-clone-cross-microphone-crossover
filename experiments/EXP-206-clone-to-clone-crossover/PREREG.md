# EXP-206: cross-generator, cross-text clone-to-clone crossover

Status: frozen before EXP-206 analysis. No EXP-206 similarity, follow-rate,
interval, or verdict had been computed when this protocol and its input
manifest were frozen.

## Question

Does the conditioning-event signal observed in EXP-205 remain detectable when
the query and both candidates are generated clips, while query and candidates
use different cloning systems, different output texts, and conditioning
recordings captured through opposite VCTK microphones?

This is one bounded upgrade to the closed F2 paper. It uses the already sealed
3,456 EXP-205 clones and the two already fixed speaker-verification readouts.
It generates no audio, trains no model, and introduces no new speaker, system,
text, checkpoint, encoder layer, or outcome-based selection.

## Fixed comparison

For every speaker, direction, query event arm, query system, query text,
candidate system, and candidate text:

- the candidate system must differ from the query system;
- the candidate text must differ from the query text;
- a mic1 query is compared with mic2 candidates, and vice versa;
- both candidates share the same candidate system and text;
- the correct candidate shares the query's conditioning event arm;
- the incorrect candidate uses the other same-speaker event arm.

The readout compares cosine similarities. A strict correct ordering scores 1,
a strict incorrect ordering 0, and an exact tie 0.5. Each speaker contributes
exactly `2 arms × 4 query systems × 4 query texts × 3 other systems × 3 other
texts = 288` comparisons per direction and readout. The speaker is the only
resampling unit; pairwise comparisons are not treated as independent.

The primary grid is fixed before analysis:

- directions: mic1 query to mic2 candidates; mic2 query to mic1 candidates;
- readouts: the exact EXP-205 ECAPA and WavLM-SV embeddings;
- systems: F5-TTS, XTTS-v2, CosyVoice2, and Seed-VC;
- texts: all four fixed EXP-205 generated texts;
- speakers: all 54 fixed EXP-205 evaluation speakers.

Same-system, same-text, real-candidate, selected-system, selected-text, or
selected-speaker analyses cannot rescue the primary result.

## Estimator and stability interval

For each speaker, direction, and readout, average the 288 follow scores. The
fixed-roster point is the equally weighted mean of the 54 speaker means. A
central 95% speaker-bootstrap stability interval uses 100,000 resamples of 54
speakers with replacement and NumPy `default_rng(2062027)`. One shared index
matrix is used for all four primary cells.

These intervals describe composition stability within the fixed roster; they
are not population-coverage confidence intervals.

## Frozen verdict

Let `L(d,r)` and `P(d,r)` denote the lower interval endpoint and point for
direction `d` and readout `r`.

`EVENT_SIGNAL_PRESENT` requires the intersection of all four one-sided checks:

- `L(primary, ECAPA) > .50`;
- `L(reverse, ECAPA) > .50`;
- `L(primary, WavLM-SV) > .50`;
- `L(reverse, WavLM-SV) > .50`.

Because all four checks must pass, this is an intersection-union claim; no
multiplicity correction is used to rescue a failed component.

`MATERIAL_EVENT_SIGNAL` additionally requires both ECAPA points to be at least
.60 and both WavLM-SV points to be at least .55. Only this material state may
support a new abstract/conclusion headline. A four-cell positive result below
those point bars may be reported as limited supporting evidence, but not as a
material upgrade. Any failed lower-bound component is `NOT_CONFIRMED`; partial
or subgroup success cannot replace it.

System-pair points may be emitted as descriptive diagnostics only after the
primary result is sealed. They receive no interval, no verdict, and no rescue
role.

## Interpretation boundary

A material result would show that event-linked similarity crosses generator,
output-text, stochastic-seed, and microphone changes under two fixed SV
readouts. It would not identify the physical carrier, prove open-set presence
detection, establish population inference, or make the two readouts
independent replications.

A null result does not invalidate EXP-205's real-candidate crossover. The
paper-base remains at its already closed commit, and EXP-206 may only narrow
the boundary or remain a registered negative result.

## Outcome firewall and stop rule

Before freeze, permitted operations are file census, byte hashing, schema and
identity validation, cache hashing without loading embedding arrays, synthetic
tests, and runtime estimation from metadata. The EXP-205 score table and
scientific result are not inputs to EXP-206.

After one independently reviewed freeze, run the authenticated analyzer once.
Fixes after execution are limited to reproducible infrastructure defects and
must preserve the estimator and verdict. No alternative representation,
within-system fallback, same-text fallback, threshold search, subgroup claim,
or second scientific upgrade enters this paper.
