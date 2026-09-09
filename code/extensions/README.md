# Post-hoc extensions (paper §4.3 and §5)

These scripts document the four post-hoc, descriptive analyses run on 2026-09-09 after
the frozen crossover. They are not part of the release verifier's reproduction path:
they need the original clone audio, the VCTK captures and GPU embedding extraction.
`ICASSP_RUNS` and `ICASSP_EXPERIMENTS` point them at copies of the private run
directories and experiment tree; `HF_HUB_CACHE` at the model cache.

| Analysis | Script(s) | Released comparison-level table | Released summary |
|---|---|---|---|
| N-candidate scaling (§4.3) | `ncandidate.py` (select / extract / analyze) | `data/ncandidate_scores.tsv` | `data/ncandidate_result.json` |
| Second-generation cloning (§4.3) | `build_jobs_gen2.py`, `gen_jobs.py`, `analyze_gen2.py` | `data/second_generation_scores.tsv` | `data/second_generation_result.json` |
| Prompt interventions (§5) | `manipulate_prompts.py`, `build_jobs_intervention.py`, `gen_jobs.py`, `analyze_intervention.py` | `data/intervention_scores.tsv` | `data/intervention_result.json` |
| Readout extraction shared by the above | `extract_readouts.py` | — | — |
| Table export | `export_scores.py` | writes the three tables | — |

Every printed point in the paper recomputes from the released tables by speaker-weighted
averaging (see `code/verify.py`); the bootstrap intervals regenerate from the same rows
with the seeds recorded in the summaries. `gen_jobs.py` re-uses the frozen EXP-205
generators, pins and per-clone ledgers (`code/generate.py`, `code/generate_seedvc.py`);
generated audio is not released.
