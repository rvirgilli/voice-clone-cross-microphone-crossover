# Extensions (paper §4.1, §4.3 and §5)

These scripts document the four post-hoc, descriptive analyses run on 2026-09-09 after
the frozen crossover, the score export of the clone-to-clone crossover, and the fresh-pair
replication run on 2026-09-10 (the crossover repeated, with readouts and decision rule
fixed, on a second outcome-blind A/B pair for every speaker that admits one). The post-hoc
analyses are not part of the release verifier's reproduction path: they need the original
clone audio, the VCTK captures and GPU embedding extraction.
`ICASSP_RUNS` and `ICASSP_EXPERIMENTS` point them at copies of the private run
directories and experiment tree; `HF_HUB_CACHE` at the model cache.

| Analysis | Script(s) | Released comparison-level table | Released summary |
|---|---|---|---|
| N-candidate scaling (§4.3) | `ncandidate.py` (select / extract / analyze) | `data/ncandidate_scores.tsv` | `data/ncandidate_result.json` |
| Second-generation cloning (§4.3) | `build_jobs_gen2.py`, `gen_jobs.py`, `analyze_gen2.py` | `data/second_generation_scores.tsv` | `data/second_generation_result.json` |
| Prompt interventions (§5) | `manipulate_prompts.py`, `build_jobs_intervention.py`, `gen_jobs.py`, `analyze_intervention.py` | `data/intervention_scores.tsv` | `data/intervention_result.json` |
| Readout extraction shared by the above | `extract_readouts.py` | — | — |
| Readout roster (§5) | `extract_readouts.py`, `analyze_readouts.py` | — (features are not released) | `data/readout_roster.json` |
| Table export | `export_scores.py` | writes the three tables | — |
| Clone-to-clone crossover (§4.1) | `export_clone_to_clone_scores.py` | `data/clone_to_clone_scores.tsv` | `data/extension_result.json` |
| Fresh-pair replication | `build_fresh_manifest.py`, `build_jobs_fresh_pair.py`, `gen_jobs.py`, `analyze_fresh_pair.py`, `run_fresh_pair.sh`, `export_fresh_pair_scores.py` | `data/fresh_pair_scores.tsv` | `data/fresh_pair_result.json`, `data/selection_manifest_fresh.json` |

Every printed point in the paper recomputes from the released tables by speaker-weighted
averaging (see `code/verify.py`); the bootstrap intervals regenerate from the same rows
with the seeds recorded in the summaries. The clone-to-clone export needs only the frozen
input manifest and the EXP-205 embedding cache it names (no audio or GPU); it writes one
row per comparison and readout with the query and both candidate clones identified by
speaker, system, text, prompt microphone, seed arm and clone hash prefix, and
`code/extension_verify.py` rebuilds the per-speaker accuracies and margins of
`data/extension_result.json` from it. `gen_jobs.py` re-uses the frozen EXP-205
generators, pins and per-clone ledgers (`code/generate.py`, `code/generate_seedvc.py`);
generated audio is not released.
