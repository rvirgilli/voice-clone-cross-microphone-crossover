# EXP-205 licence and release boundary

Checked: 2026-08-17. This is a reproducibility inventory, not legal advice.
The experiment is non-commercial academic research. It will not redistribute
model weights, VCTK/LibriTTS-R source audio, or generated voice samples. The
releaseable artifact is limited to code, hashes, aggregate/per-speaker numeric
results and provenance metadata. A separate human review is required before
any generated audio is released, regardless of the licences below.

| Component | Code licence | Model/data licence | Evidence used for this run |
|---|---|---|---|
| F5-TTS 1.1.22 | MIT (`f5_tts-1.1.22.dist-info/METADATA`) | F5-TTS checkpoint model card: CC BY-NC 4.0 | <https://huggingface.co/SWivid/F5-TTS> |
| Coqui-TTS 0.25.3 / XTTS-v2 | MPL-2.0 (`coqui_tts-0.25.3.dist-info/METADATA`) | Coqui Public Model License; non-commercial use only without a separate commercial licence | <https://docs.coqui.ai/en/latest/models/xtts.html#license> and the pinned local `TTS/.models.json` |
| CosyVoice2 | Apache-2.0 | Apache-2.0 | pinned checkout `LICENSE`; pinned snapshot `README.md` (`license: apache-2.0`) |
| Seed-VC | GPL-3.0 | GPL-3.0 model card | pinned checkout `LICENSE`; <https://huggingface.co/Plachta/Seed-VC> |
| SpeechBrain ECAPA, `speechbrain/spkrec-ecapa-voxceleb` | Apache-2.0 | Apache-2.0 | model-card licence and pinned revision `0f99f2d0ebe89ac095bcc5903c4dd8f72b367286`; <https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb> |
| Microsoft WavLM, `microsoft/wavlm-base-plus-sv` | MIT | MIT | official model card points to the UniLM licence; pinned revision `feb593a6c23c1cc3d9510425c29b0a14d2b07b1e`; <https://huggingface.co/microsoft/wavlm-base-plus-sv> and <https://github.com/microsoft/unilm/blob/master/LICENSE> |
| VCTK v0.92 | n/a | CC BY 4.0 | corpus DOI <https://doi.org/10.7488/ds/2645> and its `license_text.txt` |
| LibriTTS-R SLR141 | n/a | CC BY 4.0 | <https://www.openslr.org/141> |

The ECAPA and WavLM readouts are used only as fixed feature extractors and are
not redistributed. Their exact snapshots and file hashes are part of the
execution config; their model-card licences and pinned revisions are recorded
above.

Licensing a corpus does not by itself settle voice/privacy expectations for
every downstream use. Accordingly, EXP-205 reports research measurements and
does not publish playable impersonations or claim consent for deployment.
