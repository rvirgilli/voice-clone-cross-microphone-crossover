# Licences, model cards, and release boundary

Checked: 2026-08-28. This is a reproducibility inventory, not legal advice.
The experiment and this release are non-commercial academic research. Exact
textual evidence is in `licenses/`; `licenses/SNAPSHOT-MANIFEST.json` records
the immutable source URL and SHA-256 of every snapshot, and the offline release
verifier authenticates those bytes.

The release contains code, hashes, sanitized score-level data, aggregate and
per-speaker numeric results, and provenance metadata. It contains **no model
weights, source audio, generated audio, speaker embeddings, or playable
impersonations**.

| Component | Exact execution pin | Terms established by the staged snapshots | Release consequence |
|---|---|---|---|
| F5-TTS 1.1.22 | checkpoint SHA-256 `670900fd14e6c458b95da6e9ed317cdb20dbaf7a1c02ac06a05475a9d32b6a38`; model-card revision `84e5a410d9cead4de2f847e7c9369a6440bdfaca` | The pinned model card declares `cc-by-nc-4.0`; the complete CC BY-NC 4.0 legal code is snapshotted. The F5 code package is MIT. | The project conservatively treats F5-generated clone audio as CC BY-NC-encumbered and does not redistribute it. Commercial reuse is outside this release. |
| Coqui-TTS 0.25.3 / XTTS-v2 | model SHA-256 `c7ea20001c6a0a841c77e252d8409f6a74fb423e79b3206a0771ba5989776187`; config SHA-256 `ef262b1454dd2a77e1461b0b2cd53e19b8a7624cc131b837d36df67356bc75e8`; model-card/licence revision `6c2b0d75eae4b7047358e3b6bd9325f857d43f77` | CPML 1.0.0 explicitly permits only non-commercial use of the model **and its outputs**. Its notice clause requires recipients of any model part, modification, or output to receive the terms or their URL. The Coqui-TTS code package is MPL-2.0. | No XTTS audio is released. This academic run is non-commercial; any commercial use or output distribution must be separately reviewed and cannot rely on this package as permission. |
| CosyVoice2 | pinned checkout and model hashes in `data/generation_config.json` | Apache-2.0 checkout/model-card evidence recorded in the execution config. | No weight or generated-audio redistribution. |
| Seed-VC | pinned checkout and model hashes in `data/generation_config.json` | GPL-3.0 checkout/model-card evidence recorded in the execution config. | No weight or generated-audio redistribution. |
| SpeechBrain ECAPA readout | model revision `0f99f2d0ebe89ac095bcc5903c4dd8f72b367286`; exact file hashes in the execution config | The pinned model card declares Apache-2.0; the complete Apache 2.0 text is snapshotted. | Used only as a fixed feature extractor. Neither checkpoint nor embeddings are redistributed. |
| Microsoft WavLM readout | model revision `feb593a6c23c1cc3d9510425c29b0a14d2b07b1e`; exact file hashes in the execution config | The pinned model card calls the linked Microsoft UniSpeech licence “official.” The linked file is snapshotted at commit `6112826ac13a4327f4c9a7afa2a505e35b763514` and is **CC BY-SA 3.0**, not MIT. | Used only as a fixed feature extractor. Neither checkpoint nor embeddings are redistributed. The earlier MIT label is retired by this corrected inventory. |
| VCTK v0.92 | corpus DOI `10.7488/ds/2645` | CC BY 4.0 corpus licence. | Source recordings are not redistributed. |
| LibriTTS-R SLR141 | OpenSLR 141 | CC BY 4.0 corpus licence. | Source recordings are not redistributed. |

The model snapshots are legal/provenance evidence, not payloads and not a grant
of rights beyond their text. In particular, the XTTS CPML speaks directly to
outputs, while the F5 model card identifies the checkpoint licence; the
no-audio boundary avoids relying on a broader interpretation of either. A
separate human review is required before releasing any generated audio.

Licensing a corpus or model does not settle consent, publicity, biometric, or
privacy expectations. This work reports research measurements and does not
claim consent for deployment.
