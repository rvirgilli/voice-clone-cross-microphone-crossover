# Cross-microphone voice-clone attribution

Code, data and manuscript for **"Which Conditioning Recording Does a Voice Clone
Follow? A Balanced Cross-Microphone Crossover."**

We ask which of two recordings of the same speaker was used to prompt a voice
clone. A clone is generated from one microphone and attributed through the
other, so speaker identity, generated text, exact waveform and a shared
microphone fingerprint cannot decide the label. Over 54 VCTK speakers and four
open cloning systems (3,456 clones), the mic1→mic2 follow rate is **.896
[.869, .921]** with an ECAPA readout and **.622 [.593, .652]** with WavLM.
Closed-set attribution is strong; open-set presence verification is not.

## Contents

| Path | What it holds |
|---|---|
| `paper/` | manuscript, LaTeX source, bibliography, figures |
| `data/` | score table, trial roster, generation ledger and configs, results |
| `code/` | generation, scoring, analysis, sensitivity checks, tests |
| `PREREGISTRATION.md` | the two registered plans and their deviations |
| `SCOPE.md` | what is claimed, and the boundaries it does not cross |
| `DATA.md` | data, model and licence boundaries |
| `REPRODUCE.md` | what reproduces, and the exact commands |

## Reproduce

Python 3.11+ and [`uv`](https://docs.astral.sh/uv/):

```bash
uv sync --frozen
uv run --frozen python code/verify.py
```

This treats `data/checksums.sha256` as the explicit root of trust, requires it to
enumerate every other release payload file exactly once, then recomputes the 3,456-row
score census, follow rates and 100,000-draw bootstrap intervals, generation ledger,
54 selected A/B pairs, and microphone-channel control. It rejects unlisted payloads,
symlinks, audio, model weights and archives. It runs on CPU and does not generate or
score audio.

With a LaTeX toolchain installed, `uv run --frozen python code/verify_manuscript.py`
also performs a clean four-page build and checks that the released PDF text matches
the released source.

Regenerating the clones themselves needs the third-party models and pinned
revisions listed in `DATA.md`, and is not required to check the paper.

## Licence

Code is MIT. The manuscript, datasets, models and third-party implementations
keep their own terms — see `DATA.md` and `LICENSES.md`. No source or generated
speech and no model weights are redistributed here.
