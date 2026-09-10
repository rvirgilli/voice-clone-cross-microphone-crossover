#!/usr/bin/env bash
# EXP-212: the EXP-205 crossover on fresh pairs with the frozen EXP-205 generators
# (via EXP-210's gen_jobs.py). Resumable per clone (ledger-sealed outputs are skipped)
# and per readout.
set -euo pipefail
trap ':' USR1
# Historical run layout: ICASSP_RUNS and ICASSP_EXPERIMENTS point at copies of the private
# run directories and the experiment tree.
RUNS="${ICASSP_RUNS:-/runs}"
EXPERIMENTS="${ICASSP_EXPERIMENTS:-/experiments}"
EXP="$EXPERIMENTS/EXP-212-f2-fresh-pair-replication"
GEN="$EXPERIMENTS/EXP-210-f2-second-generation/gen_jobs.py"
RUN="$RUNS/EXP-212-f2-fresh-pair-replication"
XTTS_VENV="$RUNS/EXP-201-xtts/venv"
COSY_ROOT="$RUNS/tts-stacks/CosyVoice"
SEEDVC_ROOT="$RUNS/tts-stacks/seed-vc"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 NUMBA_CACHE_DIR="$RUN/numba-cache"
mkdir -p "$RUN" "$NUMBA_CACHE_DIR"
progress() { command -v gpu-progress >/dev/null && gpu-progress --percent "$1" --message "$2" || true; }
JOBS="$RUN/jobs.json"
[[ -f "$JOBS" ]] || uv run --no-project --python 3.11 python "$EXP/build_jobs_fresh_pair.py"

progress 2 "f5 generation"
uv run --python 3.10 --with f5-tts==1.1.22 python "$GEN" --jobs "$JOBS" --system f5
progress 15 "xtts generation"
VIRTUAL_ENV="$XTTS_VENV" uv run --active --no-project python "$GEN" --jobs "$JOBS" --system xtts
progress 45 "cosy generation"
PYTHONPATH="$COSY_ROOT:$COSY_ROOT/third_party/Matcha-TTS" VIRTUAL_ENV="$COSY_ROOT/venv" uv run --active --no-project \
  python "$GEN" --jobs "$JOBS" --system cosy
progress 65 "seedvc generation"
( cd "$SEEDVC_ROOT" && PYTHONPATH="$SEEDVC_ROOT" VIRTUAL_ENV="$SEEDVC_ROOT/venv" uv run --active --no-project \
  python "$GEN" --jobs "$JOBS" --system seedvc )

progress 92 "readout extraction"
ENV=(--python 3.11 --with librosa==0.11.0 --with soundfile --with speechbrain==1.1.0
     --with transformers==5.15.0 --with torch==2.13.0 --with 'setuptools<81' --with numpy==2.4.6)
uv run "${ENV[@]}" python "$EXP/analyze_fresh_pair.py" extract --run "$RUN"
progress 98 "analysis"
uv run "${ENV[@]}" python "$EXP/analyze_fresh_pair.py" analyze --run "$RUN"
progress 100 "complete"
echo EXP212_COMPLETE
