#!/usr/bin/env bash
# EXP-213: the EXP-205 crossover on the 54 tier speakers with the frozen EXP-205
# generators (via EXP-210's gen_jobs.py). Resumable per clone (ledger-sealed outputs are
# skipped) and per readout. Structured progress in clones at each stage boundary.
set -euo pipefail
trap ':' USR1
# Historical run layout: ICASSP_RUNS and ICASSP_EXPERIMENTS point at copies of the private
# run directories and the experiment tree.
RUNS="${ICASSP_RUNS:-/runs}"
EXPERIMENTS="${ICASSP_EXPERIMENTS:-/experiments}"
EXP="$EXPERIMENTS/EXP-213-f2-full-roster-complement"
GEN="$EXPERIMENTS/EXP-210-f2-second-generation/gen_jobs.py"
RUN="$RUNS/EXP-213-f2-full-roster-complement"
XTTS_VENV="$RUNS/EXP-201-xtts/venv"
COSY_ROOT="$RUNS/tts-stacks/CosyVoice"
SEEDVC_ROOT="$RUNS/tts-stacks/seed-vc"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 NUMBA_CACHE_DIR="$RUN/numba-cache"
mkdir -p "$RUN" "$NUMBA_CACHE_DIR"
JOBS="$RUN/jobs.json"
[[ -f "$JOBS" ]] || uv run --no-project --python 3.11 python "$EXP/build_jobs_full_roster.py"
TOTAL=$(python3 -c "import json,sys; print(len(json.load(open(sys.argv[1]))['jobs']))" "$JOBS")
progress() { command -v gpu-progress >/dev/null && gpu-progress --current "$1" --total "$TOTAL" --unit clone --phase "$2" || true; }

progress 0 "f5 generation"
uv run --python 3.10 --with f5-tts==1.1.22 python "$GEN" --jobs "$JOBS" --system f5
progress $((TOTAL / 4)) "xtts generation"
VIRTUAL_ENV="$XTTS_VENV" uv run --active --no-project python "$GEN" --jobs "$JOBS" --system xtts
progress $((TOTAL / 2)) "cosy generation"
PYTHONPATH="$COSY_ROOT:$COSY_ROOT/third_party/Matcha-TTS" VIRTUAL_ENV="$COSY_ROOT/venv" uv run --active --no-project \
  python "$GEN" --jobs "$JOBS" --system cosy
progress $((3 * TOTAL / 4)) "seedvc generation"
( cd "$SEEDVC_ROOT" && PYTHONPATH="$SEEDVC_ROOT" VIRTUAL_ENV="$SEEDVC_ROOT/venv" uv run --active --no-project \
  python "$GEN" --jobs "$JOBS" --system seedvc )

progress "$TOTAL" "readout extraction"
ENV=(--python 3.11 --with librosa==0.11.0 --with soundfile --with speechbrain==1.1.0
     --with transformers==5.15.0 --with torch==2.13.0 --with 'setuptools<81' --with numpy==2.4.6)
uv run "${ENV[@]}" python "$EXP/analyze_full_roster.py" extract --run "$RUN"
progress "$TOTAL" "analysis"
uv run "${ENV[@]}" python "$EXP/analyze_full_roster.py" analyze --run "$RUN"
progress "$TOTAL" "complete"
echo EXP213_COMPLETE
