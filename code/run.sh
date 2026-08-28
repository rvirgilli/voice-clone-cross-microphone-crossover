#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || ! "$1" =~ ^[0-9a-f]{64}$ ]]; then
  echo "usage: run.sh EXPECTED_EXECUTION_CONFIG_SHA256" >&2
  exit 64
fi
EXPECTED_CFG_SHA256="$1"

EXP="${EXP205_ROOT:?set EXP205_ROOT to the experiment source directory}"
CFG="$EXP/execution-config.json"
RUN="${EXP205_RUN:?set EXP205_RUN to a writable run directory}"
DFM="${EXP205_DETECTOR_ROOT:?set EXP205_DETECTOR_ROOT}"
XTTS_VENV="${EXP205_XTTS_VENV:?set EXP205_XTTS_VENV}"
COSY_ROOT="${EXP205_COSY_ROOT:?set EXP205_COSY_ROOT}"
COSY_VENV="$COSY_ROOT/venv"
SEEDVC_ROOT="${EXP205_SEEDVC_ROOT:?set EXP205_SEEDVC_ROOT}"
SEEDVC_VENV="$SEEDVC_ROOT/venv"

STAGE=bootstrap
write_status() {
  local exit_code="$1"
  local status=EXP205_ONE_JOB_COMPLETE
  if [[ "$exit_code" -ne 0 ]]; then
    status=INFRASTRUCTURE_FAILURE
  fi
  local temporary="$RUN/execution-status.json.tmp"
  printf '{"status":"%s","stage":"%s","exit_code":%d,"execution_config_sha256":"%s"}\n' \
    "$status" "$STAGE" "$exit_code" "$EXPECTED_CFG_SHA256" > "$temporary"
  mv "$temporary" "$RUN/execution-status.json"
}
on_exit() {
  local exit_code="$?"
  write_status "$exit_code"
}
trap on_exit EXIT

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export NUMBA_CACHE_DIR="$RUN/numba-cache"
mkdir -p "$RUN" "$NUMBA_CACHE_DIR"

OBSERVED_CFG_SHA256="$(sha256sum "$CFG" | awk '{print $1}')"
if [[ "$OBSERVED_CFG_SHA256" != "$EXPECTED_CFG_SHA256" ]]; then
  echo "execution config trust-root mismatch" >&2
  exit 65
fi

STAGE=verify_pins
uv run --project "$DFM" python "$EXP/verify_pins.py" --config "$CFG"
STAGE=build_seedvc_manifest
uv run --project "$DFM" python "$EXP/build_seedvc_manifest.py" \
  --config "$CFG" --out "$RUN/seedvc-manifest.json"

STAGE=generate_f5
uv run --python 3.10 --with f5-tts==1.1.22 \
  python "$EXP/generate.py" --config "$CFG" --system f5

STAGE=generate_xtts
VIRTUAL_ENV="$XTTS_VENV" uv run --active --no-project \
  python "$EXP/generate.py" --config "$CFG" --system xtts

STAGE=generate_cosy
PYTHONPATH="$COSY_ROOT:$COSY_ROOT/third_party/Matcha-TTS" \
  VIRTUAL_ENV="$COSY_VENV" uv run --active --no-project \
  python "$EXP/generate.py" --config "$CFG" --system cosy

cd "$SEEDVC_ROOT"
STAGE=generate_seedvc
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH="$SEEDVC_ROOT" \
  VIRTUAL_ENV="$SEEDVC_VENV" uv run --active --no-project \
  python "$EXP/generate_seedvc.py" --config "$CFG"
cd "$EXP"

STAGE=score
uv run --python 3.11 \
  --with librosa==0.11.0 \
  --with pyworld \
  --with soundfile \
  --with speechbrain==1.1.0 \
  --with transformers==5.15.0 \
  --with torch==2.13.0 \
  --with 'setuptools<81' \
  --with numpy==2.4.6 \
  python "$EXP/score.py" --config "$CFG"

STAGE=analyze
uv run --python 3.11 --with numpy==2.4.6 \
  python "$EXP/analyze.py" --config "$RUN/analysis-config.json" \
  --expected-execution-config-sha256 "$EXPECTED_CFG_SHA256"

STAGE=complete
echo EXP205_ONE_JOB_COMPLETE
