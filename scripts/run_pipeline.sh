#!/usr/bin/env bash
# Linux/HPC orchestrator. Runs the pipeline in order with
# the stage-03 gate enforced. Every stage is resumable, so re-running continues where it
# stopped. Split GPU vs API stages with the flags below when compute nodes are offline.
#
#   Usage:
#     CONFIG=configs/hpc_llama2.yaml bash scripts/run_pipeline.sh            # full run
#     CONFIG=... GPU_ONLY=1 bash scripts/run_pipeline.sh                     # stages 00-05 (no internet)
#     CONFIG=... API_ONLY=1 bash scripts/run_pipeline.sh                     # stages 06-07 (needs internet)
#     CONFIG=... NO_SMOKE=1 FROM=4 bash scripts/run_pipeline.sh              # resume at stage 4
#
# PY defaults to `python`; override to point at a specific interpreter/conda env.
set -euo pipefail
CONFIG="${CONFIG:-configs/hpc_llama2.yaml}"
PY="${PY:-python}"
FROM="${FROM:-0}"
cd "$(dirname "$0")/.."

run() {  # run <n> <desc> <script> [extra args...]
  local n="$1"; shift; local desc="$1"; shift; local script="$1"; shift
  if [ "$n" -lt "$FROM" ]; then echo "[skip] stage $n ($desc)"; return; fi
  echo ""; echo "=== Stage $n : $desc ==="
  "$PY" "$script" --config "$CONFIG" "$@"
}

if [ "${API_ONLY:-0}" != "1" ]; then
  run 0 "prepare data"            scripts/00_prepare_data.py
  [ "${NO_SMOKE:-0}" = "1" ] || run 8 "smoke test" scripts/08_smoke_test.py
  run 2 "train probe"             scripts/02_train_probe.py
  run 1 "calibrate filters"       scripts/01_calibrate.py
  run 3 "positive control (GATE)" scripts/03_positive_control.py   # non-zero exit here = gate fail
  run 4 "run grid"                scripts/04_run_attacks.py
  run 5 "attack the stack"        scripts/05_run_stack.py
fi

if [ "${GPU_ONLY:-0}" != "1" ]; then
  run 6 "gold judging (API)"      scripts/06_judge_gold.py
  run 7 "analyze"                 scripts/07_analyze.py
  RUN=$("$PY" -c "import sys;sys.path.insert(0,'.');from dcorr.config import load_config;print(load_config('$CONFIG').get('run_name','run'))")
  echo ""; echo "Done. See results/$RUN/REPORT.md"
fi
