#!/usr/bin/env bash
#SBATCH --job-name=dcorr_v2
#SBATCH --partition=longrungpu
#SBATCH --gres=gpu:h100:1
#SBATCH --exclude=gpu0001
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --array=0-2
#SBATCH --output=v2_core_%A_%a.out
#SBATCH --error=v2_core_%A_%a.err
#
# gpu0001 IS EXCLUDED ON PURPOSE. Every dcorr job that landed on it died with
# "RuntimeError: CUDA error: unrecognized error code", and every job that did not land on it
# completed: 2349226 (stacks) and 2349207_2 (Llama-2 seed 2) both failed there on 2026-10-14,
# while 2349207_0 on gpu0002, 2349207_1 on gpu0004 and the whole 2347924 array on gpu0005
# succeeded. SLURM still reports the node as IDLE rather than DRAIN, so it keeps accepting work
# that then dies. The GAMBIT jobs on this cluster already exclude it for the same reason.
#
# v2 core: the undefended control and every defense, one array task per attack replicate.
# This is the measurement the whole revision rests on, because the seed-to-seed correlation of
# the undefended search is the null that every defense pair is read against (REVIEW F2, F3).
#
# Environment copied from the v1 pipeline job, which is the configuration that produced the
# reported runs: a hand-built venv rather than modules (conda channels are blocked here), the
# HF cache on /project, compute nodes air-gapped and not mounting /home, secrets from the
# repo .env. Stages 03 and 04 only; 05/24 are separate jobs and grading needs internet, so it
# runs on the login node afterwards.
#
# Both stages resume by result key, so re-queueing after a timeout costs only the current
# chunk. Measured on an RTX 4080: about 25 h per seed for these two stages at n=100, so 48 h
# is roughly twice the expected need on an H100.
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"

source /project/$USER/venvs/dcorr/bin/activate
export PATH="$VIRTUAL_ENV/bin:$PATH"
PY="$(command -v python)"

set -a; [ -f "$SLURM_SUBMIT_DIR/.env" ] && . "$SLURM_SUBMIT_DIR/.env"; set +a
export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-${HF_TOKEN:-}}"
export HF_TOKEN="${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}"

export HF_HOME=/project/$USER/.cache/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false

# One job per GPU, so the local single-card lock is neither needed nor wanted here.
export DCORR_NO_GPU_LOCK=1
# Unbuffered stdout, or a stage that logs progress without flush=True writes nothing to
# the job log for hours and a slow stage is indistinguishable from a hung one.
export PYTHONUNBUFFERED=1

CONFIG="${CONFIG:-configs/v2_vicuna.yaml}"
SEED="${SLURM_ARRAY_TASK_ID:-0}"

echo "node $(hostname)  seed $SEED  config $CONFIG"
echo "python $PY"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# External calibration must exist before stage 04, which errors on a thresholded filter without
# it. For Vicuna it was copied from the v1 external run; a new target has none, so derive it
# here. Only task 0 does the work and the others wait, because all three array tasks share one
# calibration file and three concurrent writers would race. Guarded on the file, so a requeue
# does not recalibrate.
RUN_DIR="results/$("$PY" -c "
import sys; sys.path.insert(0, '.')
from dcorr.config import load_config
print(load_config('$CONFIG').get_path('run_name'))")"
CALIB="$RUN_DIR/calibration.json"
if [ ! -f "$CALIB" ]; then
  if [ "$SEED" = "0" ]; then
    echo "[core] no $CALIB; running external calibration on 805 held-out instructions"
    "$PY" scripts/15_calibrate_external.py --config "$CONFIG" \
          --external data/external_benign_alpacaeval.jsonl --out-name calibration.json
    crc=$?
    echo "[core] calibration exit=$crc"
    [ $crc -eq 0 ] || { echo "[core] calibration failed; not starting stage 04"; exit $crc; }
  else
    echo "[core] seed $SEED waiting for task 0 to write $CALIB"
    for _ in $(seq 1 180); do [ -f "$CALIB" ] && break; sleep 20; done
    [ -f "$CALIB" ] || { echo "[core] $CALIB never appeared; aborting"; exit 1; }
  fi
fi

"$PY" scripts/25_run_grid.py --config "$CONFIG" --seeds "$SEED" --stages 03 04
rc=$?

echo "seed $SEED stages 03+04 exit=$rc"
if [ "$SEED" = "0" ] && [ $rc -eq 0 ]; then
  echo "next, once the array finishes:"
  echo "  sbatch scripts/slurm_v2_stacks.sbatch        # 05 end-to-end, 24 cross-eval, benign"
  echo "  on the LOGIN node (needs internet), per seed:"
  echo "  python scripts/06_judge_gold.py --config $CONFIG --seed-index <k>"
fi
exit $rc
