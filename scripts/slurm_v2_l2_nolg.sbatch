#!/usr/bin/env bash
#SBATCH --job-name=dcorr_l2_nolg
#SBATCH --partition=gpu_x450,longrungpu
#SBATCH --gres=gpu:h100:1
#SBATCH --exclude=gpu0001
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --array=0-2
#SBATCH --output=l2_nolg_%A_%a.out
#SBATCH --error=l2_nolg_%A_%a.err
#
# The six-layer stack on the replication target, which slurm_v2_l2_stack.sbatch failed to
# produce. All six of its no-guard invocations (3 seeds x harmful + OR-Bench benign) exited 1
# with "ppl_filter.threshold is unset", because results/v2_llama2_nolg/calibration.json does
# not exist: each run directory carries its own copy of the calibrated thresholds, and nothing
# had ever written one for this configuration. results/v2_vicuna_nolg/calibration.json exists
# for the primary target, which is why the Vicuna six-layer stack ran and this one did not.
#
# That failure is invisible in the job's exit status, because the surrounding script ran each
# stage with `run ()` and reported per-stage exits rather than aborting -- which is right for a
# long job whose later stages are independent, but it means "COMPLETED" covered six failures.
# The l2_stack script now copies the calibration before running the no-guard stack, so a
# resubmission of it would not hit this; this job exists to recover the missing stage without
# repeating the eight hours of work that did succeed.
#
# WHY COPYING THE THRESHOLDS IS CORRECT, not a shortcut: v2_llama2_nolg.yaml extends
# v2_llama2.yaml and changes exactly two things, the guard's `enabled` flag and the stack
# order. Target, dtype, seeds, behaviours and every member's operating point are inherited. So
# the six non-guard members here ARE the same members at the same 1%-FPR thresholds as in the
# primary run, and the comparison between the seven- and six-layer stacks isolates the guard.
# Recalibrating instead would change the thresholds and break exactly that.
#
# gpu0001 IS EXCLUDED ON PURPOSE; see slurm_v2_core_llama2.sbatch.
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
export DCORR_NO_GPU_LOCK=1
export PYTHONUNBUFFERED=1

CONFIG=configs/v2_llama2_nolg.yaml
SRC=results/v2_llama2/calibration.json
DST=results/v2_llama2_nolg/calibration.json
SEED="${SLURM_ARRAY_TASK_ID:-0}"

echo "node $(hostname)  seed $SEED  six-layer stack, no guard"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

if [ ! -f "$SRC" ]; then
  echo "[nolg] $SRC is missing; the primary Llama-2 run must be calibrated first. Aborting."
  exit 1
fi
mkdir -p "$(dirname "$DST")"
if [ ! -f "$DST" ]; then
  cp "$SRC" "$DST"
  echo "[nolg] copied the primary run's thresholds to $DST"
else
  # Guarded, so a requeue cannot silently substitute different thresholds mid-array.
  if cmp -s "$SRC" "$DST"; then
    echo "[nolg] $DST already matches the primary run"
  else
    echo "[nolg] $DST EXISTS AND DIFFERS from $SRC. Refusing to overwrite: the six-layer"
    echo "[nolg] results must use the same operating points as the seven-layer ones, and"
    echo "[nolg] something has changed one of them. Resolve by hand."
    exit 1
  fi
fi

run () { echo "=== seed $SEED :: $*"; "$PY" "$@"; echo "    exit=$?"; }

# Harmful first, then both benign sets. The harmful pass is the one E3 needs.
run scripts/05_run_stack.py --config "$CONFIG" --seed-index "$SEED"
run scripts/05_run_stack.py --config "$CONFIG" --seed-index "$SEED" \
    --benign-only --benign-set orbench

OUT="results/v2_llama2_nolg/seed$SEED/stage05_stack.jsonl"
if [ -s "$OUT" ]; then
  echo "[nolg] seed $SEED wrote $(wc -l < "$OUT") rows to $OUT"
else
  echo "[nolg] seed $SEED produced NO harmful rows at $OUT; the stage did not succeed"
  exit 1
fi

echo
echo "next, with internet:"
echo "  python scripts/06_judge_gold.py --config $CONFIG --seed-index $SEED"
