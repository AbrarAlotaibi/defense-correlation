#!/usr/bin/env bash
#SBATCH --job-name=dcorr_v2_l2
#SBATCH --partition=gpu_x450,longrungpu
#SBATCH --gres=gpu:h100:1
#SBATCH --exclude=gpu0001
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --array=0-2
#SBATCH --output=v2_core_l2_%A_%a.out
#SBATCH --error=v2_core_l2_%A_%a.err
#
# gpu0001 IS EXCLUDED ON PURPOSE. Every dcorr job that landed on it died with
# "RuntimeError: CUDA error: unrecognized error code", and every job that did not land on it
# completed: 2349226 (stacks) and 2349207_2 (Llama-2 seed 2) both failed there on 2026-10-14,
# while 2349207_0 on gpu0002, 2349207_1 on gpu0004 and the whole 2347924 array on gpu0005
# succeeded. SLURM still reports the node as IDLE rather than DRAIN, so it keeps accepting work
# that then dies. The GAMBIT jobs on this cluster already exclude it for the same reason.
#
# PARTITION. Both are listed so the scheduler takes whichever frees first. longrungpu is a
# single node whose two H100s are often held for days by other users' long jobs, which left this
# job waiting with an estimated start a full day out while gpu0002 sat idle in gpu_x450. The
# gpu_x450 limit is 2-00:00:00, exactly the --time below, so the request fits either way, and
# every stage resumes by result key, so hitting a wall costs only the current chunk.
#
# The replication target: Llama-2-7b-chat, the more strongly aligned of the two.
#
# WHY A SEPARATE FILE rather than `CONFIG=... sbatch slurm_v2_core.sbatch`. The core script
# defaults CONFIG to configs/v2_vicuna.yaml. If an exported CONFIG failed to reach the job for
# any reason, the job would silently fall back to the Vicuna config and write into
# results/v2_vicuna/seed<k>, racing a Vicuna array task that is still running. scontrol does
# not display a job's environment, so that assumption cannot be checked after submission.
# Hard-coding the config here removes the failure mode instead of relying on it not happening.
#
# Expect this target to behave differently, and the difference is the point: against a
# refusal-robust model the earlier version of this work found the token-surface filters blocking
# essentially every attack prompt, which drives marginal breach rates to zero and leaves phi
# undefined rather than small. That is the structural blindness the Discussion claims, so this
# run is evidence for a limitation as much as a replication of a result.
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
# Unbuffered stdout, or a stage that logs progress without flush=True writes nothing to
# the job log for hours and a slow stage is indistinguishable from a hung one.
export PYTHONUNBUFFERED=1

CONFIG=configs/v2_llama2.yaml          # fixed on purpose; see the note above
SEED="${SLURM_ARRAY_TASK_ID:-0}"
RUN_DIR=results/v2_llama2
CALIB="$RUN_DIR/calibration.json"

echo "node $(hostname)  seed $SEED  config $CONFIG"
echo "python $PY"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# This target has no external calibration yet, and stage 04 errors on a thresholded filter
# without one. Task 0 derives it from the 805 held-out instructions; the other tasks wait,
# because all three share the file and three writers would race. Guarded on the file, so a
# requeue does not recalibrate.
if [ ! -f "$CALIB" ]; then
  if [ "$SEED" = "0" ]; then
    echo "[l2] no $CALIB; calibrating on 805 held-out instructions at the configured FPR"
    "$PY" scripts/15_calibrate_external.py --config "$CONFIG" \
          --external data/external_benign_alpacaeval.jsonl --out-name calibration.json
    crc=$?
    echo "[l2] calibration exit=$crc"
    [ $crc -eq 0 ] || { echo "[l2] calibration failed; not starting stage 04"; exit $crc; }
  else
    echo "[l2] seed $SEED waiting for task 0 to write $CALIB"
    for _ in $(seq 1 180); do [ -f "$CALIB" ] && break; sleep 20; done
    [ -f "$CALIB" ] || { echo "[l2] $CALIB never appeared; aborting"; exit 1; }
  fi
fi

"$PY" scripts/25_run_grid.py --config "$CONFIG" --seeds "$SEED" --stages 03 04
rc=$?

echo "seed $SEED stages 03+04 exit=$rc"
if [ "$SEED" = "0" ] && [ $rc -eq 0 ]; then
  echo "next: grading needs internet, so on the LOGIN node, per seed:"
  echo "  python scripts/06_judge_gold.py --config $CONFIG --seed-index <k>"
  echo "  python scripts/35_resolve_abstentions.py --config $CONFIG --seed-index <k> \\"
  echo "      --grader deepseek/deepseek-v4-pro-0813 \\"
  echo "      --grader-base-url https://router.requesty.ai/v1 \\"
  echo "      --grader-key-env REQUESTY_API_KEY --grader-max-tokens 4000"
fi
exit $rc
