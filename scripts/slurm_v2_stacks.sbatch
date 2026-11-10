#!/usr/bin/env bash
#SBATCH --job-name=dcorr_v2s
#SBATCH --partition=gpu_x450,longrungpu
#SBATCH --gres=gpu:h100:1
#SBATCH --exclude=gpu0001
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=36:00:00
#SBATCH --array=0-2
#SBATCH --output=v2_stacks_%A_%a.out
#SBATCH --error=v2_stacks_%A_%a.err
#
# v2 stacks and the remaining estimands, one array task per attack replicate.
#
# WHY AN ARRAY RATHER THAN ONE JOB. Stage 05 crafts the stack's prompts at about six minutes per
# behaviour, so a single invocation is ~10 h, and run_one_defense writes nothing until the
# invocation finishes: a wall-clock kill mid-stage loses all 10 h rather than resuming from a
# chunk. Three replicates over two stack configurations is six invocations, ~60 h, which does not
# fit the 48 h limit as one job. Split per replicate, each task is at most ~25 h with room to
# spare, and a task that does die costs one seed of one configuration.
#
# WHY TASK 0 DOES MORE. Stage 24 (the same-input cross-evaluation, estimand E2) and the benign
# passes are seed-0 quantities, run once. They also run FIRST, before any stage 05, because they
# are cheap and they carry a whole estimand: stage 24 runs no search at all, only generation over
# prompts that already exist, with generation shared wherever the deployed systems agree. Putting
# them last, as an earlier version of this script did, risked spending the allocation on stage 05
# and finishing with no E2 and no measured refusal rate.
#
# gpu0001 IS EXCLUDED ON PURPOSE. Every dcorr job that landed on it died with
# "RuntimeError: CUDA error: unrecognized error code", and every job that did not land on it
# completed: 2349226 (stacks) and 2349207_2 (Llama-2 seed 2) both failed there on 2026-10-14,
# while 2349207_0 on gpu0002, 2349207_1 on gpu0004 and the whole 2347924 array on gpu0005
# succeeded. SLURM still reports the node as IDLE rather than DRAIN, so it keeps accepting work
# that then dies. The GAMBIT jobs on this cluster already exclude it for the same reason.
#
# PARTITION. Both are listed so the scheduler takes whichever frees first. longrungpu is a single
# node whose two H100s are often held for days by other users' long jobs, which once left this
# job with an estimated start a full day out while gpu0002 sat idle in gpu_x450.
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
# Unbuffered stdout, or a stage that logs progress without flush=True writes nothing to the job
# log for hours and a slow stage is indistinguishable from a hung one.
export PYTHONUNBUFFERED=1

SEED="${SLURM_ARRAY_TASK_ID:-0}"
echo "node $(hostname)  seed $SEED"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

rc=0
if [ "$SEED" = "0" ]; then
  # E2 and the benign passes first: cheap, and each supplies a result the stacks cannot.
  "$PY" scripts/25_run_grid.py --config configs/v2_vicuna.yaml --seeds 0 --stages 24 || rc=$?
  "$PY" scripts/25_run_grid.py --config configs/v2_vicuna.yaml --seeds 0 \
        --stages 12 B12 B4 B5 || rc=$?
fi

"$PY" scripts/25_run_grid.py --config configs/v2_vicuna.yaml      --seeds "$SEED" --stages 05 || rc=$?
"$PY" scripts/25_run_grid.py --config configs/v2_vicuna_nolg.yaml --seeds "$SEED" --stages 05 || rc=$?

echo "v2 stacks seed $SEED exit=$rc"
if [ "$SEED" = "0" ] && [ $rc -eq 0 ]; then
  echo "next, on the LOGIN node (needs internet):"
  echo "  python scripts/06_judge_gold.py --config configs/v2_vicuna.yaml --seed-index 0 --src-dir cross"
  echo "  python scripts/24_cross_eval.py --config configs/v2_vicuna.yaml --seed-index 0 --export"
fi
exit $rc
