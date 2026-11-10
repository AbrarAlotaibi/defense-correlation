#!/usr/bin/env bash
#SBATCH --job-name=dcorr_l2_stack
#SBATCH --partition=gpu_x450,longrungpu
#SBATCH --gres=gpu:h100:1
#SBATCH --exclude=gpu0001
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=36:00:00
#SBATCH --array=0-2
#SBATCH --output=l2_stack_%A_%a.out
#SBATCH --error=l2_stack_%A_%a.err
#
# What the replication target is missing, and why it matters.
#
# The paper claims a replication on a strongly aligned target, but results/v2_llama2 holds
# stage 04 only: per-member attack runs and the matched benign pass. There is no stage 05, no
# undefended benign arm, and no OR-Bench pass. So of the three estimands the paper defines,
# the replication covers E1 and neither E2 nor E3, and the whole cost-of-depth section is
# single-target. A reviewer reading Section 3's framing against Section 5.8 would find that
# two thirds of the design was not in fact replicated.
#
# This job fills it: the assembled seven-layer stack, the six-layer stack without the guard,
# the undefended benign arm, and both benign sets through every member and both stacks.
#
# gpu0001 IS EXCLUDED ON PURPOSE. Every dcorr job that landed on it died with "CUDA error:
# unrecognized error code" while SLURM kept reporting the node IDLE. See
# slurm_v2_core_llama2.sbatch for the job IDs.
#
# ORDER. Cheap generation-only stages first, expensive searches last, so that a wall clock or a
# preemption still leaves usable artefacts. The benign passes replay fixed prompts and need no
# search; stage 05 runs a full adaptive search against the assembled stack and is the long
# pole. Every stage resumes by result key, so a requeue costs only the current chunk.
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

CONFIG=configs/v2_llama2.yaml
NOLG=configs/v2_llama2_nolg.yaml
SEED="${SLURM_ARRAY_TASK_ID:-0}"

echo "node $(hostname)  seed $SEED  target Llama-2-7b-chat"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

if [ ! -f results/v2_llama2/calibration.json ]; then
  echo "[l2] results/v2_llama2/calibration.json is missing; stage 04 wrote this run, so it"
  echo "[l2] should exist. Refusing to recalibrate, because new thresholds would make these"
  echo "[l2] stack results incomparable with the stage 04 numbers already in the paper."
  exit 1
fi

# Each run directory carries its own copy of the calibrated thresholds, and nothing had ever
# written one for the no-guard configuration -- so every no-guard invocation below failed with
# "ppl_filter.threshold is unset" while this job still reported COMPLETED, because `run ()`
# records a per-stage exit instead of aborting. Copying is correct rather than a shortcut:
# v2_llama2_nolg.yaml changes only the guard's enabled flag and the stack order, so its six
# members are the same members at the same 1%-FPR operating points, and recalibrating would
# change them and break the very comparison the no-guard stack exists to make.
mkdir -p results/v2_llama2_nolg
if [ ! -f results/v2_llama2_nolg/calibration.json ]; then
  cp results/v2_llama2/calibration.json results/v2_llama2_nolg/calibration.json
  echo "[l2] copied the thresholds to results/v2_llama2_nolg/calibration.json"
elif ! cmp -s results/v2_llama2/calibration.json results/v2_llama2_nolg/calibration.json; then
  echo "[l2] results/v2_llama2_nolg/calibration.json differs from the primary run's."
  echo "[l2] Refusing to overwrite: the two stacks must share operating points."
  exit 1
fi

run () {
  echo "=== seed $SEED :: $*"
  "$PY" "$@"
  echo "    exit=$?"
}

# 12 / B12: the undefended arm on each benign set. Needed before any union or composition
# figure means anything, because a member's false refusals include the ones the bare target
# would have produced anyway, and on this target that baseline is much larger than on Vicuna.
run scripts/12_undefended_benign.py --config "$CONFIG" --seed-index "$SEED"
run scripts/12_undefended_benign.py --config "$CONFIG" --seed-index "$SEED" --benign-set orbench

# B4: every member on the 400 held-out OR-Bench prompts.
run scripts/04_run_attacks.py --config "$CONFIG" --seed-index "$SEED" \
    --benign-only --benign-set orbench

# 05: the assembled stacks, attacked end to end. Seven layers first, then six without the
# guard, which is the configuration where the comparison has resolution.
run scripts/05_run_stack.py --config "$CONFIG" --seed-index "$SEED"
run scripts/05_run_stack.py --config "$NOLG"   --seed-index "$SEED"

# B5: both stacks on both benign sets. These are the measured over-refusal figures that
# Section 5.7 reports for Vicuna and currently cannot report here.
run scripts/05_run_stack.py --config "$CONFIG" --seed-index "$SEED" \
    --benign-only --benign-set orbench
run scripts/05_run_stack.py --config "$NOLG" --seed-index "$SEED" \
    --benign-only --benign-set orbench

echo
echo "seed $SEED done. Grading needs internet, so from the LOGIN node or locally:"
echo "  python scripts/06_judge_gold.py --config $CONFIG --seed-index $SEED"
echo "  python scripts/06_judge_gold.py --config $NOLG   --seed-index $SEED"
echo "  python scripts/35_resolve_abstentions.py --config $CONFIG --seed-index $SEED \\"
echo "      --grader deepseek/deepseek-v4-pro-0813 \\"
echo "      --grader-base-url https://router.requesty.ai/v1 \\"
echo "      --grader-key-env REQUESTY_API_KEY --grader-max-tokens 4000"
