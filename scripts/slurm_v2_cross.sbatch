#!/usr/bin/env bash
#SBATCH --job-name=dcorr_cross
#SBATCH --partition=gpu_x450,longrungpu
#SBATCH --gres=gpu:h100:1
#SBATCH --exclude=gpu0001
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --array=0-4
#SBATCH --output=cross_%A_%a.out
#SBATCH --error=cross_%A_%a.err
#
# Estimand E2 on the replicates and the target it is currently missing.
#
# The cross-evaluation scores every member on every other member's final prompts, which is the
# only way to measure co-failure on a SHARED input rather than across separately searched
# arms. It exists for v2_vicuna seed 0 and nowhere else. Two consequences in the paper as it
# stands:
#
#   * E2 and the transfer matrix are point estimates with no replicate spread. E1 is read
#     against a seed-pair null precisely because a single pair of runs turned out twice in this
#     project to support a conclusion the third replicate removed. E2 has no such control.
#   * There is no cross-evaluation on Llama-2, so the aggregator reversal there -- the paper's
#     most quotable practical claim -- rests on arms that searched separately. The same-input
#     test exists only on the target where that member helps.
#
# WHY THIS IS CHEAPER THAN IT LOOKS. Stage 24 replays stored prompts; it runs no search. The
# cost is generation over 7 members by 100 behaviours plus the undefended arm, so hours rather
# than the day a stage 05 array takes. The 12-hour wall is slack, not an estimate.
#
# TASK MAP. Deliberately explicit rather than computed, so a failed task can be requeued by
# index without working out which cell it was:
#   0  v2_llama2 seed 0      3  v2_vicuna seed 1
#   1  v2_llama2 seed 1      4  v2_vicuna seed 2
#   2  v2_llama2 seed 2
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

CONFIGS=(configs/v2_llama2.yaml configs/v2_llama2.yaml configs/v2_llama2.yaml \
         configs/v2_vicuna.yaml configs/v2_vicuna.yaml)
SEEDS=(0 1 2 1 2)
T="${SLURM_ARRAY_TASK_ID:-0}"
CONFIG="${CONFIGS[$T]}"
SEED="${SEEDS[$T]}"

echo "node $(hostname)  task $T  config $CONFIG  seed $SEED"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# Stage 24 reads each member's stage 04 prompts for this seed. Fail loudly if they are absent
# rather than writing a cross directory with missing cells, which the analysis would then
# report as a transfer matrix with holes in it.
RUN=$(grep -E '^run_name:' "$CONFIG" | awk '{print $2}')
if [ ! -f "results/$RUN/seed$SEED/stage04_undefended.jsonl" ]; then
  echo "[24] results/$RUN/seed$SEED/stage04_undefended.jsonl is missing: stage 04 has not run"
  echo "[24] for this seed, so there are no prompts to cross-evaluate. Aborting."
  exit 1
fi

"$PY" scripts/24_cross_eval.py --config "$CONFIG" --seed-index "$SEED"
rc=$?
echo "task $T exit=$rc"

if [ $rc -eq 0 ]; then
  echo "next, with internet:"
  echo "  python scripts/06_judge_gold.py --config $CONFIG --seed-index $SEED --src-dir cross"
  echo "  python scripts/35_resolve_abstentions.py --config $CONFIG --seed-index $SEED \\"
  echo "      --src-dir cross --grader deepseek/deepseek-v4-pro-0813 \\"
  echo "      --grader-base-url https://router.requesty.ai/v1 \\"
  echo "      --grader-key-env REQUESTY_API_KEY --grader-max-tokens 4000"
fi
exit $rc
