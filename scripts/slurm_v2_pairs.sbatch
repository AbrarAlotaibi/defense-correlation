#!/usr/bin/env bash
#SBATCH --job-name=dcorr_v2p
#SBATCH --partition=gpu_x450,longrungpu
#SBATCH --gres=gpu:h100:1
#SBATCH --exclude=gpu0001
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --array=0-3
#SBATCH --output=v2_pairs_%A_%a.out
#SBATCH --error=v2_pairs_%A_%a.err
#
# Estimand E3 at the pair level (plan item 4.7; REVIEW R2-1, R5-1.2, R7-8b).
#
# WHY PAIRS AND NOT ONLY THE FULL STACK. The guard model blocks ~96% on its own, so the
# assembled seven-layer residual sits on the floor and any method that returns near zero agrees
# with any other there: the comparison between the per-layer intersection and the directly
# attacked stack has no resolution. A two-member stack leaves room for the two to disagree, so
# this is where the paper's practical claim can actually be tested.
#
# WHICH PAIRS, AND WHY THESE FOUR. Two within-family and two across, chosen at opposite ends of
# the measured phi range so that "does a high-phi pair compose worse than a low-phi one" is a
# question these runs can answer:
#   0  probe16 + probe8          within-family, the LARGEST phi in the table (0.646)
#   1  ppl_filter + token_anomaly within-family, token surface (0.604)
#   2  probe16 + refusal_prime    across-family, high phi (0.629)
#   3  smoothllm + token_anomaly  across-family, the LOWEST non-guard phi (0.525)
# The guard model is deliberately absent from all four: a pair containing it would floor the
# residual for the same reason the full stack does.
#
# 05_run_stack.py is called directly rather than through 25_run_grid.py, because the grid
# driver's stage table carries fixed arguments and has no way to pass --layers. The subset is
# intersected with stack.order inside the script, so the deployment order (and therefore the
# output name stage05_stack_<a>+<b>.jsonl) is canonical regardless of the order given here.
#
# gpu0001 IS EXCLUDED ON PURPOSE: it fails CUDA work with "unrecognized error code" while SLURM
# still reports it IDLE. See slurm_v2_stacks.sbatch for the evidence.
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

PAIRS=("probe probe_b" "ppl_filter token_anomaly" "probe refusal_prime" "smoothllm token_anomaly")
LAYERS="${PAIRS[${SLURM_ARRAY_TASK_ID:-0}]}"

echo "node $(hostname)  pair: $LAYERS"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# shellcheck disable=SC2086
"$PY" scripts/05_run_stack.py --config configs/v2_vicuna.yaml --seed-index 0 --layers $LAYERS
rc=$?

echo "pair [$LAYERS] exit=$rc"
if [ $rc -eq 0 ]; then
  echo "next, on the LOGIN node: grade it, then scripts/26_v2_analysis.py picks the subset up"
  echo "  python scripts/06_judge_gold.py --config configs/v2_vicuna.yaml --seed-index 0"
fi
exit $rc
