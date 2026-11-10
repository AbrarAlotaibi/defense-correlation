#!/usr/bin/env bash
#SBATCH --job-name=dcorr_pairs2
#SBATCH --partition=gpu_x450,longrungpu
#SBATCH --gres=gpu:h100:1
#SBATCH --exclude=gpu0001
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=14:00:00
#SBATCH --array=0-8
#SBATCH --output=pairs2_%A_%a.out
#SBATCH --error=pairs2_%A_%a.err
#
# Replicates for the pair-level E3 comparison, which is the one place the paper's practical
# claim is actually testable, and which currently rests on a single seed.
#
# WHY. slurm_v2_pairs.sbatch ran the four pairs on seed 0 only. On that seed the measured
# two-layer residual exceeds the per-layer intersection estimate in all four pairs, and for
# ppl_filter + token_anomaly the difference is significant on McNemar at p = 0.013 (0.66
# measured against 0.55 predicted). That is the E3 result the Discussion is built around, and
# a p-value from one replicate is exactly the kind of number this project has already watched
# collapse: the behaviour-level null moved enough between two and three seeds to overturn a
# written conclusion, twice. Two more replicates make the claim a range instead of a point.
#
# THE PAIRS are the same four, for the same reasons, so the three seeds are comparable:
#   probe16 + probe8           within-family, the largest phi in Table 5
#   ppl_filter + token_anomaly within-family, token surface
#   probe16 + refusal_prime    across-family, high phi
#   smoothllm + token_anomaly  across-family, the lowest non-guard phi
# The guard is absent from all four: a pair containing it floors the residual.
#
# TASK MAP. Tasks 0-7 are (pair, seed) in pair-major order; task 8 is an unrelated leftover
# that needs one short GPU slot and is not worth its own submission:
#   0  probe+probe_b            seed 1     4  probe+refusal_prime        seed 1
#   1  probe+probe_b            seed 2     5  probe+refusal_prime        seed 2
#   2  ppl_filter+token_anomaly seed 1     6  smoothllm+token_anomaly    seed 1
#   3  ppl_filter+token_anomaly seed 2     7  smoothllm+token_anomaly    seed 2
#   8  v2_vicuna_nolg on the OR-Bench benign set -- the one "---" in the over-refusal table
#
# 05_run_stack.py is called directly rather than through 25_run_grid.py, whose stage table
# carries fixed arguments and cannot pass --layers.
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

T="${SLURM_ARRAY_TASK_ID:-0}"
echo "node $(hostname)  task $T"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

if [ "$T" = "8" ]; then
  echo "=== v2_vicuna_nolg, OR-Bench benign pass (fills the missing cell in Table 8)"
  "$PY" scripts/05_run_stack.py --config configs/v2_vicuna_nolg.yaml --seed-index 0 \
        --benign-only --benign-set orbench
  rc=$?
  echo "task 8 exit=$rc"
  exit $rc
fi

PAIRS=("probe probe_b" "ppl_filter token_anomaly" "probe refusal_prime" "smoothllm token_anomaly")
LAYERS="${PAIRS[$((T / 2))]}"
SEED=$(( (T % 2) + 1 ))

echo "pair [$LAYERS]  seed $SEED"
"$PY" scripts/05_run_stack.py --config configs/v2_vicuna.yaml --seed-index "$SEED" \
      --layers $LAYERS
rc=$?
echo "task $T exit=$rc"

if [ $rc -eq 0 ]; then
  echo "next, with internet: python scripts/06_judge_gold.py --config configs/v2_vicuna.yaml"
  echo "  --seed-index $SEED   (stage 06 globs stage05_stack*, so it picks the subset up)"
fi
exit $rc
