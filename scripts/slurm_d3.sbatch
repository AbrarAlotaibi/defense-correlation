#!/usr/bin/env bash
#SBATCH --job-name=dcorr_d3
#SBATCH --partition=gpu_x450
#SBATCH --gres=gpu:h100:1
#SBATCH --exclude=gpu0005
# gpu_x450, and gpu0005 excluded on purpose: the long D2 stack attack holds a GPU there, and
# co-scheduling two dcorr jobs on one node previously halved throughput and cost a defense to
# an OOM. This job is short and there is no reason to run it next to the expensive one.
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=dcorr_d3_%j.out
#SBATCH --error=dcorr_d3_%j.err
#
# D3 (REVIEW item M4): recalibrate the four thresholded filters on an external benign corpus.
# Scoring only - no generation, no attack search - so this is short.
#
# Environment copied verbatim from scripts/slurm_pipeline.sbatch.
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

echo "node: $(hostname)  python: $PY"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader || true

"$PY" scripts/15_calibrate_external.py \
  --config configs/hpc_vicuna_autodan.yaml \
  --external data/external_benign_alpacaeval.jsonl
echo "[D3] exit=$?"
