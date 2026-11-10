#!/usr/bin/env bash
#SBATCH --job-name=dcorr
#SBATCH --partition=REPLACE_GPU_PARTITION     # <-- from `sinfo`; a GPU partition
#SBATCH --gres=gpu:1                          # 1 GPU (>=40GB ideal: A100 / L40S)
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=dcorr_%j.out
#SBATCH --error=dcorr_%j.err
# #SBATCH --account=REPLACE_ACCOUNT           # uncomment/set if your cluster requires it

# --------------------------------------------------------------------------------------
# GPU pipeline (stages 00-05). If compute nodes are OFFLINE, this runs GPU_ONLY and you
# run the API judge (06) + analysis (07) afterward on a login node (see HPC_HANDOFF.md).
# If compute nodes DO have internet, drop GPU_ONLY=1 to run 00-07 in one job.
# --------------------------------------------------------------------------------------
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

# --- environment: adapt to your cluster (module load OR conda activate) ---------------
# module load cuda/12.4 || true
# source ~/miniconda3/etc/profile.d/conda.sh && conda activate defense-correlation
PY="${PY:-python}"

# --- secrets (set these before sbatch, or in your shell rc; never commit them) --------
export HF_TOKEN="${HF_TOKEN:?set HF_TOKEN}"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
# REQUESTY_API_KEY only needed for stage 06 (judge); harmless to export here.
export REQUESTY_API_KEY="${REQUESTY_API_KEY:-}"

# --- offline HF cache: weights were pre-staged on the login node (see handoff) --------
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

CONFIG="${CONFIG:-configs/hpc_llama2.yaml}"
echo "GPU:"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

CONFIG="$CONFIG" PY="$PY" GPU_ONLY=1 bash scripts/run_pipeline.sh
echo "GPU stages done. Now run the judge+analysis where there is internet:"
echo "  CONFIG=$CONFIG API_ONLY=1 bash scripts/run_pipeline.sh"
