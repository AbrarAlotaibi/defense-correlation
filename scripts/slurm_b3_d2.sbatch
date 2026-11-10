#!/usr/bin/env bash
#SBATCH --job-name=dcorr_b3d2
#SBATCH --partition=longrungpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=dcorr_b3d2_%j.out
#SBATCH --error=dcorr_b3d2_%j.err
#
# Two outstanding measurements, in one job. Environment copied verbatim from
# scripts/slurm_pipeline.sbatch, which is the configuration that produced the reported runs:
# venv rather than modules, HF cache on /project, compute nodes air-gapped and not mounting
# /home, secrets read from the repo .env.
#
#   B3  the undefended model on the benign set. No run has ever recorded what the bare model
#       refuses unprompted, so the per-defense FRRs of H3 are raw rather than attributable.
#       ~100 greedy decodes, minutes.
#
#   D2  direct adaptive attack on the six-layer stack with Llama Guard removed. The primary
#       run's intersection-versus-direct check compares 0.010 against 0.000, both on the
#       floor. Over the remaining six the intersection predicts 0.190, so this version has
#       resolution. Hours.
#
# Both stages are resumable and skip keys already present, so re-running is safe.
# Judging (stage 06) needs internet and is NOT done here; the tail prints the login-node
# command.
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

PRIMARY=configs/hpc_vicuna_autodan.yaml
NOLG=configs/hpc_vicuna_autodan_nolg.yaml

echo "node: $(hostname)  python: $PY"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader || true

echo ""; echo "=== B3: undefended model on the benign set ==="
"$PY" scripts/12_undefended_benign.py --config "$PRIMARY"
b3=$?
echo "[B3] exit=$b3"

echo ""; echo "=== D2: seeding the no-Llama-Guard run from the primary ==="
# Thresholds are NOT recalibrated: the six layers must sit at exactly the operating points
# the paper reports, so calibration.json is copied rather than regenerated. The undefended
# harmful file is the self-transfer seed stage 05 reads.
mkdir -p results/hpc_vicuna_autodan_nolg
for f in calibration.json stage04_undefended.jsonl; do
  if [ ! -f "results/hpc_vicuna_autodan_nolg/$f" ]; then
    cp -v "results/hpc_vicuna_autodan/$f" "results/hpc_vicuna_autodan_nolg/$f"
  fi
done

echo ""; echo "=== D2: direct attack on the six-layer stack ==="
"$PY" scripts/05_run_stack.py --config "$NOLG"
d2=$?
echo "[D2] exit=$d2"

echo ""
echo "B3 exit=$b3  D2 exit=$d2"
echo "Now judge on the LOGIN node (needs internet):"
echo "  cd $SLURM_SUBMIT_DIR && source /project/$USER/venvs/dcorr/bin/activate && \\"
echo "  HF_HOME=/project/$USER/.cache/huggingface python scripts/06_judge_gold.py --config $NOLG"
