#!/usr/bin/env bash
#SBATCH --job-name=dcorr_D
#SBATCH --partition=gpu_x450
#SBATCH --gres=gpu:h100:1
#SBATCH --exclude=gpu0001
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=36:00:00
#SBATCH --output=dcorr_D_%j.out
#SBATCH --error=dcorr_D_%j.err
#
# Task D: the primary configuration re-run with EXTERNAL thresholds.
#
# Environment copied verbatim from scripts/slurm_pipeline.sbatch, which is the configuration
# that produced the reported runs: venv rather than modules, HF cache on /project, compute
# nodes air-gapped and not mounting /home, secrets from the repo .env.
#
# Stages 04 (the grid) and 05 (direct stack attack) only. Data prep, probe training and the
# positive-control gate are already done and unchanged; calibration is supplied rather than
# recomputed, so stages 00/01/02/03/08 are deliberately skipped. Judging (06) needs internet
# and runs on the login node afterwards.
#
# Both stages are resumable and skip keys already present, so re-running is safe.
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

CFG=configs/hpc_vicuna_autodan_extthr.yaml
SRC=results/hpc_vicuna_autodan
DST=results/hpc_vicuna_autodan_extthr

echo "node: $(hostname)  python: $PY"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
# Query only the device Slurm gave us. Listing every GPU on the node is misleading: a
# neighbouring job's memory shows up and looks like our own.
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader \
  -i "${CUDA_VISIBLE_DEVICES:-0}" || nvidia-smi --query-gpu=index,memory.used --format=csv,noheader || true

# --- install the EXTERNAL thresholds as this run's calibration.json --------------------
# Written in the schema stage 04/05 expect, so _merge_calibration picks them up unchanged.
mkdir -p "$DST"
"$PY" - <<'PYEOF'
import json, pathlib
src = json.load(open('results/hpc_vicuna_autodan/calibration_external.json', encoding='utf-8'))
dst = pathlib.Path('results/hpc_vicuna_autodan_extthr/calibration.json')
if dst.is_file():
    print(f"[D] {dst} already present, leaving it alone")
else:
    out = {
        "target_fpr": src["target_fpr"],
        "model_id": src["model_id"],
        "thresholds": src["thresholds"],
        "fpr_check": {k: {"threshold": v["threshold_external"],
                          "realized_fpr": v["external_block_rate_at_external_thr"],
                          "n_benign": src["n_external"]}
                      for k, v in src["comparison"].items()},
        "_provenance": ("EXTERNAL calibration: thresholds are the (1 - target_fpr) quantile "
                        "over 805 AlpacaEval instructions, not the 100 in-sample evaluation "
                        "benign prompts. Copied from calibration_external.json by "
                        "scripts/slurm_taskD.sbatch."),
    }
    dst.write_text(json.dumps(out, indent=2), encoding='utf-8')
    print(f"[D] wrote {dst} with external thresholds: {out['thresholds']}")
PYEOF

# --- seed the self-transfer source ------------------------------------------------------
if [ ! -f "$DST/stage04_undefended.jsonl" ]; then
  cp -v "$SRC/stage04_undefended.jsonl" "$DST/stage04_undefended.jsonl"
fi

echo ""; echo "=== D: stage 04 grid under external thresholds ==="
"$PY" scripts/04_run_attacks.py --config "$CFG"
s4=$?
echo "[D] stage04 exit=$s4"

echo ""; echo "=== D: stage 05 direct stack attack ==="
"$PY" scripts/05_run_stack.py --config "$CFG"
s5=$?
echo "[D] stage05 exit=$s5"

echo ""
echo "stage04=$s4 stage05=$s5"
if [ "$s4" -ne 0 ] || [ "$s5" -ne 0 ]; then
  echo "[D][FAIL] a GPU stage exited non-zero; not reporting success" >&2
  exit 1
fi
echo "Now judge + analyse on the LOGIN node (needs internet):"
echo "  python scripts/06_judge_gold.py --config $CFG && python scripts/07_analyze.py --config $CFG"
