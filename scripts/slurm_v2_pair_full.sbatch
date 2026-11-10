#!/usr/bin/env bash
#SBATCH --job-name=dcorr_pair
#SBATCH --partition=gpu_x450,longrungpu
#SBATCH --gres=gpu:h100:1
#SBATCH --exclude=gpu0001
#SBATCH --cpus-per-task=8
#SBATCH --mem=110G
#SBATCH --time=48:00:00
#SBATCH --array=0-2
#SBATCH --output=pair_full_%A_%a.out
#SBATCH --error=pair_full_%A_%a.err
#
# The second attack family at full size: every member plus the undefended control, all 100
# behaviours, one replicate per array task. This is the arm that answers whether the E1
# verdict survives a search that works by a different mechanism, which is the reviewer
# question the reviewers of the earlier version raised and which three classes of one search
# cannot answer.
#
# VALIDATED FIRST. slurm_v2_pair_probe.sbatch ran 8 behaviours on the undefended arm (job
# 2352814) and established the two things CPU tests cannot: Qwen2.5-7B-Instruct writes the
# jailbreak prompts rather than refusing, 0 parse failures in 48 attacker calls, and the
# configured budget now reaches the search. That probe also found four silent defects,
# including attack.pair having no effect at all on the budget, so do not run this without
# re-running the probe after any change to dcorr/attacks/pair.py or the stages.
#
# WHY STAGE 03 AND THEN STAGE 04. The undefended arm is produced by the positive control,
# not by stage 04 -- "undefended" is not in enabled_defenses. Without stage 03 the pair arm
# would have no no-defense control, and therefore no seed-pair null to read its pair values
# against, which is the control Section 5.3 argues the whole analysis needs.
#
# WHY THE CONFIG KEEPS run_name: v2_vicuna. Rows append to the existing
# results/v2_vicuna/seed<k>/stage04_<member>.jsonl beside the genetic and static arms, keyed
# by attack="pair". Stage 04 resumes by (model, defense, behaviour, attack), so the arms
# already there are skipped and nothing is recomputed. One gold.jsonl then holds both
# families and every estimand can be recomputed per family with no new plumbing.
#
# RUNTIME, and why the wall is what it is. The probe did 8 behaviours on one arm at a quarter
# of this budget in under four minutes of search. Scaled to 8 arms by 100 behaviours by 24
# attacker calls, that is of order 20 to 50 hours per replicate, and the spread is wide
# because the aggregator generates q=4 copies per candidate and the guard reloads per round.
# Every stage persists per behaviour chunk and resumes by result key, so hitting the wall
# costs only the chunk in flight: requeue the same array task and it continues. longrungpu is
# listed for that reason as much as for availability.
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

CONFIG=configs/v2_vicuna_pair.yaml
SEED="${SLURM_ARRAY_TASK_ID:-0}"
RUN=results/v2_vicuna/seed$SEED

echo "node $(hostname)  seed $SEED  second attack family, full roster"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# Refuse to start on a configuration whose budget is not the one being claimed, since that is
# precisely the failure the probe uncovered and it is invisible in the output.
"$PY" - <<'PYEOF' || exit 1
import sys
sys.path.insert(0, ".")
from dcorr.config import load_config
p = load_config("configs/v2_vicuna_pair.yaml").get_path("attack.pair") or {}
need = {"enabled": True, "streams": 4, "iterations": 6}
bad = {k: p.get(k) for k, v in need.items() if p.get(k) != v}
if bad:
    print(f"[pair] attack.pair is not the validated configuration: {bad}")
    sys.exit(1)
print(f"[pair] budget {p['streams']} streams x {p['iterations']} iterations "
      f"= {p['streams'] * p['iterations']} attacker calls per behaviour")
print(f"[pair] attacker {p.get('attacker_model_id')}")
PYEOF

if [ ! -f "$RUN/stage04_undefended.jsonl" ]; then
  echo "[pair] $RUN/stage04_undefended.jsonl is missing: the genetic arm has not run for this"
  echo "[pair] seed, so there is nothing to append to and no control to compare against."
  exit 1
fi

before=$(cat "$RUN"/stage04_*.jsonl 2>/dev/null | grep -c '"attack": "pair"' || true)
echo "[pair] pair rows already present for this seed: ${before:-0}"

run () { echo "=== seed $SEED :: $*"; "$PY" "$@"; echo "    exit=$?"; }

# The undefended control first: if the attacker turns out to be unusable on this seed, that
# shows up in twenty minutes rather than after a day of defended arms.
run scripts/03_positive_control.py --config "$CONFIG" --seed-index "$SEED"
run scripts/04_run_attacks.py --config "$CONFIG" --seed-index "$SEED" --no-benign

after=$(cat "$RUN"/stage04_*.jsonl 2>/dev/null | grep -c '"attack": "pair"' || true)
echo
echo "[pair] pair rows now: ${after:-0}  (was ${before:-0})"
if [ "${after:-0}" -le "${before:-0}" ]; then
  echo "[pair] NO new pair rows were written. Not reporting success."
  exit 1
fi

echo
echo "next, with internet, once every seed has finished:"
echo "  python scripts/06_judge_gold.py --config configs/v2_vicuna.yaml --seed-index $SEED"
echo "  python scripts/35_resolve_abstentions.py --config configs/v2_vicuna.yaml \\"
echo "      --seed-index $SEED --grader deepseek/deepseek-v4-pro-0813 \\"
echo "      --grader-base-url https://router.requesty.ai/v1 \\"
echo "      --grader-key-env REQUESTY_API_KEY --grader-max-tokens 4000"
