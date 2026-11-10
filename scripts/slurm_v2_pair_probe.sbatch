#!/usr/bin/env bash
#SBATCH --job-name=dcorr_pair_probe
#SBATCH --partition=gpu_x450,longrungpu
#SBATCH --gres=gpu:h100:1
#SBATCH --exclude=gpu0001
#SBATCH --cpus-per-task=8
#SBATCH --mem=110G
#SBATCH --time=02:00:00
#SBATCH --output=pair_probe_%j.out
#SBATCH --error=pair_probe_%j.err
#
# Does the attacker model cooperate? Everything else about the second attack family is
# already tested on CPU fakes (tests/test_pair.py: objective equivalence with the genetic
# arm, record schema, determinism, resume keying, parser). The one thing fakes cannot tell us
# is whether Qwen2.5-7B-Instruct will actually write jailbreak prompts when instructed to, or
# refuse. A refusing attacker produces a weak arm for a reason that has nothing to do with the
# defenses, and would quietly understate attack success across the whole family.
#
# So: 8 behaviours, undefended arm only, 2 streams by 3 iterations. Then read
# pair_parse_failures and the proposed prompts before committing an array to it. Writes to
# results/v2_vicuna_pair_probe, never to the directory the paper reads.
#
# MEMORY. Two 7B models resident at once, the target in fp16 and the attacker in bf16, so
# ~28 GiB of weights plus activations. Fits an 80 GiB H100 comfortably; the 110G host request
# is for the two tokenizers and the dataloading, not the GPU.
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

CONFIG=configs/v2_vicuna_pair_probe.yaml
echo "node $(hostname)  config $CONFIG"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# Stage 03, not stage 04. The undefended arm is produced by the positive control, because it
# doubles as one: "undefended" is not in enabled_defenses, so the first version of this job
# passed --only undefended to stage 04, matched nothing, and exited clean in two minutes
# having run no attack at all. A job that succeeds without doing the work is worse than one
# that fails, so the check below now treats zero pair rows as a failure.
"$PY" scripts/03_positive_control.py --config "$CONFIG" --seed-index 0
rc=$?
echo "stage 03 exit=$rc"

OUT=results/v2_vicuna_pair_probe/seed0/stage04_undefended.jsonl
if [ ! -f "$OUT" ]; then
  echo "NO OUTPUT at $OUT: the attack did not run. Not reporting success."
  exit 1
fi
if true; then
  echo
  echo "=== the number this job exists to produce ==="
  "$PY" - "$OUT" <<'PYEOF'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8") if l.strip()]
pair = [r for r in rows if r.get("attack") == "pair"]
if not pair:
    print("NO pair rows: the arm did not run")
    raise SystemExit(1)
missing = [k for k in ("pair_streams", "pair_parse_failures", "iterations",
                       "attacker_model") if k not in pair[0]]
if missing:
    print(f"CANNOT REPORT: the stored rows lack {missing}. The row builder in "
          f"dcorr/run_defense.py is dropping the arm's own fields, so there is no "
          f"parse-failure rate to read. Printing a percentage over a zero denominator "
          f"would have looked like a pass.")
    raise SystemExit(1)
fails = sum(r.get("pair_parse_failures") or 0 for r in pair)
budget = sum((r["pair_streams"]) * (r["iterations"]) for r in pair)
print(f"behaviours        {len(pair)}")
print(f"attacker          {pair[0]['attacker_model']}")
print(f"budget            {pair[0]['pair_streams']} streams x "
      f"{pair[0]['iterations']} iterations = {budget} attacker calls")
print(f"parse failures    {fails}  ({100 * fails / budget:.1f}%)")
print(f"feasible found    {sum(1 for r in pair if r.get('found_feasible'))}/{len(pair)}")
print(f"evaluated cands   {sum(r.get('n_evals') or 0 for r in pair)}")
print()
print("A high failure rate means the attacker is refusing rather than attacking.")
print("Under ~15% is workable; over ~40% means switch attacker_model_id.")
print()
for r in pair[:3]:
    print(f"--- {r['behaviour_id']}  fit={r.get('in_loop_logprob')}")
    print("    proposed:", (r.get("prompt") or "")[:300].replace("\n", " "))
    print("    response:", (r.get("response") or "")[:200].replace("\n", " "))
PYEOF
fi
exit $rc
