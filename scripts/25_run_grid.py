"""Run the v2 grid for one config, seed by seed, unattended (plan Phase 2).

Each stage is its own subprocess (a fresh CUDA context per stage, and a crash costs one
stage, not the grid). Every stage resumes by its result keys, so re-running this script
after an interruption continues where it stopped. Timings go to results/<run>/timings.jsonl
and each stage's output to results/<run>/logs/seed<k>_<stage>.log.

  03  undefended adaptive + static attacks (the matched undefended control)
  04  every enabled defense, adaptive + static, plus benign responses
  12  undefended benign responses (the base model's refusal floor)
  05  the assembled stack, adaptive + static, plus benign responses
  24  same-input cross-evaluation of every defense on every search's prompts
  B12/B4/B5  the same benign passes on the held-out OR-Bench-Hard set (no attacks)

Grading (06) is separate: it calls a paid API and is run once the GPU stages are done.

Usage:
  python scripts/25_run_grid.py --config configs/v2_vicuna.yaml --seeds 0 1 2
  python scripts/25_run_grid.py --config configs/v2_vicuna_nolg.yaml --seeds 0 --stages 05
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from dcorr.config import load_config  # noqa: E402
from dcorr.runtime import host_provenance  # noqa: E402

STAGES = {                       # stage -> (script, extra arguments)
    "03": ("03_positive_control.py", []),
    "04": ("04_run_attacks.py", []),
    "12": ("12_undefended_benign.py", []),
    "05": ("05_run_stack.py", []),
    "24": ("24_cross_eval.py", []),
    "B12": ("12_undefended_benign.py", ["--benign-set", "orbench"]),
    "B4": ("04_run_attacks.py", ["--benign-only", "--benign-set", "orbench"]),
    "B5": ("05_run_stack.py", ["--benign-only", "--benign-set", "orbench"]),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--stages", nargs="+", default=list(STAGES), choices=list(STAGES))
    ap.add_argument("--no-benign", action="store_true", help="skip benign passes in 04/05")
    ap.add_argument("--only", nargs="*", help="restrict stage 04 to these defenses")
    args = ap.parse_args()

    cfg = load_config(args.config)
    run_dir = cfg.run_dir
    logs = run_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    timings = run_dir / "timings.jsonl"

    for k in args.seeds:
        for stage in args.stages:
            script, extra = STAGES[stage]
            cmd = [sys.executable, str(REPO / "scripts" / script),
                   "--config", args.config, "--seed-index", str(k), *extra]
            if args.no_benign and stage in ("04", "05"):
                cmd.append("--no-benign")
            if stage in ("04", "B4") and args.only:
                cmd += ["--only", *args.only]
            log = logs / f"seed{k}_{stage}.log"
            print(f"[grid] seed {k} stage {stage} -> {log}", flush=True)
            t0 = time.time()
            with open(log, "a", encoding="utf-8") as f:
                f.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(cmd)}\n")
                f.flush()
                rc = subprocess.run(cmd, cwd=REPO, stdout=f, stderr=subprocess.STDOUT,
                                    env=dict(os.environ, PYTHONIOENCODING="utf-8")).returncode
            dt = time.time() - t0
            with open(timings, "a", encoding="utf-8") as f:
                # The grid driver is the outermost thing that runs on a given node, so its
                # record is where a reader should be able to see which node that was. The
                # child stage records the same, because a resumed run can spread one stage's
                # chunks across several allocations.
                f.write(json.dumps({"config": args.config, "seed": k, "stage": stage,
                                    "seconds": round(dt, 1), "returncode": rc,
                                    "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
                                    **host_provenance()}) + "\n")
            print(f"[grid] seed {k} stage {stage}: rc={rc} in {dt / 60:.1f} min", flush=True)
            if rc != 0:
                print(f"[grid] stopping: see {log}", flush=True)
                return rc
    print("[grid] done.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
