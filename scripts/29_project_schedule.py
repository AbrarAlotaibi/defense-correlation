"""Project the full-run schedule from a smoke run's measured per-defense times.

The smoke configs run the v2 attack budget on a few behaviours, so per-behaviour cost is
measured rather than guessed. This scales it to the real behaviour count and seed count and
prints hours per stage, per seed and in total, so Phase 2 can be scheduled honestly.

Per-defense times come from the file timestamps in results/<run>/seed<k>, which is when each
stage04_<defense>.jsonl was completed; the first defense is timed from the stage start in
timings.jsonl. Times therefore include that defense's benign pass.

Usage:
  python scripts/29_project_schedule.py --smoke v2_vicuna_smoke --n 100 --seeds 3
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _timings(run_dir: Path) -> list[dict]:
    p = run_dir / "timings.jsonl"
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", required=True, help="smoke run name under results/")
    ap.add_argument("--smoke-n", type=int, default=5, help="behaviours in the smoke run")
    ap.add_argument("--n", type=int, default=100, help="behaviours in the real run")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--n-benign-jbb", type=int, default=100)
    ap.add_argument("--n-benign-orbench", type=int, default=400)
    ap.add_argument("--seed-dir", default="seed0")
    args = ap.parse_args()

    run_dir = REPO / "results" / args.smoke
    sd = run_dir / args.seed_dir
    if not sd.exists():
        raise SystemExit(f"no {sd}")
    scale = args.n / args.smoke_n

    # Per-defense times come from defense_timings.jsonl, which stage 04 writes as it goes.
    # File timestamps cannot be used: a resumed run leaves earlier defenses' files untouched,
    # so the gaps between them span whatever downtime there was (that produced a -200 min
    # "cost" for ppl_filter the first time this was tried).
    seen: dict[str, dict] = {}
    dt = run_dir / "defense_timings.jsonl"
    if dt.exists():
        for line in dt.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                seen[r["defense"]] = r          # last run of each defense wins
    rows = [(n, r["seconds"]) for n, r in sorted(seen.items(), key=lambda kv: -kv[1]["seconds"])]
    if not rows:
        print("# no defense_timings.jsonl yet: per-defense split unavailable; the stage "
              "totals below are still exact")

    print(f"# projection from {args.smoke}/{args.seed_dir}: {args.smoke_n} -> {args.n} "
          f"behaviours, {args.seeds} seed(s)\n")
    print(f"{'defense':<24}{'smoke (min)':>14}{'projected (h)':>16}")
    total04 = 0.0
    for name, secs in rows:
        proj = secs * scale / 3600
        total04 += proj
        print(f"{name:<24}{secs / 60:>14.1f}{proj:>16.2f}")
    if rows:
        print(f"{'stage 04 by defense':<24}{'':>14}{total04:>16.2f}  (informational)")

    # Stage totals. Attack stages scale with the behaviour count; the benign stages scale with
    # their own prompt-set size, which the smoke run also caps via data.eval_limit.
    benign_scale = {"12": args.n_benign_jbb / args.smoke_n,
                    "B12": args.n_benign_orbench / args.smoke_n,
                    "B4": args.n_benign_orbench / args.smoke_n,
                    "B5": args.n_benign_orbench / args.smoke_n}
    stages = {}
    for t in _timings(run_dir):
        if t["returncode"] == 0:
            stages[t["stage"]] = max(stages.get(t["stage"], 0.0), t["seconds"])
    per_seed = 0.0
    for stage, secs in sorted(stages.items()):
        sc = benign_scale.get(stage, scale)
        h = secs * sc / 3600
        per_seed += h
        note = f"  (x{sc:.0f} prompts)" if stage in benign_scale else ""
        print(f"{'stage ' + stage:<24}{secs / 60:>14.1f}{h:>16.2f}{note}")
    print(f"\nper seed: {per_seed:.1f} h")
    print(f"{args.seeds} seeds: {per_seed * args.seeds:.1f} h "
          f"({per_seed * args.seeds / 24:.1f} days of continuous GPU)")
    print("\nNot included: the six-layer stack run, the Llama-2 replication, pair runs "
          "(05 --layers), grading (API, no GPU) and analysis (CPU).")


if __name__ == "__main__":
    main()
