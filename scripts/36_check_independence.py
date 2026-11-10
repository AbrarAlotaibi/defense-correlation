"""Stage 36: show that the replicates and the defenses really are independent searches.

The v1 submission drew every defense's search from one RNG stream, so defenses that neither
rejected a candidate nor changed the objective replayed the same trajectory and their breach
vectors differed only by grader noise. A correlation computed on those vectors measured the
coupling, not the defenses. That is reviewer finding F2, and dcorr/seeding.py fixes it by
deriving a seed from (base, seed_index, defense, behaviour).

A fix in the seeding code is not evidence about the run. This script checks the claim where it
matters, on the produced artefacts, along both axes, and it also checks the converse: the parts
that are supposed to be identical must be identical, or the harness is simply nondeterministic
and the first result would be meaningless.

  across replicates  the same defense under different --seed-index must search differently
  across defenses    within one replicate, two defenses must not share a trajectory
  static baselines   no search is involved, so these MUST match exactly across replicates

Usage:
  python scripts/36_check_independence.py --run results/v2_vicuna --seeds 0 1
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ADAPTIVE = "adaptive"
STATIC = ("static_plain", "static_gcg")


def load(run: Path, seed: int, defense: str) -> dict:
    p = run / f"seed{seed}" / f"stage04_{defense}.jsonl"
    if not p.exists():
        return {}
    out = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            out[(r["behaviour_id"], r["attack"])] = r
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--max-shared-frac", type=float, default=0.10,
                    help="fail if more than this fraction of adaptive prompts coincide")
    args = ap.parse_args()

    run = REPO / args.run if not Path(args.run).is_absolute() else Path(args.run)
    defenses = sorted({p.name[len("stage04_"):-len(".jsonl")]
                       for s in args.seeds
                       for p in (run / f"seed{s}").glob("stage04_*.jsonl")
                       if not p.name.endswith("_benign.jsonl")})
    problems, report = [], {"run": str(run), "seeds": args.seeds, "defenses": defenses}

    # 1. the same defense, different replicate
    across_reps = {}
    for d in defenses:
        for s1, s2 in itertools.combinations(args.seeds, 2):
            a, b = load(run, s1, d), load(run, s2, d)
            keys = [k for k in a if k[1] == ADAPTIVE and k in b]
            if not keys:
                continue
            same_prompt = sum(1 for k in keys if a[k]["prompt"] == b[k]["prompt"])
            same_seed = sum(1 for k in keys
                            if a[k].get("search_seed") == b[k].get("search_seed"))
            frac = same_prompt / len(keys)
            across_reps[f"{d}:seed{s1}-seed{s2}"] = {
                "n": len(keys), "same_prompt": same_prompt, "same_prompt_frac": frac,
                "same_search_seed": same_seed}
            if same_seed:
                problems.append(f"{d}: {same_seed}/{len(keys)} shared search_seed "
                                f"between seed{s1} and seed{s2}")
            if frac > args.max_shared_frac:
                problems.append(f"{d}: {same_prompt}/{len(keys)} identical adaptive prompts "
                                f"between seed{s1} and seed{s2}")
    report["across_replicates"] = across_reps

    # 2. two defenses, same replicate
    across_def = {}
    for s in args.seeds:
        loaded = {d: load(run, s, d) for d in defenses}
        for d1, d2 in itertools.combinations(defenses, 2):
            a, b = loaded[d1], loaded[d2]
            keys = [k for k in a if k[1] == ADAPTIVE and k in b]
            if not keys:
                continue
            same_prompt = sum(1 for k in keys if a[k]["prompt"] == b[k]["prompt"])
            same_seed = sum(1 for k in keys
                            if a[k].get("search_seed") == b[k].get("search_seed"))
            across_def[f"seed{s}:{d1}-{d2}"] = {
                "n": len(keys), "same_prompt": same_prompt,
                "same_prompt_frac": same_prompt / len(keys), "same_search_seed": same_seed}
            if same_seed:
                problems.append(f"seed{s}: {d1} and {d2} share {same_seed}/{len(keys)} "
                                "search seeds")
    report["across_defenses"] = across_def

    # 3. the converse: no search, so it must reproduce exactly
    static = {}
    for d in defenses:
        for s1, s2 in itertools.combinations(args.seeds, 2):
            a, b = load(run, s1, d), load(run, s2, d)
            for atk in STATIC:
                keys = [k for k in a if k[1] == atk and k in b]
                if not keys:
                    continue
                same = sum(1 for k in keys if a[k]["prompt"] == b[k]["prompt"])
                static[f"{d}:{atk}:seed{s1}-seed{s2}"] = {
                    "n": len(keys), "same_prompt": same}
                if same != len(keys):
                    problems.append(f"{d} {atk}: only {same}/{len(keys)} prompts reproduce "
                                    f"across seed{s1} and seed{s2}; a static baseline involves "
                                    "no search and must be identical")
    report["static_baselines"] = static

    worst = max((v["same_prompt_frac"] for v in across_reps.values()), default=0.0)
    worst_d = max((v["same_prompt_frac"] for v in across_def.values()), default=0.0)
    print(f"[36] {len(defenses)} defenses, seeds {args.seeds}")
    print(f"[36] across replicates: shared search seeds "
          f"{sum(v['same_search_seed'] for v in across_reps.values())}, "
          f"worst identical-prompt fraction {worst:.2%}")
    print(f"[36] across defenses:   shared search seeds "
          f"{sum(v['same_search_seed'] for v in across_def.values())}, "
          f"worst identical-prompt fraction {worst_d:.2%}")
    print(f"[36] static baselines reproduce exactly: "
          f"{all(v['same_prompt'] == v['n'] for v in static.values())}")

    report["problems"] = problems
    out = run / "independence_check.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if problems:
        print(f"\n[36] {len(problems)} PROBLEM(S):")
        for p in problems[:20]:
            print("  -", p)
        print(f"[36] wrote {out}")
        return 1
    print(f"[36] no problems. wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
