"""Stage 43: over-refusal, composition estimate against measured stack, on both benign sets.

The paper reported the union of the members' individual false refusals, 0.60, and called it
the composition estimate for the assembled stack. The assembled stack was in fact run on the
same prompts, and it refuses 0.83: the estimate understates the deployed cost by 23 points,
and the stack refuses 23 benign prompts that no individual member refuses. The same gap
appears without the guard (0.56 against 0.82) and on the independent OR-Bench sample
(0.897 against 0.995, or 398 of 400 prompts).

That matters beyond a corrected number. The independence prediction, 1 - prod(1 - f_i), is
offered in the ensemble literature as the cost of combining members that fail on different
inputs, and the measured stack exceeds it on the matched set. So the consolation that
overlapping false refusals are cheaper than disjoint ones, which holds for the union, does
not describe the deployed pipeline: stacking induces refusals that no member produces alone,
because the members do not compose as an OR of block decisions. The priming prompt and the
perturbation aggregator change what the target generates, and the combination refuses where
neither does.

Writes results/<run>/frr_composition.json and prints the LaTeX body for tab:frr.

Usage: python scripts/43_frr_composition.py --run v2_vicuna
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
import sys

sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from dcorr.config import load_config                 # noqa: E402
from dcorr.judge import RefusalScorer                # noqa: E402

MEMBERS7 = ["refusal_prime", "llamaguard", "probe", "smoothllm", "probe_b",
            "ppl_filter", "token_anomaly"]
NAME = {
    "refusal_prime": "Refusal-priming prompt",
    "llamaguard": "Guard model (8B)",
    "probe": "Probe, middle layer",
    "smoothllm": "Perturbation aggregator",
    "probe_b": "Probe, early layer",
    "ppl_filter": "Windowed likelihood filter",
    "token_anomaly": "Token-anomaly statistic",
    "undefended": "\\emph{Undefended target alone}",
}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - m) / d, (c + m) / d)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="v2_vicuna")
    ap.add_argument("--nolg-run", default="v2_vicuna_nolg")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = ap.parse_args()

    cfg = load_config(str(REPO / "configs" / f"{args.run}.yaml"))
    scorer = RefusalScorer(model_name=cfg.get_path("target.model_id"))

    def refusals(p: Path) -> dict:
        out = {}
        if not p.exists():
            return out
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                key = r.get("benign_id") or r["prompt"]
                out[key] = scorer.is_refusal(r.get("response", ""), bool(r.get("blocked")))
        return out

    res = {"run": args.run, "seeds": args.seeds, "sets": {}}
    for bset, sfx in (("jbb", "_benign"), ("orbench", "_benign_orbench")):
        per_seed = []
        for s in args.seeds:
            sd = REPO / "results" / args.run / f"seed{s}"
            per = {m: refusals(sd / f"stage04_{m}{sfx}.jsonl") for m in MEMBERS7}
            per = {m: v for m, v in per.items() if v}
            if not per:
                continue
            keys = sorted(set().union(*[set(v) for v in per.values()]))
            und = refusals(sd / f"stage04_undefended{sfx}.jsonl")
            stack = refusals(sd / f"stage05_stack{sfx}.jsonl")
            nolg = refusals(REPO / "results" / args.nolg_run / f"seed{s}"
                            / f"stage05_stack{sfx}.jsonl")

            def rate(d):
                return (sum(d.values()), len(d)) if d else (0, 0)

            u7 = sum(1 for k in keys if any(per[m].get(k) for m in per))
            m6 = [m for m in per if m != "llamaguard"]
            u6 = sum(1 for k in keys if any(per[m].get(k) for m in m6))
            ind7 = 1.0
            for m in per:
                ind7 *= 1 - sum(per[m].values()) / len(per[m])
            ind6 = 1.0
            for m in m6:
                ind6 *= 1 - sum(per[m].values()) / len(per[m])

            per_seed.append({
                "n": len(keys),
                "members": {m: sum(v.values()) for m, v in per.items()},
                "undefended": rate(und),
                "union7": u7, "union6": u6,
                "if_independent7": 1 - ind7, "if_independent6": 1 - ind6,
                "measured7": rate(stack), "measured6": rate(nolg),
                # the property that makes the gap interpretable: is the stack a superset?
                "stack_only": sum(1 for k in keys
                                  if stack.get(k) and not any(per[m].get(k) for m in per)),
                "member_only": sum(1 for k in keys
                                   if any(per[m].get(k) for m in per) and k in stack
                                   and not stack.get(k)),
            })
        if per_seed:
            res["sets"][bset] = per_seed

    out = REPO / "results" / args.run / "frr_composition.json"
    out.write_text(json.dumps(res, indent=1), encoding="utf-8")

    for bset, rows in res["sets"].items():
        r = rows[0]
        same = all(x["measured7"] == r["measured7"] and x["union7"] == r["union7"] for x in rows)
        n = r["n"]
        print(f"\n=== {bset}  n={n}  replicates={len(rows)}  "
              f"{'identical across replicates' if same else 'VARIES ACROSS REPLICATES'}")
        for m in MEMBERS7:
            if m in r["members"]:
                k = r["members"][m]
                lo, hi = wilson(k, n)
                print(f"  {NAME[m]:30s} {k/n:.3f}  [{lo:.2f}, {hi:.2f}]")
        k, nn = r["undefended"]
        if nn:
            print(f"  {'undefended target alone':30s} {k/nn:.3f}")
        for tag, u, ind, meas, only in (
                ("seven members", r["union7"], r["if_independent7"], r["measured7"],
                 r["stack_only"]),
                ("six, no guard", r["union6"], r["if_independent6"], r["measured6"], None)):
            print(f"  -- {tag}: union {u/n:.3f}   if independent {ind:.3f}   "
                  f"measured {meas[0]}/{meas[1]}"
                  f"{'' if not meas[1] else f' = {meas[0]/meas[1]:.3f}'}"
                  + (f"   refused by the stack and by no member: {only}" if only is not None
                     else ""))
    print(f"\nwrote {out.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
