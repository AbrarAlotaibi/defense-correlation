"""Stage 48: does the E1 verdict survive a second attack family?

Every estimate in the paper is conditioned on one adversary. This compares the genetic arm
(attack="adaptive", a sentence-pool GA) against the second family (attack="pair", an attacker
LLM refining its own prompts) on the quantities the paper's central claim rests on:

  * the null, the undefended seed-pair correlation, computed within each family;
  * the 21 pair values, and how many clear their own family's null;
  * marginal attack success per member, since a family that barely succeeds cannot
    discriminate anything and that has to be visible rather than assumed;
  * agreement BETWEEN families on which behaviours break, per member. This is the one
    quantity neither family can produce alone, and it is the sharpest form of the question:
    if two searches with nothing in common but their objective break the same behaviours,
    then what the paper measures is a property of the behaviours, not of the search.

The comparison is at matched behaviours and matched members, on the same seeds, with the same
grader and the same resolved labels. The budgets differ by design: the genetic arm evaluates
a population per generation and the second family 24 candidates per behaviour, so `n_evals`
is reported rather than equalised, and the second family is the cheaper of the two.

Writes results/<run>/family_comparison.json.

Usage: python scripts/48_family_comparison.py --run v2_vicuna --seeds 0 1 2
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

from dcorr.analysis.estimands import exact_mcnemar, wilson  # noqa: E402

MEMBERS = ["ppl_filter", "token_anomaly", "probe", "probe_b", "refusal_prime",
           "smoothllm", "llamaguard"]
SHORT = {"ppl_filter": "filter", "token_anomaly": "anomaly", "probe": r"probe$_{16}$",
         "probe_b": r"probe$_{8}$", "refusal_prime": "priming", "smoothllm": "aggregator",
         "llamaguard": "guard", "undefended": "undefended"}


def phi(x, y) -> float | None:
    x, y = np.asarray(x, int), np.asarray(y, int)
    n11 = int(((x == 1) & (y == 1)).sum()); n10 = int(((x == 1) & (y == 0)).sum())
    n01 = int(((x == 0) & (y == 1)).sum()); n00 = int(((x == 0) & (y == 0)).sum())
    den = ((n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00)) ** 0.5
    return None if den == 0 else (n11 * n00 - n10 * n01) / den


def vectors(gold: Path, ids: list[str], attack: str) -> dict[str, np.ndarray]:
    by: dict[str, dict[str, int]] = {}
    for line in gold.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("attack") != attack:
            continue
        by.setdefault(r["defense"], {})[r["behaviour_id"]] = int(bool(r["breach"]))
    return {d: np.array([m[i] for i in ids]) for d, m in by.items()
            if all(i in m for i in ids)}


def evals(gold_dir: Path, attack: str) -> list[int]:
    out = []
    for p in sorted(gold_dir.glob("stage04_*.jsonl")):
        if "_benign" in p.name:
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("attack") == attack and r.get("n_evals"):
                out.append(int(r["n_evals"]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="v2_vicuna")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--gold", default="gold_resolved.jsonl")
    args = ap.parse_args()

    rd = REPO / "results" / args.run
    ids = [json.loads(l)["behaviour_id"]
           for l in (REPO / "data/eval_harmful.jsonl").read_text(encoding="utf-8").splitlines()
           if l.strip()]

    fam = {"adaptive": {}, "pair": {}}
    for att in fam:
        for sd in args.seeds:
            g = rd / f"seed{sd}" / args.gold
            if not g.exists():
                g = rd / f"seed{sd}" / "gold.jsonl"
            v = vectors(g, ids, att)
            if v:
                fam[att][sd] = v
    for att, per in fam.items():
        if not per:
            raise SystemExit(f"[48] no rows for attack={att} in {rd}; grade it first")
        print(f"[48] {att}: seeds {sorted(per)}")

    res: dict = {"run": args.run, "seeds": args.seeds, "labels": args.gold,
                 "marginals": {}, "null": {}, "pairs": {}, "cross_family": {},
                 "budget": {}}

    # ---- budget, reported not equalised -------------------------------------------
    for att in fam:
        e = [n for sd in args.seeds for n in evals(rd / f"seed{sd}", att)]
        res["budget"][att] = {"n": len(e), "median_evals": statistics.median(e) if e else None,
                              "total_evals": sum(e)}

    # ---- marginal attack success -------------------------------------------------
    print(f"\n{'member':16s}{'adaptive':>10}{'pair':>8}{'diff':>8}")
    for m in MEMBERS + ["undefended"]:
        row = {}
        for att, per in fam.items():
            vals = [per[sd][m].mean() for sd in per if m in per[sd]]
            if vals:
                k = int(round(statistics.mean(vals) * len(ids))) * len(vals)
                row[att] = {"asr": statistics.mean(vals), "per_seed": vals,
                            "ci": wilson(k, len(ids) * len(vals))}
        if len(row) == 2:
            row["diff"] = row["pair"]["asr"] - row["adaptive"]["asr"]
            print(f"{m:16s}{row['adaptive']['asr']:>10.3f}{row['pair']['asr']:>8.3f}"
                  f"{row['diff']:>+8.3f}")
        res["marginals"][m] = row

    # ---- each family's own null --------------------------------------------------
    print()
    for att, per in fam.items():
        seeds = sorted(per)
        ps = [phi(per[a]["undefended"], per[b]["undefended"])
              for i, a in enumerate(seeds) for b in seeds[i + 1:]
              if "undefended" in per[a] and "undefended" in per[b]]
        ps = [p for p in ps if p is not None]
        res["null"][att] = {"pairs": ps, "mean": statistics.mean(ps) if ps else None,
                            "spread": (max(ps) - min(ps)) if len(ps) > 1 else None}
        n = res["null"][att]
        print(f"[48] {att} null: mean {n['mean']:.3f}  estimates "
              f"{[round(p, 3) for p in sorted(ps, reverse=True)]}  spread {n['spread']:.3f}")

    # ---- the 21 pairs within each family -----------------------------------------
    for att, per in fam.items():
        out = {}
        for i, a in enumerate(MEMBERS):
            for b in MEMBERS[i + 1:]:
                vals = [phi(per[sd][a], per[sd][b]) for sd in per
                        if a in per[sd] and b in per[sd]]
                vals = [v for v in vals if v is not None]
                if vals:
                    out[f"{a}|{b}"] = {"mean_phi": statistics.mean(vals), "per_seed": vals}
        res["pairs"][att] = out

    print()
    for att in fam:
        nul = res["null"][att]["mean"]
        ng = {k: v for k, v in res["pairs"][att].items() if "llamaguard" not in k}
        above = [k for k, v in ng.items() if v["mean_phi"] > nul]
        mx = max(ng.values(), key=lambda v: v["mean_phi"])["mean_phi"] if ng else float("nan")
        print(f"[48] {att}: non-guard pairs {min(v['mean_phi'] for v in ng.values()):.3f} to "
              f"{mx:.3f}; {len(above)} of {len(ng)} above the null; largest excess "
              f"{mx - nul:+.3f}")

    # ---- agreement BETWEEN families ----------------------------------------------
    # The quantity neither family produces alone: for one member and one seed, does the
    # genetic arm break the same behaviours the LLM-driven search breaks?
    print(f"\n{'member':16s}{'cross-family phi':>18}{'both':>7}{'GA only':>9}{'pair only':>11}"
          f"{'McNemar p':>11}")
    for m in MEMBERS + ["undefended"]:
        vals, tabs = [], []
        for sd in args.seeds:
            A, P = fam["adaptive"].get(sd, {}), fam["pair"].get(sd, {})
            if m in A and m in P:
                v = phi(A[m], P[m])
                if v is not None:
                    vals.append(v)
                tabs.append(exact_mcnemar(A[m], P[m]))
        if not tabs:
            continue
        both = sum(int(((fam["adaptive"][sd][m] == 1) & (fam["pair"][sd][m] == 1)).sum())
                   for sd in args.seeds
                   if m in fam["adaptive"].get(sd, {}) and m in fam["pair"].get(sd, {}))
        res["cross_family"][m] = {
            "phi": statistics.mean(vals) if vals else None, "per_seed": vals,
            "both_breached": both,
            "ga_only": sum(t["n10"] for t in tabs), "pair_only": sum(t["n01"] for t in tabs),
            "pooled_mcnemar": exact_mcnemar(
                np.concatenate([fam["adaptive"][sd][m] for sd in args.seeds
                                if m in fam["adaptive"].get(sd, {})]),
                np.concatenate([fam["pair"][sd][m] for sd in args.seeds
                                if m in fam["pair"].get(sd, {})])),
        }
        c = res["cross_family"][m]
        pv = c["pooled_mcnemar"]["p"]
        ph = f"{c['phi']:.3f}" if c["phi"] is not None else "undef"
        print(f"{m:16s}{ph:>18}{both:>7}{c['ga_only']:>9}{c['pair_only']:>11}{pv:>11.4g}")

    out = rd / "family_comparison.json"
    out.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    print(f"\nbudget: genetic median {res['budget']['adaptive']['median_evals']} evals/behaviour, "
          f"pair median {res['budget']['pair']['median_evals']}")
    print(f"wrote {out.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
