"""Stage 44: E2 on shared inputs, with the pass-through diagnostic that makes it readable.

The cross-evaluation gives prompt-level phi up to 1.000, and taken at face value that reads as
near-deterministic co-failure between mechanisms. It is not. A member that blocks nothing at
its calibrated operating point and does not modify generation returns the target's own
response, so two such members are the same function on a shared input and their breach vectors
coincide by construction.

The diagnostic is each member's phi against the *undefended* model on the same inputs. A
member whose phi with undefended is 1.000 did nothing; one that intervenes cannot reach it.
Scoring undefended in the cross-evaluation is what makes this visible, and it reorders the
whole table: E2 here is explained almost entirely by how much of the computation each member
alters, not by what its detector looks at.

Writes results/<run>/e2_passthrough.json and prints the LaTeX body for the E2 table.

Usage: python scripts/44_e2_passthrough.py --run v2_vicuna --seed 0
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

NAME = {
    "ppl_filter": "filter",
    "refusal_prime": "priming",
    "smoothllm": "aggregator",
    "token_anomaly": "anomaly",
    "llamaguard": "guard",
    "probe": r"probe$_{16}$",
    "probe_b": r"probe$_{8}$",
    "undefended": r"\emph{undefended}",
}
ORDER = ["ppl_filter", "probe_b", "token_anomaly", "probe", "refusal_prime",
         "smoothllm", "llamaguard"]


def phi(x, y):
    n11 = sum(1 for a, b in zip(x, y) if a and b)
    n10 = sum(1 for a, b in zip(x, y) if a and not b)
    n01 = sum(1 for a, b in zip(x, y) if not a and b)
    n00 = sum(1 for a, b in zip(x, y) if not a and not b)
    den = ((n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00)) ** 0.5
    return None if den == 0 else (n11 * n00 - n10 * n01) / den


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="v2_vicuna")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gold", default=None,
                    help="gold filename in cross/; defaults to gold_resolved.jsonl then gold.jsonl")
    args = ap.parse_args()

    cross = REPO / "results" / args.run / f"seed{args.seed}" / "cross"
    gp = cross / (args.gold or "gold_resolved.jsonl")
    if not gp.exists():
        gp = cross / "gold.jsonl"
    print(f"[44] labels from cross/{gp.name}")

    gold = {}
    for line in gp.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            gold[(r["defense"], r["behaviour_id"], r["attack"])] = r
    rows = {}
    for p in sorted(cross.glob("stage04_*.jsonl")):
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                rows[(r["defense"], r["behaviour_id"], r["attack"])] = r

    defs = [d for d in ORDER if any(k[0] == d for k in rows)]
    keys = sorted({(b, a) for (d, b, a) in rows if d == "undefended"})
    keys = [k for k in keys if all((d, *k) in gold for d in defs + ["undefended"])]
    print(f"[44] {len(keys)} shared inputs, {len(defs)} members")

    vec = {d: [bool(gold[(d, *k)]["breach"]) for k in keys] for d in defs + ["undefended"]}
    resp = {d: [(rows[(d, *k)].get("response") or "") for k in keys] for d in defs + ["undefended"]}
    blocked = {d: sum(1 for k in keys if rows[(d, *k)].get("blocked")) for d in defs}

    res = {"run": args.run, "seed": args.seed, "labels": gp.name, "n": len(keys),
           "members": {}, "pairs": {}}
    for d in defs:
        ident = sum(1 for x, y in zip(resp[d], resp["undefended"]) if x == y)
        res["members"][d] = {
            "blocked": blocked[d],
            "identical_to_undefended": ident,
            "phi_with_undefended": phi(vec[d], vec["undefended"]),
            "breach_rate": sum(vec[d]) / len(keys),
        }
    for i, a in enumerate(defs):
        for b in defs[i + 1:]:
            res["pairs"][f"{a}|{b}"] = {
                "phi": phi(vec[a], vec[b]),
                "identical_responses": sum(1 for x, y in zip(resp[a], resp[b]) if x == y),
            }

    out = REPO / "results" / args.run / "e2_passthrough.json"
    out.write_text(json.dumps(res, indent=1), encoding="utf-8")

    n = len(keys)
    print(f"\n{'member':14s}{'blocks':>8}{'resp = undef':>14}{'phi w/ undef':>14}{'breach':>9}")
    for d in defs:
        m = res["members"][d]
        print(f"{d:14s}{m['blocked']:>8}{m['identical_to_undefended']:>9}/{n:<4}"
              f"{m['phi_with_undefended']:>14.3f}{m['breach_rate']:>9.3f}")

    print(f"\n{'pair':26s}{'phi':>8}{'identical responses':>21}")
    for k, v in sorted(res["pairs"].items(), key=lambda kv: -(kv[1]["phi"] or -9)):
        print(f"{k:26s}{v['phi']:>8.3f}{v['identical_responses']:>16}/{n}")

    passthrough = [d for d in defs if res["members"][d]["phi_with_undefended"] is not None
                   and res["members"][d]["phi_with_undefended"] > 0.9]
    print(f"\nmembers indistinguishable from no defense at all (phi with undefended > 0.9): "
          f"{passthrough}")

    # Does the pair tell us anything its members do not? Compare each pair's phi against the
    # smaller of its two members' phi with undefended. Where that bound is tight, the pairwise
    # statistic is carrying no information beyond how far each member departs from inaction.
    import statistics
    one, both = [], []
    for k, v in res["pairs"].items():
        a, b = k.split("|")
        pa = res["members"][a]["phi_with_undefended"]
        pb = res["members"][b]["phi_with_undefended"]
        if v["phi"] is None or pa is None or pb is None:
            continue
        dev = v["phi"] - min(pa, pb)
        v["min_phi_with_undefended"] = min(pa, pb)
        v["deviation_from_min"] = dev
        (both if (pa < 0.9 and pb < 0.9) else one).append((abs(dev), k))
    res["min_rule"] = {
        "at_most_one_intervenes": {
            "n": len(one),
            "mean_abs_deviation": statistics.mean(x for x, _ in one) if one else None,
            "max_abs_deviation": max(one)[0] if one else None,
            "worst_pair": max(one)[1] if one else None},
        "both_intervene": {
            "n": len(both),
            "mean_abs_deviation": statistics.mean(x for x, _ in both) if both else None,
            "max_abs_deviation": max(both)[0] if both else None,
            "pairs": sorted(k for _, k in both)},
    }
    r = res["min_rule"]
    print("\nphi against min(phi with undefended) over the two members:")
    for tag in ("at_most_one_intervenes", "both_intervene"):
        x = r[tag]
        if x["n"]:
            print(f"  {tag.replace('_', ' '):26s} n={x['n']:<3} mean |dev| "
                  f"{x['mean_abs_deviation']:.3f}  max {x['max_abs_deviation']:.3f}")
    out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"wrote {out.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
