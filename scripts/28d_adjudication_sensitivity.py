"""Stage 28d: what the single annotator's labels do to the results.

Two questions that 28b does not answer.

1. Agreement with the labels the paper actually uses. key.csv was written before the
   grader's abstentions were resolved, so on the abstained texts it holds no grader verdict.
   This compares the annotator with the resolved label in gold_resolved.jsonl: the primary
   grader's majority where it judged, the second grader's verdict where it abstained.

2. Sensitivity. Every row of the genetic-search arm whose response text was adjudicated gets
   the annotator's label in place of the grader's, and the behavior-level analysis is run
   again: attack success per safeguard, the undefended seed-pair null, and each pair's
   excess over it. Rows that were not adjudicated keep the grader's label.

Writes results/<run>/adjudication/adjudication_sensitivity.json.

Usage: python scripts/28d_adjudication_sensitivity.py --run v2_vicuna --seeds 0 1 2
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from dcorr.analysis.estimands import (Vectors, e1_pairs, marginals, null_from_undefended,  # noqa: E402
                                      partial_given_undefended, wilson)


def read_csv(p: Path) -> list[dict]:
    with p.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def kappa(a: list[int], b: list[int]) -> float:
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def rate(k: int, n: int) -> dict:
    lo, hi = wilson(k, n) if n else (float("nan"), float("nan"))
    return {"k": k, "n": n, "rate": k / n if n else float("nan"), "ci": [lo, hi]}


def agreement(rows: list[tuple[int, int]]) -> dict:
    """rows: (annotator, grader). Precision and recall take the annotator as the reference."""
    h = [r[0] for r in rows]
    g = [r[1] for r in rows]
    tp = sum(1 for x, y in rows if x == 1 and y == 1)
    return {"n": len(rows),
            "annotator_breach": rate(sum(h), len(h)),
            "grader_breach": rate(sum(g), len(g)),
            "agreement": rate(sum(x == y for x, y in rows), len(rows)),
            "kappa": kappa(h, g),
            "grader_precision": rate(tp, sum(g)),
            "grader_recall": rate(tp, sum(h))}


def analyse(vec: dict[int, dict[str, np.ndarray]], ids: list[str]) -> dict:
    V = Vectors(behaviours=ids, vec=vec)
    null = null_from_undefended(V)
    e1 = e1_pairs(V, resamples=200)
    part = partial_given_undefended(V)
    ng = {k: v["mean_phi"] for k, v in e1.items() if "llamaguard" not in k}
    ngp = [part[k]["mean"] for k in ng]
    above = {k: v - null["mean"] for k, v in ng.items() if v > null["mean"]}
    m = marginals(V)
    return {
        "asr": {d: {"mean": v["mean_asr"], "ci": list(v["pooled_ci"])} for d, v in m.items()},
        "null": {"pairs": null["pairs"], "mean": null["mean"],
                 "span": max(null["pairs"]) - min(null["pairs"])},
        "non_guard_pairs": len(ng),
        "non_guard_mean_phi": float(np.mean(list(ng.values()))),
        "non_guard_mean_partial": float(np.mean(ngp)),
        "difficulty_share": 1 - float(np.mean(ngp)) / float(np.mean(list(ng.values()))),
        "pairs_above_null": len(above),
        "largest_excess": max((v - null["mean"] for v in ng.values())),
        "largest_excess_pair": max(ng, key=ng.get),
        "guard_pairs_range": [min(v["mean_phi"] for k, v in e1.items() if "llamaguard" in k),
                              max(v["mean_phi"] for k, v in e1.items() if "llamaguard" in k)],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="v2_vicuna")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--attack", default="adaptive")
    args = ap.parse_args()

    rd = REPO / "results" / args.run
    ad = rd / "adjudication"
    key = {r["item_id"]: r for r in read_csv(ad / "key.csv")}
    human = {}
    for r in read_csv(ad / "sheet.csv"):
        v = (r.get("annotator_1_breach") or "").strip()
        if v in ("0", "1", "0.0", "1.0"):
            human[key[r["item_id"]]["verdict_key"]] = int(float(v))
    if len(human) != len(key):
        raise SystemExit(f"{len(human)} of {len(key)} items are labeled; finish the main set first")
    stratum = {v["verdict_key"]: v["stratum"] for v in key.values()}

    gold = {s: read_jsonl(rd / f"seed{s}" / "gold_resolved.jsonl") for s in args.seeds}

    # ---- 1. agreement with the resolved labels
    resolved: dict[str, int] = {}
    for s in args.seeds:
        for g in gold[s]:
            k = g.get("verdict_key")
            if k in human:
                b = int(bool(g["breach"]))
                if resolved.setdefault(k, b) != b:
                    raise SystemExit(f"verdict_key {k} carries two resolved labels")
    missing = [k for k in human if k not in resolved]
    if missing:
        raise SystemExit(f"{len(missing)} adjudicated texts are not in gold_resolved.jsonl")
    by = {}
    for name in ("split", "unparsed", "random"):
        by[name] = agreement([(human[k], resolved[k]) for k in human if stratum[k] == name])
    by["all"] = agreement([(human[k], resolved[k]) for k in human])

    # ---- 2. sensitivity of the behavior-level analysis
    ids = sorted({g["behaviour_id"] for g in gold[args.seeds[0]]
                  if g["defense"] == "undefended" and g.get("attack") == args.attack})
    base, subst = {}, {}
    touched = to_breach = to_clear = total = 0
    for s in args.seeds:
        b: dict[str, dict[str, int]] = {}
        h: dict[str, dict[str, int]] = {}
        for g in gold[s]:
            if g.get("attack") != args.attack or g["defense"].startswith("stack"):
                continue
            lab = int(bool(g["breach"]))
            new = human.get(g.get("verdict_key"), lab)
            total += 1
            if g.get("verdict_key") in human:
                touched += 1
                to_breach += int(new == 1 and lab == 0)
                to_clear += int(new == 0 and lab == 1)
            b.setdefault(g["defense"], {})[g["behaviour_id"]] = lab
            h.setdefault(g["defense"], {})[g["behaviour_id"]] = new
        base[s] = {d: np.array([m[i] for i in ids]) for d, m in b.items() if all(i in m for i in ids)}
        subst[s] = {d: np.array([m[i] for i in ids]) for d, m in h.items() if all(i in m for i in ids)}

    res = {
        "run": args.run, "seeds": args.seeds, "attack": args.attack,
        "agreement_with_resolved_labels": by,
        "rows": {"total": total, "adjudicated": touched,
                 "changed_to_breach": to_breach, "changed_to_non_breach": to_clear},
        "grader_labels": analyse(base, ids),
        "annotator_labels": analyse(subst, ids),
    }
    out = ad / "adjudication_sensitivity.json"
    out.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")

    def show(r):
        return f"{r['rate']:.3f} [{r['ci'][0]:.3f}, {r['ci'][1]:.3f}] ({r['k']}/{r['n']})"
    print("agreement with the resolved labels")
    for name, a in by.items():
        print(f"  {name:<9} n={a['n']:<4} agreement {show(a['agreement'])}  kappa {a['kappa']:.3f}")
        print(f"            grader precision {show(a['grader_precision'])}  recall {show(a['grader_recall'])}")
        print(f"            breach rate: annotator {show(a['annotator_breach'])}  grader {show(a['grader_breach'])}")
    r = res["rows"]
    print(f"\n{args.attack} rows: {r['total']}, adjudicated {r['adjudicated']}, "
          f"to breach {r['changed_to_breach']}, to non-breach {r['changed_to_non_breach']}")
    for tag in ("grader_labels", "annotator_labels"):
        a = res[tag]
        print(f"\n{tag}")
        print("  ASR  " + "  ".join(f"{d} {v['mean']:.3f}" for d, v in sorted(a["asr"].items())))
        print(f"  null {a['null']['mean']:.3f}  pairs {[round(x, 3) for x in a['null']['pairs']]}  span {a['null']['span']:.3f}")
        print(f"  non-guard pairs above null: {a['pairs_above_null']} of {a['non_guard_pairs']}, "
              f"largest excess {a['largest_excess']:+.3f} ({a['largest_excess_pair']})")
        print(f"  non-guard mean phi {a['non_guard_mean_phi']:.3f}, partial {a['non_guard_mean_partial']:.3f}, "
              f"difficulty share {a['difficulty_share']:.1%}")
        print(f"  guard pairs {a['guard_pairs_range'][0]:.3f} to {a['guard_pairs_range'][1]:.3f}")
    print(f"\nwrote {out.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
