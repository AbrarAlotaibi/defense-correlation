"""Stage 47: E3, the assembled stack against the estimate its own layers imply.

The paper's practical claim is that a stack has to be measured as assembled, because the
per-layer composition estimate practitioners compute does not predict it. Stage 26 already
produces the pieces, but across three places: `e3` for the configured stack, `e3_subsets` for
the jointly attacked pairs, and a separate run directory for the six-layer stack without the
guard. This assembles them into the one table the section needs, pools over whatever seeds are
graded, and says which comparison has resolution and which does not.

TWO ESTIMATES, AND THE DIFFERENCE BETWEEN THEM IS THE POINT:

  intersection   the breach rate over behaviours that EVERY member failed on, measured one
                 member at a time. This is the per-layer composition estimate: it is what a
                 deployer can compute without attacking the assembled pipeline.
  measured       the breach rate of the assembled stack, attacked end to end as deployed.

A full stack containing the guard model floors both at ~0.02, where nothing can be
distinguished from anything, which is why the pair subsets exist and why they carry the claim.

Usage: python scripts/47_e3_table.py
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

SHORT = {"ppl_filter": "filter", "probe_b": r"probe$_{8}$", "token_anomaly": "anomaly",
         "probe": r"probe$_{16}$", "refusal_prime": "priming", "smoothllm": "aggregator",
         "llamaguard": "guard"}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - m) / d, (c + m) / d)


def label(members: list[str]) -> str:
    return " + ".join(SHORT.get(m, m) for m in members)


def _breach_vectors(gold: Path, ids: list[str]) -> dict[str, list[int]]:
    """{defense: breach vector over `ids`} for the fluent adversary, from one gold file."""
    import json as _json

    by: dict[str, dict[str, int]] = {}
    for line in gold.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = _json.loads(line)
        if r.get("attack") != "adaptive":
            continue
        by.setdefault(r["defense"], {})[r["behaviour_id"]] = int(bool(r["breach"]))
    return {d: [v.get(b, 0) for b in ids] for d, v in by.items()}


def _pooled(repo: Path, primary: str, stack_defense: str, members: list[str],
            seeds: list, stack_run: str | None = None) -> dict | None:
    """Exact McNemar over every paired observation, for a stack measured more than once.

    Per-replicate p-values must not be summarised by their minimum: the smallest of k
    p-values compared against 0.05 inflates the false-positive rate about k-fold, and with
    three replicates that turned three of the four pairs into "significant" results. The
    replicates are independent searches over the same behaviours, so one test over all of
    them is both available and the only honest aggregation.

    `stack_run` lets the stack's own rows come from a different run directory than its
    members', which is how the six-layer configuration is measured.
    """
    import sys

    sys.path.insert(0, str(repo))
    import numpy as np

    from dcorr.analysis.estimands import exact_mcnemar

    ids = [json.loads(l)["behaviour_id"]
           for l in (repo / "data/eval_harmful.jsonl").read_text(encoding="utf-8").splitlines()
           if l.strip()]
    st_all, in_all = [], []
    for sd in seeds:
        pg = _gold_path(repo, primary, sd)
        sg = _gold_path(repo, stack_run or primary, sd)
        if pg is None or sg is None:
            continue
        prim = _breach_vectors(pg, ids)
        if not all(m in prim for m in members):
            return None
        stacks = _breach_vectors(sg, ids)
        st = stacks.get(stack_defense)
        if st is None:
            continue
        st_all.append(np.asarray(st))
        in_all.append(np.prod([np.asarray(prim[m]) for m in members], axis=0))
    if len(st_all) < 2:
        return None
    return exact_mcnemar(np.concatenate(st_all), np.concatenate(in_all))


def _gold_path(repo: Path, run: str, seed) -> Path | None:
    for name in ("gold_resolved.jsonl", "gold.jsonl"):
        p = repo / "results" / run / f"seed{seed}" / name
        if p.exists():
            return p
    return None


def _join_six(repo: Path, primary: str, nolg: str, members: list[str],
              seeds: list) -> dict:
    """E3 for a stack measured in one run against members measured in another.

    Same shape stage 26's e3_stack returns, so `collect` needs no special case. Uses the same
    estimands so the row is computed identically to the others: elementwise product of the
    member vectors for the intersection, exact McNemar for both comparisons.
    """
    import sys

    sys.path.insert(0, str(repo))
    import numpy as np

    from dcorr.analysis.estimands import exact_mcnemar, wilson as _wilson

    ids = [json.loads(l)["behaviour_id"]
           for l in (repo / "data/eval_harmful.jsonl").read_text(encoding="utf-8").splitlines()
           if l.strip()]

    out: dict = {}
    pooled_st, pooled_in = [], []
    for s in seeds:
        pg = repo / "results" / primary / f"seed{s}" / "gold_resolved.jsonl"
        ng = repo / "results" / nolg / f"seed{s}" / "gold_resolved.jsonl"
        if not pg.exists():
            pg = repo / "results" / primary / f"seed{s}" / "gold.jsonl"
        if not ng.exists():
            ng = repo / "results" / nolg / f"seed{s}" / "gold.jsonl"
        if not (pg.exists() and ng.exists()):
            continue
        prim = _breach_vectors(pg, ids)
        if not all(m in prim for m in members):
            missing = [m for m in members if m not in prim]
            print(f"[47] seed{s}: primary run lacks {missing}; six-layer row skipped")
            continue
        stacks = _breach_vectors(ng, ids)
        st = stacks.get("stack")
        if st is None:
            continue
        st = np.asarray(st)
        vecs = {m: np.asarray(prim[m]) for m in members}
        inter = np.prod([vecs[m] for m in members], axis=0)
        best = min(members, key=lambda m: vecs[m].mean())
        out[str(s)] = {
            "stack_asr": float(st.mean()), "stack_ci": _wilson(int(st.sum()), len(st)),
            "intersection_asr": float(inter.mean()),
            "strongest_layer": best, "strongest_asr": float(vecs[best].mean()),
            "stack_vs_strongest": exact_mcnemar(st, vecs[best]),
            "stack_vs_intersection": exact_mcnemar(st, inter),
        }
        pooled_st.append(st)
        pooled_in.append(inter)
    if len(pooled_st) > 1:
        out["__pooled__"] = exact_mcnemar(np.concatenate(pooled_st),
                                          np.concatenate(pooled_in))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="v2_vicuna")
    ap.add_argument("--nolg-run", default="v2_vicuna_nolg")
    args = ap.parse_args()

    rd = REPO / "results" / args.run
    src = rd / "analysis_v2_resolved.json"
    if not src.exists():
        src = rd / "analysis_v2.json"
    d = json.loads(src.read_text(encoding="utf-8"))
    print(f"[47] {args.run} from {src.name}  seeds {d.get('seeds')}")

    rows = []

    def collect(name: str, per_seed: dict, members: list[str], note: str = "",
                pooled: dict | None = None):
        if not per_seed:
            return
        seeds = sorted(per_seed, key=int)
        meas = [per_seed[s]["stack_asr"] for s in seeds]
        inter = [per_seed[s]["intersection_asr"] for s in seeds]
        strong = [per_seed[s]["strongest_asr"] for s in seeds]
        ps = [per_seed[s]["stack_vs_intersection"]["p"] for s in seeds]
        n = 100 * len(seeds)
        k = int(round(sum(meas) / len(meas) * 100)) * len(seeds)
        rows.append({
            "name": name, "members": members, "note": note, "seeds": seeds,
            "measured": sum(meas) / len(meas), "measured_per_seed": meas,
            "ci": wilson(k, n),
            "intersection": sum(inter) / len(inter), "intersection_per_seed": inter,
            "strongest": sum(strong) / len(strong),
            "strongest_layer": per_seed[seeds[0]]["strongest_layer"],
            "p_vs_intersection": ps,
            # One test over every paired observation, for a stack measured more than once.
            # Three marginal p values side by side invite the reader to pool them informally,
            # and the replicates are independent searches over the same behaviours, so the
            # pooled test is both available and the one to report.
            "pooled_vs_intersection": pooled,
            "gap": sum(meas) / len(meas) - sum(inter) / len(inter),
        })

    e3 = d.get("e3") or {}
    if e3:
        order = [m for m in (d.get("stack_order") or []) ] or [
            "ppl_filter", "token_anomaly", "probe", "probe_b", "llamaguard",
            "refusal_prime", "smoothllm"]
        collect("Seven members, as configured", e3, order,
                "the guard floors this comparison",
                pooled=_pooled(REPO, args.run, "stack", order, d.get("seeds") or []))

    for name, per in sorted((d.get("e3_subsets") or {}).items()):
        members = name[len("stack_"):].split("+")
        collect(label(members), per, members,
                pooled=_pooled(REPO, args.run, name, members, sorted(per, key=int)))

    # The six-layer stack needs a join across two runs, and the join is legitimate for a
    # specific reason: v2_vicuna_nolg exists only to attack the assembled six-layer stack.
    # It never ran stage 04, so it has no member arms of its own, and it does not need them --
    # it inherits v2_vicuna.yaml's thresholds, seeds, behaviours and target unchanged, so the
    # primary run's six non-guard members ARE this stack's members, measured under the same
    # conditions. The intersection therefore comes from v2_vicuna and the measured residual
    # from v2_vicuna_nolg. Stage 26 cannot do this because it reads one run at a time.
    SIX = ["ppl_filter", "token_anomaly", "probe", "probe_b", "refusal_prime", "smoothllm"]
    nd = REPO / "results" / args.nolg_run
    if (nd / "seed0").exists():
        six = _join_six(REPO, args.run, args.nolg_run, SIX, d.get("seeds") or [0, 1, 2])
        pooled = six.pop("__pooled__", None) if six else None
        if six:
            collect("Six members, no guard", six, SIX,
                    "measured in the no-guard run, intersection from the primary one",
                    pooled=pooled)
        else:
            print(f"[47] {args.nolg_run} has no graded stack rows yet; six-layer row omitted")
    else:
        print(f"[47] no {args.nolg_run} results; six-layer row omitted")

    if not rows:
        raise SystemExit("[47] no E3 anywhere. gold must contain rows whose defense is a "
                         "stack; if stage 35 has not folded them into gold_resolved, "
                         "stage 26 warns about exactly this.")

    print()
    print(f"{'stack':32s}{'seeds':>7}{'measured':>10}{'inter':>8}{'gap':>8}"
          f"{'strongest':>11}{'McNemar p':>22}")
    for r in rows:
        ps = ", ".join(f"{p:.3g}" for p in r["p_vs_intersection"])
        print(f"{r['name']:32s}{len(r['seeds']):>7}{r['measured']:>10.3f}"
              f"{r['intersection']:>8.3f}{r['gap']:>+8.3f}{r['strongest']:>11.3f}"
              f"{ps:>22}")

    out = rd / "e3_table.json"
    out.write_text(json.dumps(rows, indent=1), encoding="utf-8")

    print()
    print("LaTeX body:")
    print(r"\begin{tabular*}{\tblwidth}{@{\extracolsep{\fill}}lccccc@{}}")
    print(r"\toprule")
    print(r"stack & reps & measured & 95\% CI & intersection & strongest layer \\")
    print(r"\midrule")
    for r in rows:
        lo, hi = r["ci"]
        print(f"{r['name']} & {len(r['seeds'])} & {r['measured']:.2f} & "
              f"[{lo:.2f}, {hi:.2f}] & {r['intersection']:.2f} & {r['strongest']:.2f} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular*}")

    print()
    above = [r for r in rows if r["gap"] > 0]
    print(f"{len(above)} of {len(rows)} stacks breach MORE than their own layers' "
          f"intersection predicts.")
    # Pooled where a row has replicates, the single test where it has one. Never
    # min(per-seed p): see _pooled.
    def _headline_p(r):
        if r.get("pooled_vs_intersection"):
            return r["pooled_vs_intersection"]["p"], "pooled"
        if len(r["seeds"]) == 1:
            return r["p_vs_intersection"][0], "1 rep"
        return None, "no pooled test"

    print("  McNemar against the intersection:")
    for r in rows:
        p, how = _headline_p(r)
        if p is None:
            print(f"    {r['name']:30s} {how}")
        else:
            print(f"    {r['name']:30s} p = {p:<10.4g} ({how})"
                  + ("  SIGNIFICANT" if p < 0.05 else ""))
    print(f"wrote {out.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
