"""Stage 46: is the transfer matrix explained by which member is scored, or by which search?

The cross-evaluation gives breach rate for every (search that produced the prompts, member
scored on them) pair. The ensemble story predicts a diagonal: a prompt set optimised against
member X should work better on X than on members it never saw. It does not. The diagonal mean
is 0.578 against an off-diagonal mean of 0.635, so prompts transfer to members they were not
found against slightly *better* than to the one they were.

This quantifies the alternative: the matrix is close to rank one in the column index. Breach
rate is a function of which member is being scored, and barely a function of which search
produced the prompt. That is the transfer-matrix form of the result in stage 44 -- most members
do not alter the pipeline, so there is nothing for a search to specialise against.

Writes results/<run>/transfer_structure.json.

Usage: python scripts/46_transfer_structure.py --run v2_vicuna --seed 0
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

ORDER = ["ppl_filter", "probe_b", "token_anomaly", "probe", "refusal_prime",
         "smoothllm", "llamaguard", "undefended"]
SHORT = {"ppl_filter": "filter", "probe_b": r"probe$_{8}$", "token_anomaly": "anomaly",
         "probe": r"probe$_{16}$", "refusal_prime": "priming", "smoothllm": "aggregator",
         "llamaguard": "guard", "undefended": r"\emph{undefended}"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="v2_vicuna")
    ap.add_argument("--seed", default="0")
    args = ap.parse_args()

    rd = REPO / "results" / args.run
    d = json.loads((rd / "analysis_v2.json").read_text(encoding="utf-8"))
    T = d["transfer"].get(args.seed)
    if not T:
        raise SystemExit(f"no transfer matrix for seed {args.seed} in analysis_v2.json")

    rows = [r for r in ORDER if r in T]
    cols = [c for c in ORDER if any(c in T[r] for r in rows)]
    M = {(r, c): T[r][c] for r in rows for c in cols if c in T[r]}

    diag = [M[(x, x)] for x in rows if (x, x) in M]
    off = [v for (r, c), v in M.items() if r != c]
    # The guard is scored near zero whatever the search, so it dominates any variance
    # decomposition. Report with and without it rather than letting one member decide.
    ng = [x for x in rows if x != "llamaguard"]
    ngM = {k: v for k, v in M.items() if k[0] != "llamaguard" and k[1] != "llamaguard"}
    diag_ng = [M[(x, x)] for x in ng if (x, x) in M]
    off_ng = [v for (r, c), v in ngM.items() if r != c]

    def explained(sub: dict, key) -> float:
        """share of variance in `sub` removed by conditioning on key(cell)."""
        vals = list(sub.values())
        gm = statistics.mean(vals)
        tot = sum((v - gm) ** 2 for v in vals)
        groups: dict = {}
        for k, v in sub.items():
            groups.setdefault(key(k), []).append(v)
        resid = sum((v - statistics.mean(g)) ** 2 for g in groups.values() for v in g)
        return (tot - resid) / tot if tot else float("nan")

    res = {
        "run": args.run, "seed": args.seed,
        "diagonal_mean": statistics.mean(diag),
        "offdiagonal_mean": statistics.mean(off),
        "diagonal_mean_no_guard": statistics.mean(diag_ng),
        "offdiagonal_mean_no_guard": statistics.mean(off_ng),
        "variance_explained_by_column_all": explained(M, lambda k: k[1]),
        "variance_explained_by_row_all": explained(M, lambda k: k[0]),
        "variance_explained_by_column_no_guard": explained(ngM, lambda k: k[1]),
        "variance_explained_by_row_no_guard": explained(ngM, lambda k: k[0]),
        "per_member_own_vs_others": {},
    }
    for x in rows:
        if (x, x) not in M:
            continue
        others = [M[(r, x)] for r in rows if r != x and (r, x) in M]
        res["per_member_own_vs_others"][x] = {
            "own_search": M[(x, x)],
            "other_searches_mean": statistics.mean(others) if others else None,
            "advantage": M[(x, x)] - statistics.mean(others) if others else None,
        }

    out = rd / "transfer_structure.json"
    out.write_text(json.dumps(res, indent=1), encoding="utf-8")

    print(f"diagonal {res['diagonal_mean']:.3f}   off-diagonal {res['offdiagonal_mean']:.3f}")
    print(f"without the guard: diagonal {res['diagonal_mean_no_guard']:.3f}   "
          f"off-diagonal {res['offdiagonal_mean_no_guard']:.3f}")
    print(f"\nvariance in the matrix explained by")
    print(f"  which member is scored (column): {res['variance_explained_by_column_all']:.3f}"
          f"   without guard {res['variance_explained_by_column_no_guard']:.3f}")
    print(f"  which search produced it (row):  {res['variance_explained_by_row_all']:.3f}"
          f"   without guard {res['variance_explained_by_row_no_guard']:.3f}")
    print(f"\n{'member':16s}{'own search':>12}{'other searches':>16}{'advantage':>11}")
    for x, v in res["per_member_own_vs_others"].items():
        if v["advantage"] is None:
            continue
        print(f"{x:16s}{v['own_search']:>12.2f}{v['other_searches_mean']:>16.2f}"
              f"{v['advantage']:>+11.2f}")

    print(f"\nLaTeX body:")
    print(r"\begin{tabular*}{\tblwidth}{@{\extracolsep{\fill}}l" + "c" * len(cols) + r"@{}}")
    print(r"\toprule")
    print("search against & " + " & ".join(SHORT.get(c, c) for c in cols) + r" \\")
    print(r"\midrule")
    for r in rows:
        cells = []
        for c in cols:
            v = M.get((r, c))
            cells.append("---" if v is None
                         else (f"\\textbf{{{v:.2f}}}" if r == c else f"{v:.2f}"))
        print(f"{SHORT.get(r, r)} & " + " & ".join(cells) + r" \\")
    print(r"\bottomrule")
    print(r"\end{tabular*}")
    print(f"\nwrote {out.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
