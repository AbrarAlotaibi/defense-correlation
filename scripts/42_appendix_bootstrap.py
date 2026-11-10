"""Stage 42: the per-seed bootstrap intervals for every pair, as a LaTeX appendix table.

The Results section quotes a median interval width to argue that no pair is distinguishable
from the null. That argument rests entirely on the intervals being wide, so a reader who
wants to check it should not have to take the median on trust. This emits all 21 pairs by
3 replicates with their bootstrap intervals, plus the column of per-seed nulls they are
compared against.

Usage: python scripts/42_appendix_bootstrap.py --run v2_vicuna --gold resolved > table.tex
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

SHORT = {
    "ppl_filter": "filter",
    "refusal_prime": "priming",
    "smoothllm": "aggregator",
    "token_anomaly": "anomaly",
    "llamaguard": "guard",
    "probe": r"probe$_{16}$",
    "probe_b": r"probe$_{8}$",
}


def label(pair: str) -> str:
    a, b = pair.split("|")
    return f"{SHORT.get(a, a)} $|$ {SHORT.get(b, b)}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="v2_vicuna")
    ap.add_argument("--analysis", default=None,
                    help="analysis json; defaults to analysis_v2_resolved.json then analysis_v2.json")
    args = ap.parse_args()

    rd = REPO / "results" / args.run
    if args.analysis:
        src = Path(args.analysis)
    else:
        src = rd / "analysis_v2_resolved.json"
        if not src.exists():
            src = rd / "analysis_v2.json"
    d = json.loads(src.read_text(encoding="utf-8"))

    e1 = d["e1"]
    seeds = sorted({s for v in e1.values() for s in v["per_seed"]}, key=int)
    nullp = d["null"]["pairs"]

    widths = [x["ci"][1] - x["ci"][0] for v in e1.values() for x in v["per_seed"].values()]

    # Guard pairs last, and within each block strongest mean first, matching tab:e1's ordering
    # so a reader can move between the two tables without re-sorting.
    keys = sorted(e1, key=lambda k: ("llamaguard" in k, -e1[k]["mean_phi"]))

    out = [
        r"\begin{table}[pos=!htbp]",
        r"\centering",
        r"\caption{Behaviour-level $\phi$ per replicate with bootstrap 95\% intervals, "
        r"2000 resamples over behaviours, for every pair. The Results section quotes the "
        f"median width of these {len(widths)} intervals, {statistics.median(widths):.3f}; "
        r"this table is the distribution behind it. The final row is the undefended "
        r"seed-pair correlation, the null each column is read against.}",
        rf"\label{{tab:bootstrap_{args.run}}}",
        r"\footnotesize",
        r"\begin{tabular*}{\tblwidth}{@{\extracolsep{\fill}}l" + "c" * len(seeds) + r"@{}}",
        r"\toprule",
        "pair & " + " & ".join(f"replicate {i+1}" for i in range(len(seeds))) + r" \\",
        r"\midrule",
    ]
    prev_guard = False
    for k in keys:
        is_guard = "llamaguard" in k
        if is_guard and not prev_guard:
            out.append(r"\addlinespace")
        prev_guard = is_guard
        cells = []
        for s in seeds:
            v = e1[k]["per_seed"].get(s)
            cells.append("---" if v is None else
                         f"{v['phi']:.3f} [{v['ci'][0]:.2f}, {v['ci'][1]:.2f}]")
        out.append(f"{label(k)} & " + " & ".join(cells) + r" \\")

    out += [
        r"\midrule",
        r"\emph{Undefended} ($\phi_{\mathrm{null}}$) & "
        + " & ".join(f"\\emph{{{v:.3f}}}" for v in nullp) + r" \\",
        r"\bottomrule",
        r"\end{tabular*}",
        r"\end{table}",
    ]
    print("\n".join(out))

    print(f"\n% source: {src.relative_to(REPO)}", end="")
    print(f"\n% {len(keys)} pairs x {len(seeds)} replicates; median width "
          f"{statistics.median(widths):.4f}, min {min(widths):.4f}, max {max(widths):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
