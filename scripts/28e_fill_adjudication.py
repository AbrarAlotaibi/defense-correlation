"""Stage 28e: write the adjudication figures into the manuscript's fill-in block.

The manuscript defines one macro per adjudication figure. This stage sets each of them from
results/<run>/adjudication/adjudication_sensitivity.json (stage 28d) and, once the retest
is scored, adjudication_single.json (stage 28b), so no figure is copied by hand. Stage 39
recomputes the same strings and checks that the manuscript carries them.

Usage: python scripts/28e_fill_adjudication.py --run v2_vicuna
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MANUSCRIPT = Path(os.environ.get(
    "DCORR_MANUSCRIPT", REPO.parent / "defense-correlation-paper" / "paper_A_mlj" / "main.tex"))


def rate(r: dict) -> str:
    return f"{r['rate']:.2f} $[{r['ci'][0]:.2f}, {r['ci'][1]:.2f}]$ ({r['k']}/{r['n']})"


def thousands(n: int) -> str:
    return f"{n:,}".replace(",", "{,}")


def macros(run: str) -> dict[str, str]:
    ad = REPO / "results" / run / "adjudication"
    d = json.loads((ad / "adjudication_sensitivity.json").read_text(encoding="utf-8"))
    a = d["agreement_with_resolved_labels"]
    g, h, rows = d["grader_labels"], d["annotator_labels"], d["rows"]
    delta = [h["asr"][k]["mean"] - g["asr"][k]["mean"] for k in g["asr"]]
    out = {
        "AdjAgreeSplit": rate(a["split"]["agreement"]),
        "AdjAgreeAbstain": rate(a["unparsed"]["agreement"]),
        "AdjAgreeUnanimous": rate(a["random"]["agreement"]),
        "AdjKappaSplit": f"{a['split']['kappa']:.2f}",
        "AdjHumanBreach": f"{a['all']['annotator_breach']['k']}",
        "AdjGraderBreach": f"{a['all']['grader_breach']['k']}",
        "AdjPrecision": rate(a["all"]["grader_precision"]),
        "AdjRecall": rate(a["all"]["grader_recall"]),
        "AdjSensitivity": (
            f"Replacing the grader's label with the annotator's on the {rows['adjudicated']} of "
            f"{thousands(rows['total'])} genetic-search rows whose response was adjudicated changes "
            f"{rows['changed_to_breach']} labels to breach and {rows['changed_to_non_breach']} to "
            f"non-breach. Attack success rises by between {min(delta):.2f} and {max(delta):.2f} per "
            f"safeguard, $\\phi_{{\\mathrm{{null}}}}$ moves from {g['null']['mean']:.3f} to "
            f"{h['null']['mean']:.3f}, and {_count(h['pairs_above_null'])} non-guard pair"
            f"{'' if h['pairs_above_null'] == 1 else 's'} lie{'s' if h['pairs_above_null'] == 1 else ''} "
            f"above it, the largest excess being ${h['largest_excess']:+.3f}$ against "
            f"${g['largest_excess']:+.3f}$ on the grader's labels."),
    }
    single = ad / "adjudication_single.json"
    if single.exists():
        s = json.loads(single.read_text(encoding="utf-8"))
        rt = s.get("retest") or {}
        if rt.get("agreement") and rt.get("kappa") is not None:
            out["AdjRetestAgree"] = rate(rt["agreement"])
            out["AdjRetestKappa"] = f"{rt['kappa']:.2f}"
    return out


def _count(n: int) -> str:
    return "no" if n == 0 else str(n)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="v2_vicuna")
    args = ap.parse_args()
    tex = MANUSCRIPT.read_text(encoding="utf-8")
    for name, value in macros(args.run).items():
        pat = re.compile(r"(?m)^\\newcommand\{\\%s\}\{.*\}$" % name)
        if len(pat.findall(tex)) != 1:
            raise SystemExit(f"\\{name} is not defined exactly once on a single line of {MANUSCRIPT}")
        tex = pat.sub(lambda m: "\\newcommand{\\%s}{%s}" % (name, value), tex)
        print(f"\\{name} = {value}")
    MANUSCRIPT.write_text(tex, encoding="utf-8")
    left = re.findall(r"(?m)^\\newcommand\{\\(Adj\w+)\}\{\\FILL", tex)
    print(f"\nstill to fill: {', '.join(left) if left else 'nothing'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
