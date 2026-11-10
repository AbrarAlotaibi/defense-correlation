"""Stage 09: is the measured failure correlation an artefact of behaviour difficulty?

THE OBJECTION. Cross-row defenses might correlate simply because some behaviours are
intrinsically easier to jailbreak: every defense fails on the easy ones, which produces a
positive phi with no shared blind spot at all. If that explains the result, H2's rejection
is vacuous. This is the first thing a referee will ask, so we test it directly.

THE TEST. For each pair (d1, d2) we stratify behaviours by a difficulty score computed from
the OTHER defenses only - the number of the remaining defenses that were breached on that
behaviour - and then compute the Cochran-Mantel-Haenszel common odds ratio, which measures
the d1-d2 association *within* difficulty strata. Using only the other defenses keeps the
stratifier independent of the pair being tested. If the association survives stratification
(CMH OR > 1, p < 0.05), the correlation is not merely item difficulty.

We also report the raw odds ratio for comparison, and Cramer's V of each defense against the
difficulty score, which quantifies how much of each defense's failure is difficulty-driven.

Writes: results/<run>/confound_check.json and a printed summary.
"""
from __future__ import annotations

import argparse
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from _bootstrap import load_env  # noqa: E402

from dcorr.config import load_config
from dcorr.io_utils import read_jsonl, write_json


def breach_matrix(gold: list[dict], attack: str = "adaptive"):
    by: dict[str, dict[str, int]] = defaultdict(dict)
    for g in gold:
        if g.get("attack") != attack:
            continue
        by[g["defense"]][g["behaviour_id"]] = 1 if g.get("breach") else 0
    defenses = sorted(d for d in by if d not in ("undefended", "stack"))
    bids = sorted({b for d in defenses for b in by[d]})
    M = {d: np.array([by[d].get(b, 0) for b in bids], dtype=int) for d in defenses}
    return M, bids


def cmh(a: np.ndarray, b: np.ndarray, strata: np.ndarray):
    """Cochran-Mantel-Haenszel common odds ratio + chi-square test across strata."""
    num = den = 0.0
    stat_num = stat_den = 0.0
    for s in np.unique(strata):
        m = strata == s
        n = int(m.sum())
        if n < 2:
            continue
        n11 = float(((a == 1) & (b == 1) & m).sum())
        n10 = float(((a == 1) & (b == 0) & m).sum())
        n01 = float(((a == 0) & (b == 1) & m).sum())
        n00 = float(((a == 0) & (b == 0) & m).sum())
        num += n11 * n00 / n
        den += n10 * n01 / n
        r1, r0 = n11 + n10, n01 + n00
        c1, c0 = n11 + n01, n10 + n00
        stat_num += n11 - (r1 * c1 / n)
        if n > 1:
            stat_den += (r1 * r0 * c1 * c0) / (n * n * (n - 1))
    if den == 0 or stat_den == 0:
        return float("inf") if num > 0 else float("nan"), float("nan")
    or_cmh = num / den
    chi2 = (abs(stat_num) - 0.5) ** 2 / stat_den
    from scipy.stats import chi2 as chi2dist

    p = float(1 - chi2dist.cdf(chi2, 1))
    return float(or_cmh), p


def raw_or(a: np.ndarray, b: np.ndarray) -> float:
    n11 = float(((a == 1) & (b == 1)).sum()) + 0.5
    n10 = float(((a == 1) & (b == 0)).sum()) + 0.5
    n01 = float(((a == 0) & (b == 1)).sum()) + 0.5
    n00 = float(((a == 0) & (b == 0)).sum()) + 0.5
    return (n11 * n00) / (n10 * n01)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--min-marginal", type=float, default=0.05,
                    help="skip defenses whose marginal ASR is below this (degenerate)")
    args = ap.parse_args()
    load_env()
    cfg = load_config(args.config)
    gold = read_jsonl(cfg.results_dir / "gold.jsonl")
    if not gold:
        raise SystemExit("no gold.jsonl")

    M, bids = breach_matrix(gold)
    live = [d for d, v in M.items() if args.min_marginal <= v.mean() <= 1 - 1e-9]
    print(f"behaviours={len(bids)}  live defenses={live}\n")

    out = {"n_behaviours": len(bids), "live_defenses": live, "pairs": []}
    print(f"{'pair':<34} {'phi':>6} {'rawOR':>7} {'cmhOR':>7} {'p_cmh':>8}  verdict")
    for d1, d2 in itertools.combinations(live, 2):
        a, b = M[d1], M[d2]
        others = [d for d in live if d not in (d1, d2)]
        strata = np.sum([M[d] for d in others], axis=0) if others else np.zeros(len(a), int)
        phi = float(np.corrcoef(a, b)[0, 1]) if a.std() and b.std() else 0.0
        o_raw = raw_or(a, b)
        o_cmh, p = cmh(a, b, strata)
        survives = (o_cmh > 1.0) and (p == p) and (p < 0.05)
        verdict = "SURVIVES stratification" if survives else (
            "explained by difficulty" if o_cmh <= 1.0 else "underpowered within strata")
        print(f"{d1[:15]+' x '+d2[:15]:<34} {phi:6.2f} {o_raw:7.2f} {o_cmh:7.2f} {p:8.4f}  {verdict}")
        out["pairs"].append({"d1": d1, "d2": d2, "phi": phi, "raw_or": o_raw,
                             "cmh_or": o_cmh, "p_cmh": p, "survives": bool(survives)})

    n_surv = sum(1 for p in out["pairs"] if p["survives"])
    out["n_pairs"] = len(out["pairs"])
    out["n_survives"] = n_surv
    print(f"\n{n_surv}/{len(out['pairs'])} pairs retain a positive association after "
          f"stratifying on behaviour difficulty.")
    write_json(cfg.results_dir / "confound_check.json", out)
    print(f"wrote {cfg.results_dir / 'confound_check.json'}")


if __name__ == "__main__":
    main()
