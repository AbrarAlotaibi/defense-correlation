"""Task D analysis: does the result survive out-of-sample threshold calibration?

Compares the primary run (thresholds set in-sample on the 100 evaluation benign prompts)
against the external-threshold run (thresholds set on 805 AlpacaEval instructions, disjoint
from both the evaluation set and the probe's training pool).

The point of the comparison is narrow and important: probe16 appears in 5 of the 15 measurable
pairs and in the only pair that survives difficulty stratification, and its 1% operating point
is the one the external corpus moves furthest. If probe16 x probe8 still survives BH under
external thresholds, the mechanism-specific reading of H1 is supported on a threshold that
transfers. If it does not, that reading has to be retired.

Both threshold sets are reported side by side; neither replaces the other.

Emits, per the analysis brief:
  D3_external_thresholds.csv  defense, threshold_insample, threshold_external,
                              benign_block_rate_external, residual_ASR_external
  D3_table10_external.csv     pair, p1, p2, Delta, phi, phi_lo, phi_hi, q
  D3_cmh_external.csv         pair, crude_OR, CMH_OR, p, q

Usage:
  python scripts/22_task_d_external.py --primary results/hpc_vicuna_autodan \
      --external results/hpc_vicuna_autodan_extthr \
      --config configs/hpc_vicuna_autodan.yaml --out fusion/taskD
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# scripts/ for fusion_analysis, repo root for dcorr
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fusion_analysis import cmh_all_pairs  # noqa: E402

SEED = 20260727
N_BOOT = 10000
DEFENSES = ["ppl_filter", "token_anomaly", "llamaguard", "refusal_prime",
            "smoothllm", "probe", "probe_b"]
PRETTY = {"ppl_filter": "perplexity", "token_anomaly": "token-anomaly",
          "llamaguard": "llama-guard", "refusal_prime": "refusal-prime",
          "smoothllm": "smoothllm", "probe": "probe16", "probe_b": "probe8"}


def read_jsonl(p: Path):
    with open(p, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def phi(x, y):
    if x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def haldane_or(b1, b2, c=0.5):
    n11 = float(np.sum((b1 == 1) & (b2 == 1))) + c
    n10 = float(np.sum((b1 == 1) & (b2 == 0))) + c
    n01 = float(np.sum((b1 == 0) & (b2 == 1))) + c
    n00 = float(np.sum((b1 == 0) & (b2 == 0))) + c
    return (n11 * n00) / (n10 * n01)


def bh(p):
    p = np.asarray(p, float)
    q = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    v = p[ok]
    o = np.argsort(v)
    m = len(v)
    adj = np.minimum.accumulate((v[o] * m / np.arange(1, m + 1))[::-1])[::-1]
    out = np.empty(m)
    out[o] = np.clip(adj, 0, 1)
    q[ok] = out
    return q


def fmt_p(x):
    return "nan" if x != x else (f"{x:.3e}" if x < 0.001 else f"{x:.4f}")


def breach_matrix(run: Path, behaviours, attack="adaptive") -> pd.DataFrame:
    idx = {b: i for i, b in enumerate(behaviours)}
    cols = {d: np.zeros(len(behaviours), dtype=int) for d in DEFENSES}
    for r in read_jsonl(run / "gold.jsonl"):
        if r.get("attack") != attack or r["defense"] not in cols:
            continue
        if r["behaviour_id"] in idx:
            cols[r["defense"]][idx[r["behaviour_id"]]] = int(bool(r["breach"]))
    return pd.DataFrame(cols)


def frr(run: Path, cfg_path: Path) -> dict:
    """Realised benign refusal per defense under whatever thresholds that run used."""
    from dcorr.config import load_config
    from dcorr.judge import RefusalScorer
    cfg = load_config(str(cfg_path))
    sc = RefusalScorer(
        max_response_length=int(cfg.get_path("refusal.max_response_length", 600)),
        model_name=cfg.get_path("target.model_id"))
    out = {}
    for d in DEFENSES:
        p = run / f"stage04_{d}_benign.jsonl"
        if not p.is_file():
            continue
        rows = read_jsonl(p)
        out[d] = float(np.mean([bool(sc.is_refusal(r.get("response", ""),
                                                   r.get("blocked", False))) for r in rows]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--primary", required=True, type=Path)
    ap.add_argument("--external", required=True, type=Path)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--boot", type=int, default=N_BOOT)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    g = a.external / "gold.jsonl"
    if not g.is_file():
        raise SystemExit(f"{g} missing - run stage 06 on the external config first")

    behaviours = [json.loads(l)["behaviour_id"]
                  for l in open("data/eval_harmful.jsonl", encoding="utf-8") if l.strip()]
    B_ext = breach_matrix(a.external, behaviours)
    B_in = breach_matrix(a.primary, behaviours)

    cal_in = json.load(open(a.primary / "calibration.json", encoding="utf-8"))["thresholds"]
    cal_ex = json.load(open(a.external / "calibration.json", encoding="utf-8"))["thresholds"]
    frr_ex = frr(a.external, a.config)
    frr_in = frr(a.primary, a.config)

    # ---- D3 thresholds -----------------------------------------------------------------
    with open(a.out / "D3_external_thresholds.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["defense", "threshold_insample", "threshold_external",
                    "benign_block_rate_insample", "benign_block_rate_external",
                    "residual_ASR_insample", "residual_ASR_external"])
        for d in DEFENSES:
            w.writerow([PRETTY[d],
                        f"{cal_in[d]:.6f}" if d in cal_in else "n/a",
                        f"{cal_ex[d]:.6f}" if d in cal_ex else "n/a",
                        f"{frr_in.get(d, float('nan')):.3f}",
                        f"{frr_ex.get(d, float('nan')):.3f}",
                        f"{B_in[d].mean():.3f}", f"{B_ext[d].mean():.3f}"])

    # ---- D3 table 10 under external thresholds -----------------------------------------
    live = [d for d in DEFENSES if B_ext[d].std() > 0 and d != "llamaguard"]
    pairs = list(itertools.combinations(live, 2))
    recs = []
    for d1, d2 in pairs:
        x, y = B_ext[d1].to_numpy(), B_ext[d2].to_numpy()
        p1, p2 = float(x.mean()), float(y.mean())
        obs = phi(x, y)
        bs = np.array([phi(x[s], y[s]) for s in
                       (rng.integers(0, len(x), len(x)) for _ in range(a.boot))])
        bs = bs[~np.isnan(bs)]
        cnt = sum(1 for _ in range(2000)
                  if abs(phi(rng.permutation(x), y)) >= abs(obs) - 1e-12)
        recs.append({"pair": f"{PRETTY[d1]} x {PRETTY[d2]}", "p1": p1, "p2": p2,
                     "joint": float((x & y).mean()), "Delta": float((x & y).mean()) - p1 * p2,
                     "phi": obs, "lo": float(np.quantile(bs, .025)),
                     "hi": float(np.quantile(bs, .975)), "p": (cnt + 1) / 2001})
    qs = bh([r["p"] for r in recs])
    with open(a.out / "D3_table10_external.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["pair", "p1", "p2", "joint", "Delta", "phi", "phi_lo", "phi_hi", "q"])
        for r, q in zip(recs, qs):
            w.writerow([r["pair"], f"{r['p1']:.3f}", f"{r['p2']:.3f}", f"{r['joint']:.3f}",
                        f"{r['Delta']:.3f}", f"{r['phi']:.3f}", f"{r['lo']:.3f}",
                        f"{r['hi']:.3f}", fmt_p(q)])

    # ---- D3 CMH under external thresholds ----------------------------------------------
    cmh = cmh_all_pairs(B_ext[live])
    names = list(B_ext[live].columns)
    arr = B_ext[live].to_numpy()
    cmh["crude_OR_haldane"] = [haldane_or(arr[:, names.index(r.d1)],
                                          arr[:, names.index(r.d2)])
                               for r in cmh.itertuples()]
    with open(a.out / "D3_cmh_external.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["pair", "crude_OR", "CMH_OR", "p", "q"])
        for r in cmh.sort_values("p").itertuples():
            w.writerow([f"{PRETTY[r.d1]} x {PRETTY[r.d2]}",
                        f"{r.crude_OR_haldane:.3f}", f"{r.CMH_OR:.3f}",
                        fmt_p(r.p), fmt_p(getattr(r, "q", float("nan")))])

    # ---- the question this run exists to answer ----------------------------------------
    # A CMH odds ratio of 0 is NOT evidence of no association: it means every stratum has an
    # empty concordant cell (a*d = 0 throughout), so the numerator vanishes and the statistic
    # is undefined rather than small. Detect that explicitly, because reporting it as a
    # refutation would be wrong in the same way the Llama Guard degeneracy would be.
    def informative_strata(d1, d2):
        others = [d for d in live if d not in (d1, d2)]
        diff = B_ext[others].to_numpy().sum(axis=1)
        x, y = B_ext[d1].to_numpy(), B_ext[d2].to_numpy()
        n = 0
        for s in np.unique(diff):
            m = diff == s
            if m.sum() < 2:
                continue
            if ((x[m] == 1) & (y[m] == 1)).sum() * ((x[m] == 0) & (y[m] == 0)).sum() > 0:
                n += 1
        return n

    degenerate = {f"{PRETTY[d1]} x {PRETTY[d2]}": informative_strata(d1, d2)
                  for d1, d2 in pairs if informative_strata(d1, d2) == 0}

    key = cmh[((cmh.d1 == "probe") & (cmh.d2 == "probe_b")) |
              ((cmh.d1 == "probe_b") & (cmh.d2 == "probe"))]
    probe_strata = informative_strata("probe", "probe_b")
    if probe_strata == 0:
        survives = "UNDEFINED - every difficulty stratum is saturated, so the CMH numerator " \
                   "is identically zero. This is not a refutation; the test cannot be " \
                   "computed for this pair under these thresholds."
    else:
        survives = bool(len(key)) and float(key.iloc[0].q) < 0.05
    n_surv = int((cmh["q"] < 0.05).sum()) if "q" in cmh else 0
    verdict = {
        "probe16_x_probe8_survives_BH_under_external_thresholds": survives,
        "probe16_x_probe8_informative_strata": probe_strata,
        "pairs_with_undefined_CMH": degenerate,
        "probe16_x_probe8_CMH_OR": float(key.iloc[0].CMH_OR) if len(key) else None,
        "probe16_x_probe8_q": float(key.iloc[0].q) if len(key) else None,
        "n_pairs_surviving_BH": n_surv,
        "n_pairs_measurable": len(pairs),
        "degenerate_defenses_under_external_thresholds":
            [PRETTY[d] for d in DEFENSES if B_ext[d].std() == 0],
        "marginal_asr_insample": {PRETTY[d]: round(float(B_in[d].mean()), 3) for d in DEFENSES},
        "marginal_asr_external": {PRETTY[d]: round(float(B_ext[d].mean()), 3) for d in DEFENSES},
        "benign_frr_insample": {PRETTY[d]: round(v, 3) for d, v in frr_in.items()},
        "benign_frr_external": {PRETTY[d]: round(v, 3) for d, v in frr_ex.items()},
        "seed": SEED,
        "grader_prompt_sha256": hashlib.sha256(
            Path("dcorr/judge/strongreject_prompt.txt").read_bytes()).hexdigest(),
    }
    (a.out / "D3_verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")

    print(f"marginal ASR  in-sample -> external:")
    for d in DEFENSES:
        print(f"  {PRETTY[d]:14} {B_in[d].mean():.2f} -> {B_ext[d].mean():.2f}"
              f"   FRR {frr_in.get(d, float('nan')):.2f} -> {frr_ex.get(d, float('nan')):.2f}")
    print(f"\npairs measurable: {len(pairs)}; surviving BH after stratification: {n_surv}")
    print(f"probe16 x probe8 survives under external thresholds: {survives}")
    if len(key):
        print(f"  CMH OR {float(key.iloc[0].CMH_OR):.2f}, q {float(key.iloc[0].q):.4f}")
    print(f"wrote 4 files to {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
