"""Stage 11: emit every manuscript number derivable from stored artifacts.

Answers items A1, B1, B2, B4 and B5 of the data request. Nothing here needs new compute;
it is all pivots and resampling over results/hpc_vicuna_autodan/.

CONVENTIONS, matched to the request:
  * odds ratios      Haldane-Anscombe +0.5 applied UNIFORMLY, not only to zero cells.
                     This reproduces the raw_or column of confound_check.json.
  * multiplicity     Benjamini-Hochberg over the fifteen non-degenerate pairs.
  * intervals        10,000-resample bootstrap percentile over behaviours.
  * precision        three decimals.

Outputs land in fusion/manuscript/ as CSV, one file per item, plus manifest.json.

Usage:
  python scripts/11_manuscript_numbers.py --fusion fusion --run results/hpc_vicuna_autodan \
      --out fusion/manuscript
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fusion_analysis import (  # noqa: E402
    cmh_all_pairs, disagreement, double_fault, greedy_diversity_selection,
    k_of_n_curve, pair_counts, phi_coef, q_statistic, row_based_selection,
)

BOOT = 10000


def haldane_or(b1, b2, c=0.5):
    """Crude odds ratio with a uniform +c added to all four cells."""
    n11 = float(np.sum((b1 == 1) & (b2 == 1))) + c
    n10 = float(np.sum((b1 == 1) & (b2 == 0))) + c
    n01 = float(np.sum((b1 == 0) & (b2 == 1))) + c
    n00 = float(np.sum((b1 == 0) & (b2 == 0))) + c
    return (n11 * n00) / (n10 * n01)


def boot(fn, x, y, rng, n_boot=BOOT, alpha=0.05):
    v = np.empty(n_boot)
    for k in range(n_boot):
        i = rng.integers(0, len(x), len(x))
        v[k] = fn(x[i], y[i])
    v = v[~np.isnan(v)]
    if v.size == 0:
        return np.nan, np.nan
    return float(np.quantile(v, alpha / 2)), float(np.quantile(v, 1 - alpha / 2))


def bh(p):
    """Benjamini-Hochberg over a 1-D array, NaNs preserved."""
    p = np.asarray(p, dtype=float)
    q = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    v = p[ok]
    order = np.argsort(v)
    m = len(v)
    adj = v[order] * m / np.arange(1, m + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.clip(adj, 0, 1)
    q[ok] = out
    return q


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fusion", required=True, type=Path)
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--boot", type=int, default=BOOT)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    breach = pd.read_csv(a.fusion / "breach.csv")
    refusal = pd.read_csv(a.fusion / "refusal.csv")
    rowmap = dict(pd.read_csv(a.fusion / "rows.csv").itertuples(index=False))
    live = [c for c in breach.columns if c != "llamaguard"]
    manifest: dict = {"n_behaviours": len(breach), "n_benign": len(refusal),
                      "defenses": list(breach.columns), "boot": a.boot,
                      "or_convention": "Haldane-Anscombe +0.5 applied uniformly",
                      "bh_over": "the 15 non-degenerate pairs (Llama Guard excluded)"}

    # ---- A1 + B4a: CMH for every pair, Haldane crude OR, BH over the 15 -----------------
    B = breach.to_numpy()
    names = list(breach.columns)
    cmh15 = cmh_all_pairs(breach[live])          # BH computed over 15 inside
    cmh21 = cmh_all_pairs(breach)                # all 21, for the supplement
    for tbl in (cmh15, cmh21):
        tbl["crude_OR_haldane"] = [
            haldane_or(B[:, names.index(r.d1)], B[:, names.index(r.d2)])
            for r in tbl.itertuples()
        ]
    cols = ["d1", "d2", "crude_OR_haldane", "crude_OR", "CMH_OR", "p", "q",
            "n_strata", "min_stratum"]
    cmh15[cols].round(4).to_csv(a.out / "A1_B4a_cmh_15pairs.csv", index=False)
    cmh21[[c for c in cols if c in cmh21.columns]].round(4).to_csv(
        a.out / "B4a_cmh_all21pairs.csv", index=False)

    a1 = cmh15[((cmh15.d1 == "ppl_filter") & (cmh15.d2 == "token_anomaly"))
               | ((cmh15.d1 == "token_anomaly") & (cmh15.d2 == "ppl_filter"))].iloc[0]
    manifest["A1_ppl_x_token_anomaly"] = {
        "crude_OR_haldane": round(float(a1.crude_OR_haldane), 3),
        "CMH_OR": round(float(a1.CMH_OR), 3),
        "p": round(float(a1.p), 4),
        "q": round(float(a1.q), 4),
        "n_strata": int(a1.n_strata), "min_stratum": int(a1.min_stratum),
    }

    # ---- B1: the full k-of-n curve, every k --------------------------------------------
    curve = k_of_n_curve(breach, refusal)
    curve.round(4).to_csv(a.out / "B1_k_of_n_full.csv", index=False)
    curve6 = k_of_n_curve(breach[live], refusal[live])
    curve6.round(4).to_csv(a.out / "B1_k_of_n_no_llamaguard.csv", index=False)

    # ---- B2: selection at every size, and the row-based ceiling ------------------------
    greedy = greedy_diversity_selection(breach[live], refusal[live])
    rowsel = row_based_selection(breach[live], refusal[live], rowmap)
    greedy.round(4).to_csv(a.out / "B2_greedy_no_llamaguard.csv", index=False)
    rowsel.round(4).to_csv(a.out / "B2_rowbased_no_llamaguard.csv", index=False)
    rows_present = sorted({rowmap[d] for d in live})
    agree_to = 0
    for s in range(1, min(len(greedy), len(rowsel)) + 1):
        g = greedy[greedy["size"] == s].iloc[0]
        r = rowsel[rowsel["size"] == s].iloc[0]
        if set(g.members.split(", ")) == set(r.members.split(", ")):
            agree_to = s
        else:
            break
    manifest["B2"] = {
        "dependency_rows_spanned": rows_present,
        "rowbased_max_size": int(rowsel["size"].max()),
        "rowbased_ceiling_reason": (
            "one member per row; the six non-Llama-Guard defenses span "
            f"{len(rows_present)} rows, so rule 2 cannot exceed size {len(rows_present)}"),
        "greedy_max_size": int(greedy["size"].max()),
        "identical_membership_up_to_size": agree_to,
    }

    # ---- B4b: all 21 diversity rows, with B5 intervals ---------------------------------
    C = 1 - B
    div = []
    for i, j in itertools.combinations(range(len(names)), 2):
        c1, c2 = C[:, i], C[:, j]
        rec = {"d1": names[i], "d2": names[j],
               "p1": float(B[:, i].mean()), "p2": float(B[:, j].mean())}
        if c1.std() == 0 or c2.std() == 0:
            rec.update({k: np.nan for k in
                        ("Q", "Q_lo", "Q_hi", "dis", "dis_lo", "dis_hi",
                         "DF", "DF_lo", "DF_hi", "phi", "phi_lo", "phi_hi")})
            rec["degenerate"] = True
        else:
            for lbl, fn in (("Q", q_statistic), ("dis", disagreement),
                            ("DF", double_fault), ("phi", phi_coef)):
                rec[lbl] = fn(c1, c2)
                rec[lbl + "_lo"], rec[lbl + "_hi"] = boot(fn, c1, c2, rng, a.boot)
            rec["degenerate"] = False
        rec["DF_indep"] = float(B[:, i].mean() * B[:, j].mean())
        div.append(rec)
    div = pd.DataFrame(div)
    div.round(4).to_csv(a.out / "B4b_B5_diversity_all21_with_ci.csv", index=False)

    # ---- B4c: the 21-pair benign-refusal phi matrix ------------------------------------
    R = refusal.to_numpy()
    rec = []
    for i, j in itertools.combinations(range(len(names)), 2):
        x, y = R[:, i], R[:, j]
        p = phi_coef(x, y)
        lo, hi = boot(phi_coef, x, y, rng, a.boot) if not np.isnan(p) else (np.nan, np.nan)
        rec.append({"d1": names[i], "d2": names[j], "f1": float(x.mean()),
                    "f2": float(y.mean()), "phi": p, "phi_lo": lo, "phi_hi": hi})
    ref_phi = pd.DataFrame(rec)
    ref_phi.round(4).to_csv(a.out / "B4c_benign_refusal_phi_21pairs.csv", index=False)
    mat = pd.DataFrame(np.eye(len(names)), index=names, columns=names)
    for r in ref_phi.itertuples():
        mat.loc[r.d1, r.d2] = mat.loc[r.d2, r.d1] = r.phi
    mat.round(3).to_csv(a.out / "B4c_benign_refusal_phi_matrix.csv")
    manifest["B4c"] = {"n_pairs": len(ref_phi),
                       "mean_phi": round(float(ref_phi.phi.mean()), 3),
                       "min_phi": round(float(ref_phi.phi.min()), 3),
                       "max_phi": round(float(ref_phi.phi.max()), 3),
                       "positive": int((ref_phi.phi > 0).sum())}

    (a.out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"--- A1: perplexity x token-anomaly (the missing Table 12 row) ---")
    for k, v in manifest["A1_ppl_x_token_anomaly"].items():
        print(f"  {k:20} {v}")
    print(f"\n--- B1: k-of-n, all seven defenses ---")
    print(curve.round(3).to_string(index=False))
    print(f"\n--- B2: selection, Llama Guard excluded ---")
    print("greedy:");  print(greedy.round(3).to_string(index=False))
    print("row-based:"); print(rowsel.round(3).to_string(index=False))
    print(f"  rows spanned: {rows_present}")
    print(f"  identical membership up to size {agree_to}")
    print(f"\n--- B4c: benign-refusal phi ---")
    print(f"  {manifest['B4c']}")
    print(f"\nwrote {len(list(a.out.glob('*')))} files to {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
