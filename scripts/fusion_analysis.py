#!/usr/bin/env python3
"""
fusion_analysis.py

Computes the classifier-fusion analysis for the layered-defense paper:

  (1) Pairwise diversity measures for every defense pair, in the vocabulary of
      Kuncheva and Whitaker (2003): Q-statistic, disagreement, double-fault,
      and the correlation phi already reported in Table 6.
  (2) An operating curve over combination rules: the current unanimity-veto
      stack is the k=1 special case of a k-of-n rule, and sweeping k trades
      residual attack success against benign false refusal.
  (3) Ensemble selection: the paper's row-based rule 2 against a
      diversity-greedy selection over the same defenses, at every stack size.

It writes a LaTeX fragment (fusion_results.tex) containing the tables and a set
of \\newcommand macros, so the manuscript can cite the numbers without any of
them being transcribed by hand.

--------------------------------------------------------------------------
INPUT FORMAT
--------------------------------------------------------------------------
Two CSVs, both with a header row of defense names. Column order must match.

  breach.csv    one row per behaviour (n=100 for the primary run).
                cell = 1 if the defense was BREACHED on that behaviour under
                the adversary in question, 0 if it held.
                This is exactly the per-behaviour breach vector already in the
                repository, the one every phi in Table 6 is computed from.

  refusal.csv   one row per benign prompt (n=100 matched benign set).
                cell = 1 if the deployed configuration REFUSED that benign
                prompt, 0 otherwise. These are the system-level f_i of H3.

Optional:

  --base-refusal PATH   a single-column CSV of the undefended model's refusals
                        on the same benign prompts. Used to report the
                        attributable refusal burden separately from the base
                        model's own floor (the 7 to 8% Vicuna floor discussed
                        under H3). Does not change the headline numbers.

  --rows PATH           a two-column CSV, defense,row  mapping each defense to
                        its dependency row in Table 8. Required for the
                        row-based selection comparison in part (3).

--------------------------------------------------------------------------
CONVENTIONS, STATED EXPLICITLY BECAUSE THE SIGNS ARE EASY TO FLIP
--------------------------------------------------------------------------
The fusion literature defines diversity over CORRECTNESS. Here a defense is
"correct" on a behaviour when it BLOCKS the attack, so correctness c = 1 - b.
All four measures below are computed on c, matching Kuncheva and Whitaker
directly. Two consequences worth noting in the paper:

  * phi is invariant to flipping both variables, so phi on correctness equals
    the phi on breach already reported. No number changes.
  * the double-fault measure is, by definition, the rate at which both members
    are wrong, which is exactly the joint breach rate ASR_{d1d2} of Eq. (9).
    The paper's Delta is therefore the excess double-fault over independence.
    This is a naming correspondence, not a new result, and should be presented
    as such.

--------------------------------------------------------------------------
CAVEAT THE PAPER MUST CARRY
--------------------------------------------------------------------------
Breach vectors were obtained by re-optimising the attack against each defense
separately. Combination rules evaluated on those vectors are therefore
estimates under the same assumption already used for the pairwise intersection,
which was validated against a direct attack on the assembled stack at k=1 only.
Any k>1 operating point reported from this script is an estimate and should be
labelled as such until a direct attack is run against that configuration.

Usage:
    python3 fusion_analysis.py --breach breach.csv --refusal refusal.csv \
        --rows rows.csv --out fusion_results.tex
    python3 fusion_analysis.py --demo        # synthetic data, checks the pipeline
"""

import argparse
import itertools
import sys

import numpy as np
import pandas as pd

RNG = np.random.default_rng(0)
N_BOOT = 10000


# ---------------------------------------------------------------- diversity


def pair_counts(c1, c2):
    """Correctness-based 2x2 counts, Kuncheva and Whitaker notation."""
    n11 = int(np.sum((c1 == 1) & (c2 == 1)))  # both blocked
    n10 = int(np.sum((c1 == 1) & (c2 == 0)))
    n01 = int(np.sum((c1 == 0) & (c2 == 1)))
    n00 = int(np.sum((c1 == 0) & (c2 == 0)))  # both breached = double fault
    return n11, n10, n01, n00


def q_statistic(c1, c2):
    n11, n10, n01, n00 = pair_counts(c1, c2)
    num = n11 * n00 - n01 * n10
    den = n11 * n00 + n01 * n10
    return np.nan if den == 0 else num / den


def disagreement(c1, c2):
    n11, n10, n01, n00 = pair_counts(c1, c2)
    return (n10 + n01) / (n11 + n10 + n01 + n00)


def double_fault(c1, c2):
    n11, n10, n01, n00 = pair_counts(c1, c2)
    return n00 / (n11 + n10 + n01 + n00)


def phi_coef(c1, c2):
    """Matthews / Pearson correlation of the two indicator vectors."""
    if c1.std() == 0 or c2.std() == 0:
        return np.nan
    return float(np.corrcoef(c1, c2)[0, 1])


MEASURES = {
    "Q": q_statistic,
    "dis": disagreement,
    "DF": double_fault,
    "phi": phi_coef,
}


def boot_ci(fn, c1, c2, n_boot=N_BOOT, alpha=0.05):
    n = len(c1)
    vals = np.empty(n_boot)
    for b in range(n_boot):
        idx = RNG.integers(0, n, n)
        vals[b] = fn(c1[idx], c2[idx])
    vals = vals[~np.isnan(vals)]
    if vals.size == 0:
        return np.nan, np.nan
    return float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))


def diversity_table(breach, degenerate_tol=0):
    """All pairs. Returns a DataFrame. Skips pairs with a constant vector."""
    names = list(breach.columns)
    correct = 1 - breach.to_numpy()
    rows = []
    for i, j in itertools.combinations(range(len(names)), 2):
        c1, c2 = correct[:, i], correct[:, j]
        if c1.std() <= degenerate_tol or c2.std() <= degenerate_tol:
            rows.append(
                dict(d1=names[i], d2=names[j], p1=np.nan, p2=np.nan,
                     Q=np.nan, dis=np.nan, DF=np.nan, phi=np.nan,
                     phi_lo=np.nan, phi_hi=np.nan, DF_indep=np.nan,
                     degenerate=True)
            )
            continue
        b1, b2 = 1 - c1, 1 - c2
        df_indep = b1.mean() * b2.mean()
        lo, hi = boot_ci(phi_coef, c1, c2)
        rows.append(
            dict(
                d1=names[i], d2=names[j],
                p1=float(b1.mean()), p2=float(b2.mean()),
                Q=q_statistic(c1, c2),
                dis=disagreement(c1, c2),
                DF=double_fault(c1, c2),
                phi=phi_coef(c1, c2),
                phi_lo=lo, phi_hi=hi,
                DF_indep=float(df_indep),
                degenerate=False,
            )
        )
    return pd.DataFrame(rows)


def measure_agreement(div):
    """How far the four measures agree on the ranking of pairs (Spearman)."""
    from scipy.stats import spearmanr

    live = div[~div.degenerate]
    keys = ["Q", "dis", "DF", "phi"]
    out = pd.DataFrame(index=keys, columns=keys, dtype=float)
    for a in keys:
        for b in keys:
            r, _ = spearmanr(live[a], live[b])
            out.loc[a, b] = r
    return out


# --------------------------------------------------------- combination rules


def k_of_n_curve(breach, refusal, subset=None):
    """
    Block the input if at least k members block it.

    k=1 is the unanimity-veto stack currently reported: one blocker suffices,
    so the attack succeeds only when every member is breached.
    Increasing k requires more independent flags and so admits more attacks
    while refusing fewer benign prompts.
    """
    cols = list(subset) if subset is not None else list(breach.columns)
    B = breach[cols].to_numpy()
    R = refusal[cols].to_numpy()
    n_layers = len(cols)
    blocks_attack = (1 - B).sum(axis=1)   # how many members blocked the attack
    blocks_benign = R.sum(axis=1)         # how many members refused the benign prompt
    rows = []
    for k in range(1, n_layers + 1):
        asr = float(np.mean(blocks_attack < k))   # stack breached
        frr = float(np.mean(blocks_benign >= k))  # stack refused
        rows.append(dict(k=k, n=n_layers, ASR=asr, FRR=frr))
    return pd.DataFrame(rows)


def single_layer_reference(breach, refusal):
    ref = pd.DataFrame(
        dict(ASR=breach.mean(axis=0), FRR=refusal.mean(axis=0))
    ).sort_values("ASR")
    return ref


# ------------------------------------------------------------- selection


def greedy_diversity_selection(breach, refusal, max_size=None):
    """
    Start from the single strongest defense (lowest marginal ASR), then
    repeatedly add the defense that minimises the double-fault rate with the
    current set, i.e. the one most likely to hold where the set fails.
    Evaluated under the unanimity-veto rule (k=1) for comparability with the
    paper's assembled stack.
    """
    names = list(breach.columns)
    B = breach.to_numpy()
    R = refusal.to_numpy()
    max_size = max_size or len(names)
    chosen = [int(np.argmin(B.mean(axis=0)))]
    out = []

    def evaluate(idx):
        joint_breach = B[:, idx].all(axis=1)
        refused = R[:, idx].any(axis=1)
        return float(joint_breach.mean()), float(refused.mean())

    asr, frr = evaluate(chosen)
    out.append(dict(size=1, members=names[chosen[0]], ASR=asr, FRR=frr))
    while len(chosen) < max_size:
        current_fail = B[:, chosen].all(axis=1)
        best, best_df = None, np.inf
        for c in range(len(names)):
            if c in chosen:
                continue
            df = float(np.mean(current_fail & (B[:, c] == 1)))
            if df < best_df:
                best, best_df = c, df
        chosen.append(best)
        asr, frr = evaluate(chosen)
        out.append(
            dict(size=len(chosen), members=", ".join(names[c] for c in chosen),
                 ASR=asr, FRR=frr)
        )
    return pd.DataFrame(out)


def row_based_selection(breach, refusal, rowmap, max_size=None):
    """
    The paper's rule 2: at most one defense per dependency row, taking the
    strongest available member of each row, rows entered in order of that
    member's marginal ASR. Evaluated under the same k=1 rule.
    """
    names = list(breach.columns)
    B = breach.to_numpy()
    R = refusal.to_numpy()
    marg = B.mean(axis=0)
    best_per_row = {}
    for i, nm in enumerate(names):
        r = rowmap.get(nm)
        if r is None:
            continue
        if r not in best_per_row or marg[i] < marg[best_per_row[r]]:
            best_per_row[r] = i
    order = sorted(best_per_row.values(), key=lambda i: marg[i])
    max_size = max_size or len(order)
    out = []
    for s in range(1, min(max_size, len(order)) + 1):
        idx = order[:s]
        joint_breach = B[:, idx].all(axis=1)
        refused = R[:, idx].any(axis=1)
        out.append(
            dict(size=s, members=", ".join(names[i] for i in idx),
                 ASR=float(joint_breach.mean()), FRR=float(refused.mean()))
        )
    return pd.DataFrame(out)


# ------------------------------------------------------------------ output


def fmt(x, nd=3):
    return "--" if (x is None or (isinstance(x, float) and np.isnan(x))) else f"{x:.{nd}f}"


def write_tex(div, agree, curve, ref, greedy, rowsel, path):
    live = div[~div.degenerate]
    L = []
    L.append("% Auto-generated by fusion_analysis.py. Do not edit by hand.")
    L.append("% Re-run the script to refresh.\n")

    # macros for use in prose
    k1 = curve[curve.k == 1].iloc[0]
    best = ref.iloc[0]
    L.append(f"\\newcommand{{\\StackKoneASR}}{{{fmt(k1.ASR)}}}")
    L.append(f"\\newcommand{{\\StackKoneFRR}}{{{fmt(k1.FRR)}}}")
    L.append(f"\\newcommand{{\\BestLayerName}}{{{best.name}}}")
    L.append(f"\\newcommand{{\\BestLayerASR}}{{{fmt(best.ASR)}}}")
    L.append(f"\\newcommand{{\\BestLayerFRR}}{{{fmt(best.FRR)}}}")
    for _, r in curve.iterrows():
        w = ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven"][int(r.k)]
        L.append(f"\\newcommand{{\\Stack{w}ASR}}{{{fmt(r.ASR)}}}")
        L.append(f"\\newcommand{{\\Stack{w}FRR}}{{{fmt(r.FRR)}}}")
    L.append(f"\\newcommand{{\\QMin}}{{{fmt(live.Q.min())}}}")
    L.append(f"\\newcommand{{\\QMax}}{{{fmt(live.Q.max())}}}")
    L.append(f"\\newcommand{{\\DFMin}}{{{fmt(live.DF.min())}}}")
    L.append(f"\\newcommand{{\\DFMax}}{{{fmt(live.DF.max())}}}")
    L.append(f"\\newcommand{{\\NPairsLive}}{{{len(live)}}}")
    L.append(f"\\newcommand{{\\NPairsQPos}}{{{int((live.Q > 0).sum())}}}")
    L.append(f"\\newcommand{{\\NPairsDFExcess}}{{{int((live.DF > live.DF_indep).sum())}}}")
    L.append(f"\\newcommand{{\\SpearmanQphi}}{{{fmt(agree.loc['Q','phi'], 2)}}}")
    L.append(f"\\newcommand{{\\SpearmanDisphi}}{{{fmt(agree.loc['dis','phi'], 2)}}}")
    L.append(f"\\newcommand{{\\SpearmanDFphi}}{{{fmt(agree.loc['DF','phi'], 2)}}}")
    L.append("")

    # Table: diversity measures
    L.append(r"\begin{table*}[width=\FullWidth,pos=t]")
    L.append(r"\centering")
    L.append(r"\caption{Pairwise diversity for every measurable defense pair, in the")
    L.append(r"vocabulary of classifier fusion~\citep{kuncheva2003measures}. Measures are")
    L.append(r"computed on the correctness indicator (a defense is correct when it blocks),")
    L.append(r"so $Q$, disagreement, and double-fault carry their standard signs. The")
    L.append(r"double-fault rate is by definition the joint breach rate of")
    L.append(r"Eq.~\eqref{eq:joint}, and $DF_{\mathrm{ind}}$ is its value under independence,")
    L.append(r"so $DF - DF_{\mathrm{ind}}$ is the $\Delta$ of Table~\ref{tab:corrmeasure}.")
    L.append(r"$\phi$ is invariant to the sign convention and is unchanged from that table.}")
    L.append(r"\label{tab:diversity}")
    L.append(r"\small")
    L.append(r"\begin{tabular*}{\tblwidth}{@{\extracolsep{\fill}}llrrrrrl@{}}")
    L.append(r"\toprule")
    L.append(r"Defense pair & & $\hat p_1$ & $\hat p_2$ & $Q$ & dis. & $DF$ & $DF_{\mathrm{ind}}$ \\")
    L.append(r"\midrule")
    for _, r in live.sort_values("DF", ascending=False).iterrows():
        L.append(
            f"{r.d1} & $\\times$ {r.d2} & {fmt(r.p1,2)} & {fmt(r.p2,2)} & "
            f"{fmt(r.Q)} & {fmt(r.dis)} & {fmt(r.DF)} & {fmt(r.DF_indep)} \\\\"
        )
    L.append(r"\bottomrule")
    L.append(r"\end{tabular*}")
    L.append(r"\end{table*}")
    L.append("")

    # Table: combination rules
    L.append(r"\begin{table}[pos=t]")
    L.append(r"\centering")
    L.append(r"\caption{Combination rules over the same seven defenses. The stack blocks")
    L.append(r"an input when at least $k$ members block it; $k=1$ is the unanimity-veto")
    L.append(r"configuration reported in Section~\ref{sec:corr-results}. Residual attack")
    L.append(r"success and benign false refusal are estimated from the stored per-defense")
    L.append(r"vectors and are subject to the caveat of Section~\ref{sec:limits-corr}.}")
    L.append(r"\label{tab:combrules}")
    L.append(r"\small")
    L.append(r"\begin{tabular*}{\tblwidth}{@{\extracolsep{\fill}}lrr@{}}")
    L.append(r"\toprule")
    L.append(r"Rule & residual ASR & benign FRR \\")
    L.append(r"\midrule")
    for _, r in curve.iterrows():
        lbl = f"$k={int(r.k)}$ of {int(r.n)}"
        if r.k == 1:
            lbl += " (veto, as reported)"
        L.append(f"{lbl} & {fmt(r.ASR)} & {fmt(r.FRR)} \\\\")
    L.append(r"\midrule")
    L.append(f"strongest single layer ({best.name}) & {fmt(best.ASR)} & {fmt(best.FRR)} \\\\")
    L.append(r"\bottomrule")
    L.append(r"\end{tabular*}")
    L.append(r"\end{table}")
    L.append("")

    # Table: selection
    L.append(r"\begin{table}[pos=t]")
    L.append(r"\centering")
    L.append(r"\caption{Ensemble selection at each stack size, under the veto rule:")
    L.append(r"the row-based heuristic of composition rule 2 against a diversity-greedy")
    L.append(r"selection that adds the member minimising double-fault with the current set.}")
    L.append(r"\label{tab:selection}")
    L.append(r"\small")
    L.append(r"\begin{tabular*}{\tblwidth}{@{\extracolsep{\fill}}lrrrr@{}}")
    L.append(r"\toprule")
    L.append(r"& \multicolumn{2}{c}{row-based (rule 2)} & \multicolumn{2}{c}{diversity-greedy} \\")
    L.append(r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}")
    L.append(r"Size & ASR & FRR & ASR & FRR \\")
    L.append(r"\midrule")
    for s in range(1, max(len(greedy), len(rowsel)) + 1):
        gr = greedy[greedy["size"] == s]
        rs = rowsel[rowsel["size"] == s]
        ga, gf = (fmt(gr.iloc[0].ASR), fmt(gr.iloc[0].FRR)) if len(gr) else ("--", "--")
        ra, rf = (fmt(rs.iloc[0].ASR), fmt(rs.iloc[0].FRR)) if len(rs) else ("--", "--")
        L.append(f"{s} & {ra} & {rf} & {ga} & {gf} \\\\")
    L.append(r"\bottomrule")
    L.append(r"\end{tabular*}")
    L.append(r"\end{table}")

    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")


# -------------------------------------------------------------------- demo


def make_demo():
    """
    Synthetic data with the qualitative structure reported in the paper:
    one near-perfect semantic layer, two highly correlated probes, and
    several mid-strength layers sharing a behaviour-difficulty gradient.
    FOR PIPELINE TESTING ONLY. These are not results.
    """
    n = 100
    names = ["perplexity", "token-anomaly", "llama-guard", "refusal-prime",
             "smoothllm", "probe16", "probe8"]
    difficulty = RNG.normal(size=n)          # shared common cause
    base_p = dict(perplexity=0.66, **{"token-anomaly": 0.35}, **{"llama-guard": 0.01},
                  **{"refusal-prime": 0.60}, smoothllm=0.54, probe16=0.68, probe8=0.60)
    probe_shared = RNG.normal(size=n)
    B = {}
    for nm in names:
        z = 0.9 * difficulty + 0.6 * RNG.normal(size=n)
        if nm.startswith("probe"):
            z = 0.5 * difficulty + 1.6 * probe_shared + 0.4 * RNG.normal(size=n)
        thr = np.quantile(z, 1 - base_p[nm])
        B[nm] = (z > thr).astype(int)
    breach = pd.DataFrame(B)[names]
    f = dict(perplexity=0.08, **{"token-anomaly": 0.09}, **{"llama-guard": 0.26},
             **{"refusal-prime": 0.39}, smoothllm=0.26, probe16=0.08, probe8=0.09)
    floor = (RNG.random(n) < 0.075).astype(int)   # shared base-model refusal floor
    R = {nm: np.maximum(floor, (RNG.random(n) < max(f[nm] - 0.075, 0)).astype(int))
         for nm in names}
    refusal = pd.DataFrame(R)[names]
    rows = dict(perplexity="token surface", **{"token-anomaly": "token surface"},
                **{"llama-guard": "semantic"}, **{"refusal-prime": "first-token"},
                smoothllm="perturbation", probe16="internal", probe8="internal")
    return breach, refusal, rows


# -------------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--breach")
    ap.add_argument("--refusal")
    ap.add_argument("--rows", help="CSV: defense,row")
    ap.add_argument("--base-refusal")
    ap.add_argument("--exclude", default="",
                    help="comma-separated defenses to drop before analysis. Use this to "
                         "re-run the selection comparison without a near-perfect member: "
                         "with one layer at ASR 0.01 there is no headroom for any rule to "
                         "improve on, so the comparison is floor-limited and uninformative.")
    ap.add_argument("--out", default="fusion_results.tex")
    ap.add_argument("--boot", type=int, default=N_BOOT)
    ap.add_argument("--demo", action="store_true")
    a = ap.parse_args()

    globals()["N_BOOT"] = a.boot

    if a.demo:
        print("DEMO MODE: synthetic data. The numbers below are not results.\n")
        breach, refusal, rowmap = make_demo()
    else:
        if not (a.breach and a.refusal):
            ap.error("--breach and --refusal are required unless --demo is given")
        breach = pd.read_csv(a.breach)
        refusal = pd.read_csv(a.refusal)
        if list(breach.columns) != list(refusal.columns):
            ap.error(f"column mismatch:\n  breach : {list(breach.columns)}\n"
                     f"  refusal: {list(refusal.columns)}")
        rowmap = {}
        if a.rows:
            rm = pd.read_csv(a.rows)
            rowmap = dict(zip(rm.iloc[:, 0], rm.iloc[:, 1]))

    if a.exclude:
        drop = [c.strip() for c in a.exclude.split(",") if c.strip()]
        missing = [c for c in drop if c not in breach.columns]
        if missing:
            ap.error(f"--exclude names unknown defenses: {missing}")
        breach = breach.drop(columns=drop)
        refusal = refusal.drop(columns=drop)
        print(f"excluded: {drop}\n")

    for nm, d in (("breach", breach), ("refusal", refusal)):
        bad = set(np.unique(d.to_numpy())) - {0, 1}
        if bad:
            ap.error(f"{nm} must be binary; found {sorted(bad)}")

    print(f"defenses: {list(breach.columns)}")
    print(f"behaviours: {len(breach)}   benign prompts: {len(refusal)}\n")

    ref = single_layer_reference(breach, refusal)
    print("--- single layers (marginal) ---")
    print(ref.round(3).to_string(), "\n")

    div = diversity_table(breach)
    live = div[~div.degenerate]
    print(f"--- pairwise diversity: {len(live)} measurable pairs, "
          f"{len(div) - len(live)} degenerate ---")
    print(live[["d1", "d2", "Q", "dis", "DF", "DF_indep", "phi"]]
          .sort_values("DF", ascending=False).round(3).to_string(index=False), "\n")
    print(f"pairs with Q > 0 (positively associated errors): "
          f"{int((live.Q > 0).sum())} / {len(live)}")
    print(f"pairs with double-fault above independence: "
          f"{int((live.DF > live.DF_indep).sum())} / {len(live)}\n")

    agree = measure_agreement(div)
    print("--- Spearman agreement between diversity measures ---")
    print(agree.round(2).to_string(), "\n")

    curve = k_of_n_curve(breach, refusal)
    print("--- combination rules (k-of-n over all defenses) ---")
    print(curve.round(3).to_string(index=False), "\n")

    greedy = greedy_diversity_selection(breach, refusal)
    print("--- diversity-greedy selection ---")
    print(greedy.round(3).to_string(index=False), "\n")

    if rowmap:
        rowsel = row_based_selection(breach, refusal, rowmap)
        print("--- row-based selection (composition rule 2) ---")
        print(rowsel.round(3).to_string(index=False), "\n")
    else:
        rowsel = pd.DataFrame(columns=["size", "members", "ASR", "FRR"])
        print("no --rows given, skipping the row-based selection comparison\n")

    if a.base_refusal:
        base = pd.read_csv(a.base_refusal).iloc[:, 0].to_numpy()
        print(f"undefended base-model refusal floor: {base.mean():.3f}\n")

    div.to_csv("fusion_diversity.csv", index=False)
    curve.to_csv("fusion_combination_rules.csv", index=False)
    write_tex(div, agree, curve, ref, greedy, rowsel, a.out)
    print(f"wrote {a.out}, fusion_diversity.csv, fusion_combination_rules.csv")


if __name__ == "__main__":
    sys.exit(main())
