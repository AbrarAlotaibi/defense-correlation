#!/usr/bin/env python3
"""
input_level_phi.py

Resolves the estimand mismatch (REVIEW.md R3) between the composition model and
the measurement.

THE PROBLEM
-----------
Eq. (9) defines b_l(i) as the event that layer l is breached on INPUT i, and the
joint residual is derived assuming both layers face the same input. The
measurement re-optimises the attack against each defense separately, so the
stored vectors are indexed by BEHAVIOUR: for a pair (d1, d2), d1's entry comes
from a prompt optimised against d1 and d2's from a different prompt optimised
against d2. Those are two different estimands:

  phi_behaviour   both layers scored on the same behaviour, different prompts.
                  This is what a deployer faces when the adversary is free to
                  retarget, and it is what the paper currently reports.

  phi_input       both layers scored on the same prompt. This is the quantity
                  in Eq. (9) as written.

They are not interchangeable and the difference is measurable. This script
measures it.

THE MEASUREMENT
---------------
Score every defense on prompt sets it did not generate, then compute phi within
each shared set. Three tiers, cheapest first.

  TIER 0, free, no new generations.
    The two static baselines already reported are shared-input by construction:
    the plain behaviour prompt is byte-identical across defenses, and the
    transfer suffix is one fixed suffix reused for all of them. Feeding those
    two breach matrices to this script gives an input-level phi today. The
    limitation is low marginals (0.00 to 0.28), so several pairs will be
    degenerate and the intervals will be wide. It is a bracket, not a
    replacement.

  TIER 1, roughly 100 x 7 generations.
    Run the fluent adversary once against the UNDEFENDED model, take the
    resulting prompt set, and score all seven defenses on it. One shared set,
    adaptive against the model but not against any defense.

  TIER 2, roughly 42 x 100 generations, the definitive version.
    The full cross-evaluation matrix: for each source defense s, score all seven
    defenses on the prompts optimised against s. The diagonal is the data you
    already have. For a pair (d1, d2) this yields phi under an adversary
    adaptive against d1 and again under one adaptive against d2, which brackets
    the deployment case from both sides.

None of these tiers requires a new attack search. They are scoring passes over
prompts that already exist, plus the judge.

INPUT FORMAT
------------
One long CSV, cross_breach.csv, with four columns:

    source, defense, behavior, breach

  source    the prompt set the input came from. Use "adaptive:<defense>" for
            prompts optimised against that defense (the diagonal you already
            have), and any label you like for shared sets, e.g. "static:plain",
            "static:transfer", "shared:autodan-undefended".
  defense   the defense that was evaluated on that prompt.
  behavior  behaviour id, shared across sources so rows can be aligned.
  breach    1 if the defense was breached on that input, else 0.

Rows for (source="adaptive:X", defense="X") reproduce the vectors behind
Table 6, and the script uses them to recompute phi_behaviour as a self-check. If
that check does not reproduce the published values, the data assembly is wrong
and everything downstream is suspect, so it is reported first and loudly.

Usage:
    python3 input_level_phi.py --cross cross_breach.csv --out r3_results.tex
    python3 input_level_phi.py --demo
"""

import argparse
import itertools
import sys

import numpy as np
import pandas as pd

RNG = np.random.default_rng(0)
N_BOOT = 10000


def phi_coef(b1, b2):
    if b1.std() == 0 or b2.std() == 0:
        return np.nan
    return float(np.corrcoef(b1, b2)[0, 1])


def boot_ci(b1, b2, n_boot=None, alpha=0.05):
    n_boot = n_boot or N_BOOT
    n = len(b1)
    v = np.empty(n_boot)
    for k in range(n_boot):
        idx = RNG.integers(0, n, n)
        v[k] = phi_coef(b1[idx], b2[idx])
    v = v[~np.isnan(v)]
    if v.size == 0:
        return np.nan, np.nan
    return float(np.quantile(v, alpha / 2)), float(np.quantile(v, 1 - alpha / 2))


def wide(df, source):
    """behaviour x defense matrix for one prompt set."""
    sub = df[df.source == source]
    return sub.pivot(index="behavior", columns="defense", values="breach").sort_index()


def phi_behaviour(df, d1, d2):
    """The paper's estimand: each defense scored on its own optimised prompts."""
    a = df[(df.source == f"adaptive:{d1}") & (df.defense == d1)].sort_values("behavior")
    b = df[(df.source == f"adaptive:{d2}") & (df.defense == d2)].sort_values("behavior")
    if a.empty or b.empty:
        return np.nan, (np.nan, np.nan), np.nan, np.nan
    common = np.intersect1d(a.behavior, b.behavior)
    x = a.set_index("behavior").loc[common, "breach"].to_numpy()
    y = b.set_index("behavior").loc[common, "breach"].to_numpy()
    return phi_coef(x, y), boot_ci(x, y), float(x.mean()), float(y.mean())


def phi_input(df, source, d1, d2):
    """Both defenses scored on the same prompt set."""
    W = wide(df, source)
    if d1 not in W.columns or d2 not in W.columns:
        return np.nan, (np.nan, np.nan), np.nan, np.nan
    x = W[d1].to_numpy()
    y = W[d2].to_numpy()
    return phi_coef(x, y), boot_ci(x, y), float(x.mean()), float(y.mean())


def analyse(df, defenses=None):
    defenses = defenses or sorted(df.defense.unique())
    shared = [s for s in sorted(df.source.unique()) if not s.startswith("adaptive:")]
    rows = []
    for d1, d2 in itertools.combinations(defenses, 2):
        pb, (lo, hi), p1, p2 = phi_behaviour(df, d1, d2)
        rec = dict(d1=d1, d2=d2, phi_behaviour=pb, phi_beh_lo=lo, phi_beh_hi=hi,
                   p1=p1, p2=p2)
        # adaptive-against-one-member sources (tier 2)
        for tgt, lbl in ((d1, "phi_input_src_d1"), (d2, "phi_input_src_d2")):
            src = f"adaptive:{tgt}"
            if src in set(df.source):
                pi, (l2, h2), _, _ = phi_input(df, src, d1, d2)
                rec[lbl] = pi
                rec[lbl + "_lo"], rec[lbl + "_hi"] = l2, h2
        # shared sets (tiers 0 and 1)
        for s in shared:
            pi, (l3, h3), q1, q2 = phi_input(df, s, d1, d2)
            rec[f"phi_{s}"] = pi
            # keep the interval: a shared-input phi resting on six or eight breach events
            # is not interpretable without it, and these columns are the low-marginal ones.
            rec[f"phi_{s}_lo"], rec[f"phi_{s}_hi"] = l3, h3
            rec[f"marg_{s}"] = np.nan if np.isnan(q1) else round((q1 + q2) / 2, 3)
        rows.append(rec)
    return pd.DataFrame(rows), shared


def selfcheck(df, published=None):
    """Recompute phi_behaviour and compare with the published Table 6 values."""
    if not published:
        return None
    out = []
    for (d1, d2), val in published.items():
        pb, _, _, _ = phi_behaviour(df, d1, d2)
        out.append(dict(pair=f"{d1} x {d2}", recomputed=pb, published=val,
                        diff=np.nan if np.isnan(pb) else round(pb - val, 3)))
    return pd.DataFrame(out)


def load_published(path):
    """Published phi per pair, from a Table 6 CSV with d1, d2, phi columns."""
    pub = pd.read_csv(path)
    need = {"d1", "d2", "phi"}
    if not need.issubset(pub.columns):
        raise SystemExit(f"--published needs columns {sorted(need)}; got {list(pub.columns)}")
    return {(r.d1, r.d2): float(r.phi) for r in pub.itertuples()}


def write_tex(res, shared, path):
    L = ["% Auto-generated by input_level_phi.py. Do not edit by hand.",
         r"\begin{table}[pos=t]", r"\centering",
         r"\caption{Failure correlation under the two estimands. "
         r"$\phi_{\mathrm{beh}}$ scores each defense on prompts optimised against "
         r"itself, which is what a deployer faces when the adversary retargets, and "
         r"is the quantity reported in Table~\ref{tab:corrmeasure}. "
         r"$\phi_{\mathrm{inp}}$ scores both defenses on a common prompt set, which "
         r"is the quantity in Eq.~\eqref{eq:joint} as written. Columns "
         r"$\mathrm{src}\,d_1$ and $\mathrm{src}\,d_2$ use the prompts optimised "
         r"against the first and second member respectively.}",
         r"\label{tab:estimands}", r"\small",
         r"\begin{tabular*}{\tblwidth}{@{\extracolsep{\fill}}lrrr" + "r" * len(shared) + r"@{}}",
         r"\toprule",
         r"Defense pair & $\phi_{\mathrm{beh}}$ & $\mathrm{src}\,d_1$ & $\mathrm{src}\,d_2$"
         + "".join(f" & {s.replace('_', ' ')}" for s in shared) + r" \\",
         r"\midrule"]

    def f(x):
        return "--" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.2f}"

    for _, r in res.sort_values("phi_behaviour", ascending=False).iterrows():
        cells = [f(r.get("phi_behaviour")), f(r.get("phi_input_src_d1")),
                 f(r.get("phi_input_src_d2"))]
        cells += [f(r.get(f"phi_{s}")) for s in shared]
        L.append(f"{r.d1} $\\times$ {r.d2} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular*}", r"\end{table}"]
    with open(path, "w") as fh:
        fh.write("\n".join(L) + "\n")


def make_demo():
    """Synthetic cross-evaluation matrix. FOR PIPELINE TESTING ONLY."""
    n = 100
    names = ["perplexity", "token-anomaly", "refusal-prime", "smoothllm",
             "probe16", "probe8"]
    difficulty = RNG.normal(size=n)
    probe_shared = RNG.normal(size=n)
    rows = []
    sources = [f"adaptive:{d}" for d in names] + ["static:plain", "static:transfer"]
    for src in sources:
        # an attack optimised against s is stronger against s than against others
        for d in names:
            z = 0.9 * difficulty + 0.6 * RNG.normal(size=n)
            if d.startswith("probe"):
                z = 0.5 * difficulty + 1.5 * probe_shared + 0.4 * RNG.normal(size=n)
            p = 0.55
            if src == f"adaptive:{d}":
                p = 0.65
            elif src.startswith("adaptive:"):
                p = 0.40
            elif src == "static:plain":
                p = 0.07
            else:
                p = 0.20
            thr = np.quantile(z, 1 - p)
            b = (z > thr).astype(int)
            for i in range(n):
                rows.append(dict(source=src, defense=d, behavior=i, breach=int(b[i])))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cross")
    ap.add_argument("--out", default="r3_results.tex")
    ap.add_argument("--boot", type=int, default=N_BOOT)
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--published",
                    help="Table 6 CSV (columns d1, d2, phi). Recomputes phi_behaviour from "
                         "the assembled data and compares. Reported first; a mismatch aborts, "
                         "since it means the assembly is wrong and everything downstream with "
                         "it. Use --tolerate-selfcheck to downgrade that to a warning.")
    ap.add_argument("--tolerate-selfcheck", action="store_true")
    a = ap.parse_args()
    globals()["N_BOOT"] = a.boot

    if a.demo:
        print("DEMO MODE: synthetic data. The numbers below are not results.\n")
        df = make_demo()
    else:
        if not a.cross:
            ap.error("--cross is required unless --demo is given")
        df = pd.read_csv(a.cross)
        need = {"source", "defense", "behavior", "breach"}
        if not need.issubset(df.columns):
            ap.error(f"cross CSV needs columns {sorted(need)}; got {list(df.columns)}")

    # The docstring promises this is reported first and loudly, so it is.
    if a.published:
        chk = selfcheck(df, load_published(a.published))
        live = chk.dropna(subset=["recomputed"])
        worst = float(live["diff"].abs().max()) if len(live) else np.nan
        print("=== SELF-CHECK: phi_behaviour against the published values ===")
        print(live.round(4).to_string(index=False))
        print(f"pairs compared: {len(live)} of {len(chk)}; max |diff| = {worst:.2e}")
        if not len(live) or worst > 5e-4:
            msg = ("SELF-CHECK FAILED. The assembled data does not reproduce the published "
                   "phi, so every number below is suspect.")
            if not a.tolerate_selfcheck:
                print(msg, file=sys.stderr)
                return 2
            print("WARNING: " + msg)
        else:
            print("self-check passed: the assembly reproduces the published table\n")

    print(f"sources  : {sorted(df.source.unique())}")
    print(f"defenses : {sorted(df.defense.unique())}")
    print(f"behaviours: {df.behavior.nunique()}\n")

    # coverage: which cells of the source x defense matrix exist
    cov = df.groupby(["source", "defense"]).size().unstack(fill_value=0)
    print("--- cells present (rows = prompt set, cols = defense evaluated) ---")
    print((cov > 0).astype(int).to_string(), "\n")

    res, shared = analyse(df)
    cols = ["d1", "d2", "p1", "p2", "phi_behaviour", "phi_input_src_d1",
            "phi_input_src_d2"] + [f"phi_{s}" for s in shared]
    cols = [c for c in cols if c in res.columns]
    print("--- phi under each estimand ---")
    print(res[cols].round(3).to_string(index=False), "\n")

    live = res.dropna(subset=["phi_behaviour"])
    for lbl in ["phi_input_src_d1", "phi_input_src_d2"] + [f"phi_{s}" for s in shared]:
        if lbl not in live.columns:
            continue
        sub = live.dropna(subset=[lbl])
        if sub.empty:
            print(f"{lbl}: no measurable pairs")
            continue
        d = sub[lbl] - sub.phi_behaviour
        agree = int((np.sign(sub[lbl]) == np.sign(sub.phi_behaviour)).sum())
        print(f"{lbl}: n={len(sub)}, mean shift vs phi_behaviour {d.mean():+.3f} "
              f"(range {d.min():+.3f} to {d.max():+.3f}), same sign {agree}/{len(sub)}, "
              f"positive {int((sub[lbl] > 0).sum())}/{len(sub)}")
    print()

    res.to_csv("r3_estimand_comparison.csv", index=False)
    write_tex(res, shared, a.out)
    print(f"wrote {a.out}, r3_estimand_comparison.csv")
    print("\nREAD THE SIGN COLUMN FIRST. If phi_input is positive for the same pairs")
    print("as phi_behaviour, the composition conclusion holds under both estimands and")
    print("R3 becomes a clarification. If the two disagree in sign anywhere, the")
    print("behaviour-level framing has to be argued rather than assumed.")


if __name__ == "__main__":
    sys.exit(main())
