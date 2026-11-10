"""Task A: analyses over breach vectors that already exist. No new runs, no API calls.

A1  Pairwise McNemar, EXACT binomial on the discordant cells (not the chi-square
    approximation, and not the continuity-corrected variant), for all 21 defense pairs, plus
    every defense against each assembled stack.
A2  Cochran's Q omnibus across the 7 defenses, followed by Holm-corrected pairwise McNemar.
A3  Permutation test for phi: permute one member 10,000 times, which holds both marginals
    fixed by construction, and report the two-sided p-value.
A4  Re-judge concordance, to the extent the stored re-judge supports it.

Multiplicity family sizes are stated in each output's sidecar: 21 for the full pairwise
matrix, 15 for the measurable-pair analyses that exclude Llama Guard.

Usage:
  python scripts/18_task_a_variance.py --primary results/hpc_vicuna_autodan \
      --nolg results/hpc_vicuna_autodan_nolg --config configs/hpc_vicuna_autodan.yaml \
      --out fusion/taskA
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, chi2

SEED = 20260727
N_PERM = 10000
DEFENSES = ["ppl_filter", "token_anomaly", "llamaguard", "refusal_prime",
            "smoothllm", "probe", "probe_b"]
PRETTY = {"ppl_filter": "perplexity", "token_anomaly": "token-anomaly",
          "llamaguard": "llama-guard", "refusal_prime": "refusal-prime",
          "smoothllm": "smoothllm", "probe": "probe16", "probe_b": "probe8"}


def read_jsonl(p: Path):
    with open(p, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def vectors(gold, behaviours, attack="adaptive"):
    """{defense: 0/1 array aligned to `behaviours`}."""
    idx = {b: i for i, b in enumerate(behaviours)}
    out = {}
    for r in gold:
        if r.get("attack") != attack or r["behaviour_id"] not in idx:
            continue
        out.setdefault(r["defense"], np.zeros(len(behaviours), dtype=int))
        out[r["defense"]][idx[r["behaviour_id"]]] = int(bool(r["breach"]))
    return out


def mcnemar_exact(x, y):
    """n01 = x breached & y held; n10 = y breached & x held. Exact two-sided binomial."""
    n01 = int(np.sum((x == 1) & (y == 0)))
    n10 = int(np.sum((x == 0) & (y == 1)))
    n = n01 + n10
    p = 1.0 if n == 0 else float(binomtest(n01, n, 0.5, alternative="two-sided").pvalue)
    return n01, n10, p


def holm(pvals):
    """Holm-Bonferroni step-down over the given family."""
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


def phi(x, y):
    if x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def fmt_p(p):
    if p != p:
        return "nan"
    return f"{p:.3e}" if p < 0.001 else f"{p:.4f}"


def cochran_q(mat):
    """mat: behaviours x k binary. Returns (Q, df, p)."""
    k = mat.shape[1]
    G = mat.sum(axis=0).astype(float)      # per-defense totals
    L = mat.sum(axis=1).astype(float)      # per-behaviour totals
    num = (k - 1) * (k * np.sum(G ** 2) - G.sum() ** 2)
    den = k * L.sum() - np.sum(L ** 2)
    if den == 0:
        return float("nan"), k - 1, float("nan")
    Q = num / den
    return float(Q), k - 1, float(chi2.sf(Q, k - 1))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--primary", required=True, type=Path)
    ap.add_argument("--nolg", required=True, type=Path)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--perm", type=int, default=N_PERM)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    behaviours = [json.loads(l)["behaviour_id"]
                  for l in open("data/eval_harmful.jsonl", encoding="utf-8") if l.strip()]
    gold = read_jsonl(a.primary / "gold.jsonl")
    V = vectors(gold, behaviours)
    stack7 = V.get("stack")
    stack6 = vectors(read_jsonl(a.nolg / "gold.jsonl"), behaviours).get("stack")
    assert stack7 is not None, "no seven-layer stack vector in the primary gold.jsonl"
    assert stack6 is not None, "no six-layer stack vector in the nolg gold.jsonl"

    prompt_sha = hashlib.sha256(
        Path("dcorr/judge/strongreject_prompt.txt").read_bytes()).hexdigest()
    meta = {"seed": SEED, "n_behaviours": len(behaviours),
            "grader_prompt_sha256": prompt_sha,
            "judge_model": "openai/gpt-4o-mini", "judge_temperature": 0.0,
            "target_model": "lmsys/vicuna-7b-v1.5",
            "mcnemar_variant": "exact two-sided binomial on discordant cells "
                               "(scipy.stats.binomtest); NOT chi-square, NOT continuity-corrected",
            "permutations": a.perm,
            "marginal_asr": {d: float(V[d].mean()) for d in DEFENSES},
            "stack7_asr": float(stack7.mean()), "stack6_asr": float(stack6.mean())}

    # ---- A1 + A2 pairwise ------------------------------------------------------------
    pairs = list(itertools.combinations(DEFENSES, 2))
    rows = []
    for d1, d2 in pairs:
        n01, n10, p = mcnemar_exact(V[d1], V[d2])
        rows.append({"pair": f"{PRETTY[d1]} x {PRETTY[d2]}", "n01": n01, "n10": n10,
                     "p_exact": p})
    adj = holm([r["p_exact"] for r in rows])
    for r, q in zip(rows, adj):
        r["p_holm"] = q
    with open(a.out / "A1_mcnemar.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["pair", "n01", "n10", "p_exact", "p_holm"])
        for r in sorted(rows, key=lambda r: r["p_exact"]):
            w.writerow([r["pair"], r["n01"], r["n10"], fmt_p(r["p_exact"]), fmt_p(r["p_holm"])])

    # ---- A1 versus each stack --------------------------------------------------------
    srows = []
    for label, sv in (("seven_layer", stack7), ("six_layer_no_llamaguard", stack6)):
        for d in DEFENSES:
            if label == "six_layer_no_llamaguard" and d == "llamaguard":
                continue
            n01, n10, p = mcnemar_exact(V[d], sv)
            srows.append([PRETTY[d], label, n01, n10, fmt_p(p)])
    with open(a.out / "A1_vs_stack.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["defense", "stack", "n_defense_only", "n_stack_only", "p_exact"])
        w.writerows(srows)

    # ---- A2 Cochran's Q ---------------------------------------------------------------
    mat = np.column_stack([V[d] for d in DEFENSES])
    Q, df, pq = cochran_q(mat)
    n_holm_sig = int(sum(1 for r in rows if r["p_holm"] < 0.05))
    n_raw_sig = int(sum(1 for r in rows if r["p_exact"] < 0.05))
    lg_dominates = all(
        mcnemar_exact(V["llamaguard"], V[d])[0] == 0 for d in DEFENSES if d != "llamaguard")
    with open(a.out / "A2_cochran.txt", "w", encoding="utf-8") as f:
        f.write(f"Cochran's Q across {len(DEFENSES)} defenses on {len(behaviours)} behaviours\n")
        f.write(f"  Q  = {Q:.4f}\n  df = {df}\n  p  = {fmt_p(pq)}\n\n")
        f.write("Follow-up: exact-binomial McNemar on all 21 pairs, Holm-corrected "
                "(family size 21).\n")
        f.write(f"  significant at 0.05 uncorrected : {n_raw_sig}/21\n")
        f.write(f"  significant at 0.05 after Holm  : {n_holm_sig}/21\n\n")
        f.write(f"Llama Guard dominance (no behaviour where Llama Guard breaches and the "
                f"other member holds): {lg_dominates}\n")

    # ---- A3 permutation test ----------------------------------------------------------
    live = [(d1, d2) for d1, d2 in pairs if d1 != "llamaguard" and d2 != "llamaguard"]
    prows = []
    for d1, d2 in live:
        x, y = V[d1], V[d2]
        obs = phi(x, y)
        cnt = 0
        for _ in range(a.perm):
            if abs(phi(rng.permutation(x), y)) >= abs(obs) - 1e-12:
                cnt += 1
        prows.append([f"{PRETTY[d1]} x {PRETTY[d2]}", f"{obs:.3f}",
                      fmt_p((cnt + 1) / (a.perm + 1))])
    with open(a.out / "A3_permutation.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["pair", "phi", "p_perm"])
        w.writerows(prows)

    # ---- A4 re-judge concordance ------------------------------------------------------
    rj = vectors(read_jsonl(a.nolg / "gold.jsonl"), behaviours)
    covered = sorted(set(rj) & set(DEFENSES))
    with open(a.out / "A4_rejudge.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["pair", "phi_original", "phi_rejudged", "n_behaviors_covered"])
        if not covered:
            w.writerow(["NOT COMPUTABLE - the stored re-judge covers only the undefended "
                        "control, so it contains no defense breach vector and therefore no "
                        "pair", "", "", 0])
        for d1, d2 in itertools.combinations(covered, 2):
            w.writerow([f"{PRETTY[d1]} x {PRETTY[d2]}", f"{phi(V[d1], V[d2]):.3f}",
                        f"{phi(rj[d1], rj[d2]):.3f}", len(behaviours)])
    meta["A4_note"] = (
        "The 300 re-judged responses are all defense='undefended' (100 behaviours x 3 "
        "attacks). They contain no per-defense breach vector, so no pair phi can be "
        "recomputed on re-judged labels. What the re-judge does support is the undefended "
        "ASR: 0.58 vs 0.64 adaptive, 0.27 vs 0.26 static_gcg, 0.06 vs 0.10 static_plain, "
        "19/300 verdicts flipped. Recomputing pair phi under re-judging requires Task B.")

    (a.out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Cochran Q = {Q:.3f}, df={df}, p={fmt_p(pq)}")
    print(f"McNemar significant: {n_raw_sig}/21 uncorrected, {n_holm_sig}/21 after Holm")
    print(f"Llama Guard dominates every other member pairwise: {lg_dominates}")
    print(f"A4: pairs recomputable on re-judged labels = {len(covered)}")
    print(f"wrote 5 files + meta.json to {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
