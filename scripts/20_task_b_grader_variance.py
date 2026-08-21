"""Task B analysis: majority-vote labels and a grader-only interval.

Consumes results/<run>/gold_multi.jsonl (scripts/19_rejudge_multi.py), which holds REPS
independent judgments of every response.

B3  Table 10 rebuilt on majority-vote labels: p1, p2, joint, Delta, phi, phi/phi_max,
    behaviour-bootstrap CI, BH q.
B4  Grader-only interval: resample ONE judgment per response 1,000 times and recompute phi
    and Delta each time. This is the interval the behaviour bootstrap does not provide -- it
    holds the behaviour set fixed and varies only which judgment you happened to get.
B5  Per-response agreement across the repetitions.

The two intervals answer different questions and are reported side by side rather than
combined: B3's CI covers "a different sample of behaviours", B4's covers "a different draw
from the grader".

Multiplicity: BH over the 15 measurable pairs (Llama Guard excluded), matching the family the
manuscript uses.

Usage:
  python scripts/20_task_b_grader_variance.py --run results/hpc_vicuna_autodan \
      --out fusion/taskB
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

SEED = 20260727
N_BOOT = 10000
N_GRADER = 1000
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


def phi_max(p, q):
    """Largest phi attainable given the two marginals."""
    if min(p, q) <= 0 or max(p, q) >= 1:
        return float("nan")
    return (min(p, q) - p * q) / np.sqrt(p * q * (1 - p) * (1 - q))


def bh(pvals):
    p = np.asarray(pvals, float)
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


def fmt_p(p):
    if p != p:
        return "nan"
    return f"{p:.3e}" if p < 0.001 else f"{p:.4f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--attack", default="adaptive")
    ap.add_argument("--grader-resamples", type=int, default=N_GRADER)
    ap.add_argument("--boot", type=int, default=N_BOOT)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    rows = read_jsonl(a.run / "gold_multi.jsonl")
    behaviours = [json.loads(l)["behaviour_id"]
                  for l in open("data/eval_harmful.jsonl", encoding="utf-8") if l.strip()]
    bidx = {b: i for i, b in enumerate(behaviours)}

    # response -> list of judgments (blocked rows contribute a single fixed False)
    judg: dict[tuple, list[int]] = defaultdict(list)
    blocked: set[tuple] = set()
    for r in rows:
        key = (r["defense"], r["behaviour_id"], r["attack"])
        if r["behaviour_id"] not in bidx:
            continue
        if r.get("blocked"):
            blocked.add(key)
        elif r.get("judged"):
            judg[key].append(int(bool(r["breach"])))

    reps = sorted({len(v) for v in judg.values()})
    print(f"[B] {len(judg)} judged responses, {len(blocked)} blocked; "
          f"judgments per response: {reps}")

    # ---- B5 agreement -----------------------------------------------------------------
    adaptive_only = {k: v for k, v in judg.items() if k[2] == a.attack}
    n_full = [v for v in judg.values() if len(v) >= 3]
    unanimous = sum(1 for v in n_full if len(set(v)) == 1)
    split = len(n_full) - unanimous
    maj32 = sum(1 for v in n_full if 0 < sum(v) < len(v) and
                max(sum(v), len(v) - sum(v)) == 3 and len(v) == 5)
    with open(a.out / "B5_agreement.txt", "w", encoding="utf-8") as f:
        f.write(f"n_responses          {len(n_full)}\n")
        f.write(f"pct_unanimous        {100 * unanimous / max(1, len(n_full)):.2f}\n")
        f.write(f"pct_split            {100 * split / max(1, len(n_full)):.2f}\n")
        f.write(f"pct_3_2              {100 * maj32 / max(1, len(n_full)):.2f}\n")
        f.write(f"\njudgments per response: {reps}\n")
        f.write("Blocked responses are excluded: they are never sent to the grader, so they "
                "carry no grader variance.\n")

    def vec(pick) -> dict[str, np.ndarray]:
        """pick(list)->0/1. Build a defense->vector map for the chosen attack."""
        out = {d: np.zeros(len(behaviours), dtype=int) for d in DEFENSES}
        for (d, b, atk), v in judg.items():
            if atk != a.attack or d not in out:
                continue
            out[d][bidx[b]] = pick(v)
        return out   # blocked keys stay 0, which is their fixed verdict

    majority = vec(lambda v: int(sum(v) * 2 > len(v)))

    # ---- marginals under majority vote -------------------------------------------------
    marg = {d: float(majority[d].mean()) for d in DEFENSES}
    und = {}
    for atk in ("adaptive", "static_plain", "static_gcg"):
        vals = [v for (d, b, k), v in judg.items() if d == "undefended" and k == atk]
        if vals:
            und[atk] = round(sum(int(sum(v) * 2 > len(v)) for v in vals) / len(vals), 3)

    # ---- B3 table 10 on majority labels ------------------------------------------------
    live = [d for d in DEFENSES if d != "llamaguard"]
    pairs = list(itertools.combinations(live, 2))
    recs = []
    for d1, d2 in pairs:
        x, y = majority[d1], majority[d2]
        p1, p2 = float(x.mean()), float(y.mean())
        joint = float((x & y).mean())
        obs = phi(x, y)
        bs = np.empty(a.boot)
        for i in range(a.boot):
            s = rng.integers(0, len(x), len(x))
            bs[i] = phi(x[s], y[s])
        bs = bs[~np.isnan(bs)]
        # permutation p-value for BH, distribution-free and consistent with Task A
        cnt = sum(1 for _ in range(2000)
                  if abs(phi(rng.permutation(x), y)) >= abs(obs) - 1e-12)
        recs.append({"pair": f"{PRETTY[d1]} x {PRETTY[d2]}", "p1": p1, "p2": p2,
                     "joint": joint, "Delta": joint - p1 * p2, "phi": obs,
                     "phi_max": phi_max(p1, p2),
                     "phi_lo": float(np.quantile(bs, .025)),
                     "phi_hi": float(np.quantile(bs, .975)),
                     "p_perm": (cnt + 1) / 2001})
    qs = bh([r["p_perm"] for r in recs])
    with open(a.out / "B3_table10_majority.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["pair", "p1", "p2", "joint", "Delta", "phi", "phi_over_phi_max",
                    "phi_lo", "phi_hi", "q"])
        for r, q in zip(recs, qs):
            w.writerow([r["pair"], f"{r['p1']:.3f}", f"{r['p2']:.3f}", f"{r['joint']:.3f}",
                        f"{r['Delta']:.3f}", f"{r['phi']:.3f}",
                        f"{r['phi'] / r['phi_max']:.3f}" if r["phi_max"] == r["phi_max"] else "nan",
                        f"{r['phi_lo']:.3f}", f"{r['phi_hi']:.3f}", fmt_p(q)])

    # ---- B4 grader-only interval -------------------------------------------------------
    keys = sorted(k for k in judg if k[2] == a.attack)
    grader = {p: {"phi": [], "Delta": []} for p in pairs}
    for _ in range(a.grader_resamples):
        draw = {d: np.zeros(len(behaviours), dtype=int) for d in DEFENSES}
        for (d, b, _atk) in keys:
            if d not in draw:
                continue
            v = judg[(d, b, a.attack)]
            draw[d][bidx[b]] = v[rng.integers(0, len(v))]
        for (d1, d2) in pairs:
            x, y = draw[d1], draw[d2]
            grader[(d1, d2)]["phi"].append(phi(x, y))
            grader[(d1, d2)]["Delta"].append(
                float((x & y).mean()) - float(x.mean()) * float(y.mean()))
    with open(a.out / "B4_grader_intervals.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["pair", "phi_majority", "phi_grader_lo", "phi_grader_hi",
                    "Delta_majority", "Delta_grader_lo", "Delta_grader_hi"])
        for r, (d1, d2) in zip(recs, pairs):
            ph = np.array([v for v in grader[(d1, d2)]["phi"] if v == v])
            dl = np.array(grader[(d1, d2)]["Delta"])
            w.writerow([r["pair"], f"{r['phi']:.3f}",
                        f"{np.quantile(ph, .025):.3f}", f"{np.quantile(ph, .975):.3f}",
                        f"{r['Delta']:.3f}",
                        f"{np.quantile(dl, .025):.3f}", f"{np.quantile(dl, .975):.3f}"])

    meta = {"seed": SEED, "reps_per_response": reps,
            "grader_resamples": a.grader_resamples, "behaviour_bootstrap": a.boot,
            "bh_family_size": len(pairs),
            "grader_prompt_sha256": hashlib.sha256(
                Path("dcorr/judge/strongreject_prompt.txt").read_bytes()).hexdigest(),
            "judge_model": "openai/gpt-4o-mini", "judge_temperature": 0.0,
            "marginal_asr_majority": {PRETTY[d]: round(v, 3) for d, v in marg.items()},
            "undefended_asr_majority": und}
    (a.out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"[B5] unanimous {100 * unanimous / max(1, len(n_full)):.1f}%, "
          f"3-2 splits {100 * maj32 / max(1, len(n_full)):.1f}%")
    print(f"[B3] marginals under majority vote: "
          f"{ {PRETTY[d]: round(v, 3) for d, v in marg.items()} }")
    print(f"[B3] undefended under majority vote: {und}")
    print(f"wrote 4 files to {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
