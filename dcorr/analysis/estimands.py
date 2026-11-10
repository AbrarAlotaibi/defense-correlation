"""v2 analysis: the three estimands, the seed-level controls, and their uncertainty.

Answers REVIEW R2-1/3, R5-1/4/7 and F2/F3 with the replicated v2 data:

  E1  behaviour-level co-vulnerability: phi between two defenses' breach vectors from
      separate (independently seeded) searches, per seed and summarised across seeds;
  E2  prompt-level co-failure: phi between two defenses evaluated on the SAME prompts
      (scripts/24_cross_eval.py), per source search;
  E3  the assembled stack: its residual against the per-layer intersection and against its
      strongest single layer.

Controls, all from data the v2 grid produces:
  * reliability r_d - phi between a defense's breach vectors from two independent seeds of
    its own search (test-retest). For the undefended model this is the NULL a defense pair
    must exceed before "co-vulnerability" means more than attack reproducibility;
  * disattenuated phi = phi_12 / sqrt(r_1 r_2) (classical correction for unreliability);
  * partial phi given the undefended model's breach rate across seeds - a difficulty proxy
    measured independently of every defended outcome (R2-3, R7-2).

All functions are pure (numpy in, numbers out) so they can be tested on hand-built vectors.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np


# ----------------------------------------------------------------------------- basics
def phi(a, b) -> float:
    """Matthews/phi coefficient of two binary vectors; nan if either is constant."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape or a.size == 0:
        raise ValueError("phi needs two non-empty vectors of equal length")
    pa, pb = a.mean(), b.mean()
    den = math.sqrt(pa * (1 - pa) * pb * (1 - pb))
    if den == 0:
        return float("nan")
    return float(((a * b).mean() - pa * pb) / den)


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """Wilson score interval for k successes in n trials."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def bootstrap_ci(fn, *vectors, resamples: int = 2000, ci: float = 0.95,
                 seed: int = 0) -> tuple[float, float]:
    """Percentile CI of fn(*vectors) over resampled behaviours (rows)."""
    vs = [np.asarray(v) for v in vectors]
    n = len(vs[0])
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(resamples):
        idx = rng.integers(0, n, n)
        s = fn(*(v[idx] for v in vs))
        if not math.isnan(s):
            stats.append(s)
    if not stats:
        return (float("nan"), float("nan"))
    lo, hi = np.percentile(stats, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100])
    return (float(lo), float(hi))


def partial_corr(a, b, z) -> float:
    """Pearson partial correlation of a and b given z (z may be non-binary)."""
    r_ab, r_az, r_bz = (np.corrcoef(x, y)[0, 1] for x, y in ((a, b), (a, z), (b, z)))
    if any(math.isnan(r) for r in (r_ab, r_az, r_bz)):
        return float("nan")
    den = math.sqrt((1 - r_az ** 2) * (1 - r_bz ** 2))
    return float((r_ab - r_az * r_bz) / den) if den > 0 else float("nan")


def exact_mcnemar(a, b) -> dict:
    """Exact (binomial) McNemar test on paired binary outcomes, two-sided."""
    a = np.asarray(a, dtype=int)
    b = np.asarray(b, dtype=int)
    n01 = int(((a == 0) & (b == 1)).sum())
    n10 = int(((a == 1) & (b == 0)).sum())
    n = n01 + n10
    if n == 0:
        return {"n01": 0, "n10": 0, "p": 1.0}
    k = min(n01, n10)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return {"n01": n01, "n10": n10, "p": min(1.0, 2 * p)}


# ----------------------------------------------------------------------------- estimands
@dataclass
class Vectors:
    """Breach vectors on a common, ordered behaviour list: vec[seed][defense] -> np.array."""
    behaviours: list[str]
    vec: dict[int, dict[str, np.ndarray]]

    @property
    def seeds(self) -> list[int]:
        return sorted(self.vec)

    def defenses(self) -> list[str]:
        common = set.intersection(*(set(v) for v in self.vec.values()))
        return sorted(common)


def marginals(V: Vectors) -> dict:
    out = {}
    for d in V.defenses():
        per = {}
        for s in V.seeds:
            v = V.vec[s][d]
            per[s] = {"asr": float(v.mean()), "ci": wilson(int(v.sum()), len(v))}
        pooled = np.concatenate([V.vec[s][d] for s in V.seeds])
        out[d] = {"per_seed": per, "mean_asr": float(pooled.mean()),
                  "pooled_ci": wilson(int(pooled.sum()), len(pooled))}
    return out


def reliability(V: Vectors) -> dict[str, dict]:
    """Test-retest phi of each defense across independent seeds (needs >= 2 seeds)."""
    out = {}
    for d in V.defenses():
        vals = [phi(V.vec[i][d], V.vec[j][d]) for i, j in itertools.combinations(V.seeds, 2)]
        finite = [x for x in vals if not math.isnan(x)]
        out[d] = {"pairs": vals, "mean": float(np.mean(finite)) if finite else float("nan")}
    return out


def e1_pairs(V: Vectors, exclude: tuple[str, ...] = ("undefended",), resamples: int = 2000) -> dict:
    """Behaviour-level phi for every defense pair: per seed (with a bootstrap CI), mean
    across seeds, and disattenuated by each defense's test-retest reliability."""
    rel = reliability(V) if len(V.seeds) >= 2 else {}
    defs = [d for d in V.defenses() if d not in exclude]
    out = {}
    for a, b in itertools.combinations(defs, 2):
        per = {}
        for s in V.seeds:
            x, y = V.vec[s][a], V.vec[s][b]
            per[s] = {"phi": phi(x, y), "ci": bootstrap_ci(phi, x, y, resamples=resamples, seed=s)}
        finite = [p["phi"] for p in per.values() if not math.isnan(p["phi"])]
        mean = float(np.mean(finite)) if finite else float("nan")
        ra, rb = rel.get(a, {}).get("mean", float("nan")), rel.get(b, {}).get("mean", float("nan"))
        dis = mean / math.sqrt(ra * rb) if ra > 0 and rb > 0 else float("nan")
        # Disattenuation divides by sqrt(ra*rb) and is unbounded, so when the observed phi is
        # already as large as the reliabilities permit the quotient exceeds 1, which is
        # inadmissible for a correlation. That is not a correlation above unity, it is the
        # correction reaching the limit of what this measurement can support. Report the raw
        # quotient for transparency and flag it, rather than capping it and hiding the fact.
        out[f"{a}|{b}"] = {"per_seed": per, "mean_phi": mean, "disattenuated": dis,
                           "disattenuation_saturated": bool(dis == dis and dis > 1.0),
                           "reliabilities": {a: ra, b: rb}}
    return out


def null_from_undefended(V: Vectors) -> dict:
    """phi between independent undefended searches: the level any pair must exceed."""
    if "undefended" not in V.defenses() or len(V.seeds) < 2:
        return {"available": False}
    r = reliability(V)["undefended"]
    return {"available": True, "pairs": r["pairs"], "mean": r["mean"]}


def difficulty_from_undefended(V: Vectors) -> np.ndarray:
    """Per-behaviour undefended breach rate across seeds: a difficulty proxy measured
    without any defended outcome."""
    return np.mean([V.vec[s]["undefended"] for s in V.seeds], axis=0)


def partial_given_undefended(V: Vectors, exclude: tuple[str, ...] = ("undefended",)) -> dict:
    diff = difficulty_from_undefended(V)
    defs = [d for d in V.defenses() if d not in exclude]
    out = {}
    for a, b in itertools.combinations(defs, 2):
        vals = [partial_corr(V.vec[s][a], V.vec[s][b], diff) for s in V.seeds]
        finite = [x for x in vals if not math.isnan(x)]
        out[f"{a}|{b}"] = {"per_seed": vals,
                           "mean": float(np.mean(finite)) if finite else float("nan")}
    return out


def e2_pairs(cross: dict[str, dict[str, np.ndarray]], exclude=("undefended",)) -> dict:
    """Prompt-level phi: cross[source][defense] is the breach vector of `defense` evaluated
    on `source`'s final prompts. Reported per source and averaged over sources."""
    out = {}
    defs = sorted(set.intersection(*(set(v) for v in cross.values())) - set(exclude))
    for a, b in itertools.combinations(defs, 2):
        per = {src: phi(cross[src][a], cross[src][b]) for src in sorted(cross)}
        finite = [x for x in per.values() if not math.isnan(x)]
        out[f"{a}|{b}"] = {"per_source": per,
                           "mean_phi": float(np.mean(finite)) if finite else float("nan")}
    return out


def transfer_matrix(cross: dict[str, dict[str, np.ndarray]]) -> dict[str, dict[str, float]]:
    """ASR of each defense (columns) on each source search's prompts (rows)."""
    return {src: {d: float(v.mean()) for d, v in sorted(row.items())}
            for src, row in sorted(cross.items())}


def e3_stack(V: Vectors, stack: dict[int, np.ndarray], layers: list[str]) -> dict:
    """Stack residual per seed vs. the per-layer intersection and the strongest layer.

    `layers` must name members present in V. A run that attacked only the assembled stack --
    v2_vicuna_nolg is one, since it exists to add a configuration rather than re-measure the
    members -- has no member vectors of its own, and its intersection has to be computed from
    the run that does. Returning empty rather than raising on that case, because the caller is
    better placed to say which run the members should come from; min() over an empty layer
    list used to raise ValueError from three frames down.
    """
    out = {}
    if not layers:
        return out
    for s in V.seeds:
        if s not in stack:
            continue
        st = stack[s]
        inter = np.prod([V.vec[s][d] for d in layers], axis=0)
        best = min(layers, key=lambda d: V.vec[s][d].mean())
        out[s] = {
            "stack_asr": float(st.mean()), "stack_ci": wilson(int(st.sum()), len(st)),
            "intersection_asr": float(inter.mean()),
            "strongest_layer": best, "strongest_asr": float(V.vec[s][best].mean()),
            "stack_vs_strongest": exact_mcnemar(st, V.vec[s][best]),
            "stack_vs_intersection": exact_mcnemar(st, inter),
        }
    return out


# ----------------------------------------------------------------------------- extras
def table2x2(a, b) -> dict[str, int]:
    """Raw contingency table of two breach vectors (REVIEW R5-m3)."""
    a = np.asarray(a, dtype=int)
    b = np.asarray(b, dtype=int)
    return {"n11": int(((a == 1) & (b == 1)).sum()), "n10": int(((a == 1) & (b == 0)).sum()),
            "n01": int(((a == 0) & (b == 1)).sum()), "n00": int(((a == 0) & (b == 0)).sum())}


def nonpairwise(breach: np.ndarray) -> dict[str, float]:
    """Kuncheva & Whitaker (2003) non-pairwise diversity of L members over N items.

    `breach` is L x N with 1 = member fails (is breached) on the item. Computed on
    correctness y = 1 - breach, as in the source:
      theta  measure of difficulty: variance over items of the fraction of members correct
             - large when items are uniformly easy or hard for everyone (a common cause);
      E      entropy: 1 when members split as evenly as possible, 0 when unanimous;
      KW     Kohavi-Wolpert variance; kappa  interrater agreement (1 = members identical).
    """
    y = 1 - np.asarray(breach, dtype=int)
    L, N = y.shape
    if L < 2 or N == 0:
        raise ValueError("needs >= 2 members and >= 1 item")
    lj = y.sum(axis=0)                                   # members correct per item
    theta = float(np.var(lj / L))
    E = float(np.mean(np.minimum(lj, L - lj) / (L - math.ceil(L / 2))))
    KW = float(np.sum(lj * (L - lj)) / (N * L * L))
    p = y.mean()
    kappa = float(1 - (np.sum(lj * (L - lj)) / L) / (N * (L - 1) * p * (1 - p))) \
        if 0 < p < 1 else float("nan")
    return {"theta": theta, "entropy": E, "kw_variance": KW, "kappa": kappa}


def auroc(scores, labels) -> float:
    """Probability a random positive outscores a random negative (ties count half)."""
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=int)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    greater = (pos[:, None] > neg[None, :]).sum()
    ties = (pos[:, None] == neg[None, :]).sum()
    return float((greater + 0.5 * ties) / (len(pos) * len(neg)))


def best_of_strategies(by_attack: dict[str, np.ndarray]) -> np.ndarray:
    """A breach counts if ANY attack strategy breached the behaviour (REVIEW R5-6, R8)."""
    return np.max(np.stack(list(by_attack.values())), axis=0)


def heldout_selection(vec: dict[str, np.ndarray], rows: dict[str, str], k_max: int = 4,
                      splits: int = 500, seed: int = 0) -> dict:
    """Member selection validated on held-out behaviours (REVIEW R5-8, R2-4).

    For each random 50/50 split of behaviours, members are chosen on the TRAIN half by
      greedy   - start from the lowest-ASR member, then add whichever member minimises the
                 joint breach (double-fault) of the growing set;
      row      - one member per dependency row (lowest train ASR within the row), rows added
                 in order of that member's ASR (the paper's composition rule 2);
      single   - the lowest-ASR member alone;
    and each chosen set is scored on the TEST half by its intersection residual (a behaviour
    breaches the set only if it breaches every member). `random` is the mean over all
    k-subsets. Intersection residual is not an end-to-end attack on the assembled stack; it
    is the quantity these selection rules are built to minimise.
    """
    names = sorted(vec)
    X = np.stack([np.asarray(vec[n], dtype=int) for n in names])
    n = X.shape[1]
    rng = np.random.default_rng(seed)
    acc = {m: {k: [] for k in range(1, k_max + 1)} for m in ("greedy", "row", "random")}
    single = []
    for _ in range(splits):
        perm = rng.permutation(n)
        tr, te = perm[: n // 2], perm[n // 2:]
        asr = X[:, tr].mean(axis=1)
        # greedy on double-fault
        chosen = [int(np.argmin(asr))]
        cur = X[chosen[0], tr].copy()
        while len(chosen) < min(k_max, len(names)):
            cand = [i for i in range(len(names)) if i not in chosen]
            best = min(cand, key=lambda i: ((cur & X[i, tr]).mean(), asr[i]))
            chosen.append(best)
            cur = cur & X[best, tr]
        for k in range(1, len(chosen) + 1):
            acc["greedy"][k].append(X[chosen[:k]][:, te].prod(axis=0).mean())
        # one per row
        by_row: dict[str, int] = {}
        for i, nm in enumerate(names):
            r = rows.get(nm, nm)
            if r not in by_row or asr[i] < asr[by_row[r]]:
                by_row[r] = i
        order = sorted(by_row.values(), key=lambda i: asr[i])
        for k in range(1, min(k_max, len(order)) + 1):
            acc["row"][k].append(X[order[:k]][:, te].prod(axis=0).mean())
        # random subsets (exact mean over all k-subsets)
        for k in range(1, min(k_max, len(names)) + 1):
            vals = [X[list(c)][:, te].prod(axis=0).mean()
                    for c in itertools.combinations(range(len(names)), k)]
            acc["random"][k].append(float(np.mean(vals)))
        single.append(X[int(np.argmin(asr)), te].mean())
    summary = {m: {k: float(np.mean(v)) for k, v in ks.items() if v} for m, ks in acc.items()}
    summary["single_best"] = float(np.mean(single))
    summary["splits"] = splits
    return summary
