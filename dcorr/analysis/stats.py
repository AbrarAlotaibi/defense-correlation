"""Correlation statistics on paired binary breach vectors.

phi (Matthews) coefficient IS the rho that Figure 3(a) is drawn over, because for two
Bernoulli variables
    ASR_{d1 d2} = p1 p2 + rho * sqrt(p1(1-p1) p2(1-p2)).
Bootstrap over behaviours for CIs; McNemar for stack vs best single layer.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import numpy as np


@dataclass
class PairStat:
    d1: str
    d2: str
    same_row: bool
    n: int
    p1: float
    p2: float
    joint: float          # ASR of the intersection (both breached)
    indep: float          # p1 * p2
    excess: float         # joint - indep
    phi: float
    phi_ci: tuple[float, float]
    excess_ci: tuple[float, float]
    phi_p_value: float    # two-sided bootstrap p that phi == 0

    def as_dict(self) -> dict:
        d = asdict(self)
        d["phi_ci"] = list(self.phi_ci)
        d["excess_ci"] = list(self.excess_ci)
        return d


def phi_coefficient(a: np.ndarray, b: np.ndarray) -> float:
    """Matthews / phi coefficient of two binary vectors. 0 when either is constant."""
    a = a.astype(float)
    b = b.astype(float)
    n = len(a)
    if n == 0:
        return 0.0
    n11 = float(np.sum((a == 1) & (b == 1)))
    n10 = float(np.sum((a == 1) & (b == 0)))
    n01 = float(np.sum((a == 0) & (b == 1)))
    n00 = float(np.sum((a == 0) & (b == 0)))
    num = n11 * n00 - n10 * n01
    den = math.sqrt((n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00))
    if den == 0.0:
        return 0.0
    return num / den


def _bootstrap(a: np.ndarray, b: np.ndarray, fn, resamples: int, ci: float,
               rng: np.random.Generator) -> tuple[float, float, np.ndarray]:
    n = len(a)
    stats = np.empty(resamples)
    for i in range(resamples):
        idx = rng.integers(0, n, n)
        stats[i] = fn(a[idx], b[idx])
    lo = float(np.quantile(stats, (1 - ci) / 2))
    hi = float(np.quantile(stats, 1 - (1 - ci) / 2))
    return lo, hi, stats


def pair_stat(d1: str, d2: str, a: np.ndarray, b: np.ndarray, same_row: bool,
              resamples: int = 10000, ci: float = 0.95, seed: int = 0) -> PairStat:
    a = np.asarray(a).astype(int)
    b = np.asarray(b).astype(int)
    n = len(a)
    rng = np.random.default_rng(seed)

    p1 = float(a.mean()) if n else 0.0
    p2 = float(b.mean()) if n else 0.0
    joint = float(((a == 1) & (b == 1)).mean()) if n else 0.0
    indep = p1 * p2
    phi = phi_coefficient(a, b)

    phi_lo, phi_hi, phi_boot = _bootstrap(a, b, phi_coefficient, resamples, ci, rng)
    ex_lo, ex_hi, _ = _bootstrap(
        a, b,
        lambda x, y: float(((x == 1) & (y == 1)).mean()) - float(x.mean()) * float(y.mean()),
        resamples, ci, rng,
    )
    # Two-sided bootstrap p-value that phi == 0.
    frac_le0 = float(np.mean(phi_boot <= 0))
    frac_ge0 = float(np.mean(phi_boot >= 0))
    p_value = min(1.0, 2 * min(frac_le0, frac_ge0))

    return PairStat(
        d1=d1, d2=d2, same_row=same_row, n=n,
        p1=p1, p2=p2, joint=joint, indep=indep, excess=joint - indep,
        phi=phi, phi_ci=(phi_lo, phi_hi), excess_ci=(ex_lo, ex_hi),
        phi_p_value=p_value,
    )


def mcnemar(a: np.ndarray, b: np.ndarray) -> dict:
    """Exact McNemar on paired binaries. b01/b10 are the discordant counts.

    Here a = stack breach, b = best-single-layer breach; a significant result with
    b01 > b10 means the stack is breached on strictly fewer behaviours than the best
    single layer, i.e. stacking helped.
    """
    a = np.asarray(a).astype(int)
    b = np.asarray(b).astype(int)
    b01 = int(np.sum((a == 0) & (b == 1)))   # single breached, stack held
    b10 = int(np.sum((a == 1) & (b == 0)))   # stack breached, single held
    n = b01 + b10
    if n == 0:
        return {"b01": 0, "b10": 0, "p_value": 1.0, "stat": 0.0}
    from scipy.stats import binomtest

    p = binomtest(min(b01, b10), n, 0.5, alternative="two-sided").pvalue
    return {"b01": b01, "b10": b10, "p_value": float(p),
            "stat": (abs(b01 - b10) - 1) ** 2 / n if n else 0.0}


def benjamini_hochberg(pvals: list[float]) -> list[float]:
    """BH-adjusted q-values, preserving input order."""
    n = len(pvals)
    if n == 0:
        return []
    order = np.argsort(pvals)
    ranked = np.asarray(pvals)[order]
    q = ranked * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(q, 0, 1)
    return [float(x) for x in out]


def phi_from_rates(p1: float, p2: float, rho: float) -> float:
    """The analytic joint-breach curve Figure 3(a) plots the measured points over."""
    return p1 * p2 + rho * math.sqrt(max(0.0, p1 * (1 - p1) * p2 * (1 - p2)))
