"""How much of the defense-pair association is behaviour difficulty? (REVIEW R5-7, R2-3, R7-2)

Model (random-intercept logistic, fitted by variational Bayes over every defense and seed):

    logit P(breach_{d,i,s}) = alpha_d + u_i,      u_i ~ N(0, sigma_b^2)

u_i is behaviour i's latent difficulty, shared by every defense and every independent
search. Unlike the v1 CMH stratifier, it is estimated from the whole design rather than by
conditioning on other defenses' outcomes. Under this model two defenses fail INDEPENDENTLY
given the behaviour, so the phi it predicts for a pair is the association that difficulty
alone produces. The excess of the observed phi over that prediction is the part difficulty
does not explain - the quantity the "shared mechanism" reading needs.

  icc              sigma_b^2 / (sigma_b^2 + pi^2/3): share of latent variance from behaviours
  predicted_phi    phi implied by the fitted model with conditional independence (population
                   level: integrated over u ~ N(0, sigma_b^2) by Gauss-Hermite quadrature)
  excess           observed mean phi - predicted_phi, per pair
"""
from __future__ import annotations

import itertools
import math

import numpy as np

from .estimands import Vectors, phi


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def fit_difficulty_model(V: Vectors, exclude: tuple[str, ...] = ("stack",)) -> dict:
    import pandas as pd
    from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

    defs = [d for d in V.defenses() if d not in exclude]
    rows = [{"y": int(V.vec[s][d][k]), "defense": d, "behaviour": b}
            for s in V.seeds for d in defs for k, b in enumerate(V.behaviours)]
    df = pd.DataFrame(rows)
    model = BinomialBayesMixedGLM.from_formula(
        "y ~ 0 + C(defense)", {"behaviour": "0 + C(behaviour)"}, df)
    res = model.fit_vb()
    names = model.exog_names
    alpha = {n[len("C(defense)["):-1]: float(m) for n, m in zip(names, res.fe_mean)}
    sd_b = float(np.exp(res.vcp_mean[0]))
    return {"alpha": alpha, "sd_behaviour": sd_b,
            "icc": sd_b ** 2 / (sd_b ** 2 + math.pi ** 2 / 3),
            "n_obs": len(df), "defenses": defs}


def predicted_phi(alpha_a: float, alpha_b: float, sd_b: float, nodes: int = 64) -> float:
    """Population phi of two defenses that fail independently given a N(0, sd_b^2)
    behaviour effect: E[p_a p_b] - E[p_a]E[p_b] over u, normalised by the marginal SDs."""
    x, w = np.polynomial.hermite_e.hermegauss(nodes)   # probabilists' Hermite: weight e^{-x^2/2}
    w = w / w.sum()
    u = sd_b * x
    pa, pb = _sigmoid(alpha_a + u), _sigmoid(alpha_b + u)
    ea, eb, eab = (w * pa).sum(), (w * pb).sum(), (w * pa * pb).sum()
    den = math.sqrt(ea * (1 - ea) * eb * (1 - eb))
    return float((eab - ea * eb) / den) if den > 0 else float("nan")


def difficulty_decomposition(V: Vectors, exclude_pairs: tuple[str, ...] = ("undefended", "stack")) -> dict:
    fit = fit_difficulty_model(V)
    defs = [d for d in fit["defenses"] if d not in exclude_pairs]
    pairs = {}
    for a, b in itertools.combinations(defs, 2):
        obs = [phi(V.vec[s][a], V.vec[s][b]) for s in V.seeds]
        obs = [o for o in obs if not math.isnan(o)]
        mean = float(np.mean(obs)) if obs else float("nan")
        pred = predicted_phi(fit["alpha"][a], fit["alpha"][b], fit["sd_behaviour"])
        pairs[f"{a}|{b}"] = {"observed": mean, "predicted_difficulty_only": pred,
                             "excess": mean - pred if not math.isnan(mean) else float("nan")}
    return {"fit": fit, "pairs": pairs}
