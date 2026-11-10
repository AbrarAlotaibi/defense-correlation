"""Tests for dcorr/analysis/estimands.py against hand-computed values (CPU only)."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dcorr.analysis.estimands import (  # noqa: E402
    Vectors, bootstrap_ci, difficulty_from_undefended, e1_pairs, e2_pairs, e3_stack,
    exact_mcnemar, marginals, null_from_undefended, partial_corr, phi, reliability,
    transfer_matrix, wilson,
)


def test_phi_known_values():
    assert phi([1, 1, 0, 0], [1, 1, 0, 0]) == pytest.approx(1.0)
    assert phi([1, 1, 0, 0], [0, 0, 1, 1]) == pytest.approx(-1.0)
    assert phi([1, 1, 0, 0], [1, 0, 1, 0]) == pytest.approx(0.0)
    # 2x2 table n11=3 n10=1 n01=1 n00=3 -> phi = (9-1)/sqrt(4*4*4*4) = 0.5
    assert phi([1, 1, 1, 1, 0, 0, 0, 0], [1, 1, 1, 0, 1, 0, 0, 0]) == pytest.approx(0.5)
    assert math.isnan(phi([1, 1, 1], [0, 1, 0]))


def test_wilson_matches_reference():
    lo, hi = wilson(1, 100)                   # standard reference: 0.0018, 0.0545
    assert lo == pytest.approx(0.00177, abs=1e-4) and hi == pytest.approx(0.0545, abs=1e-4)
    assert wilson(0, 10)[0] == 0.0


def test_exact_mcnemar():
    r = exact_mcnemar([1, 0, 0, 0], [0, 0, 0, 0])            # one discordant pair
    assert (r["n10"], r["n01"], r["p"]) == (1, 0, 1.0)
    r = exact_mcnemar([1] * 10 + [0] * 10, [0] * 20)          # 10 vs 0 -> p = 2/1024
    assert r["p"] == pytest.approx(2 / 1024)


def test_partial_corr_removes_a_common_cause():
    rng = np.random.default_rng(0)
    z = rng.normal(size=5000)
    a = z + rng.normal(size=5000)
    b = z + rng.normal(size=5000)
    assert np.corrcoef(a, b)[0, 1] == pytest.approx(0.5, abs=0.05)
    assert partial_corr(a, b, z) == pytest.approx(0.0, abs=0.05)


def _V():
    # 8 behaviours, 2 seeds. "undefended" fails on the same 4 easy behaviours in both seeds
    # (reliability 1); d1 equals undefended; d2 is unrelated to it.
    u = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    d2 = np.array([1, 0, 1, 0, 1, 0, 1, 0])
    return Vectors(behaviours=[f"b{i}" for i in range(8)],
                   vec={0: {"undefended": u, "d1": u.copy(), "d2": d2},
                        1: {"undefended": u, "d1": u.copy(), "d2": 1 - d2}})


def test_reliability_and_null():
    V = _V()
    rel = reliability(V)
    assert rel["undefended"]["mean"] == pytest.approx(1.0)
    assert rel["d2"]["mean"] == pytest.approx(-1.0)          # seed 1 is seed 0 flipped
    assert null_from_undefended(V)["mean"] == pytest.approx(1.0)


def test_e1_and_disattenuation():
    out = e1_pairs(V := _V(), resamples=200)
    assert out["d1|d2"]["per_seed"][0]["phi"] == pytest.approx(0.0)
    assert out["d1|d2"]["mean_phi"] == pytest.approx(0.0)
    assert math.isnan(out["d1|d2"]["disattenuated"])        # r_d2 < 0 -> undefined
    assert "undefended" not in " ".join(out)                  # excluded from pairs


def test_partial_given_undefended_and_difficulty():
    V = _V()
    diff = difficulty_from_undefended(V)
    assert list(diff) == [1, 1, 1, 1, 0, 0, 0, 0]


def test_marginals_and_e3():
    V = _V()
    m = marginals(V)
    assert m["d1"]["mean_asr"] == pytest.approx(0.5)
    stack = {0: np.array([1, 0, 0, 0, 0, 0, 0, 0]), 1: np.zeros(8, dtype=int)}
    e3 = e3_stack(V, stack, ["d1", "d2"])
    assert e3[0]["intersection_asr"] == pytest.approx(2 / 8)   # d1 & d2 both 1 on b0, b2
    assert e3[0]["stack_asr"] == pytest.approx(1 / 8)
    assert e3[1]["stack_asr"] == 0.0


def test_e2_and_transfer():
    cross = {"srcA": {"d1": np.array([1, 1, 0, 0]), "d2": np.array([1, 1, 0, 0])},
             "srcB": {"d1": np.array([1, 0, 1, 0]), "d2": np.array([0, 1, 0, 1])}}
    e2 = e2_pairs(cross)
    assert e2["d1|d2"]["per_source"] == {"srcA": pytest.approx(1.0), "srcB": pytest.approx(-1.0)}
    assert e2["d1|d2"]["mean_phi"] == pytest.approx(0.0)
    assert transfer_matrix(cross)["srcA"]["d1"] == 0.5


def test_bootstrap_ci_brackets_point_estimate():
    rng = np.random.default_rng(3)
    a = (rng.random(200) < 0.5).astype(int)
    b = np.where(rng.random(200) < 0.8, a, 1 - a)
    lo, hi = bootstrap_ci(phi, a, b, resamples=500)
    assert lo < phi(a, b) < hi


def test_table2x2():
    from dcorr.analysis.estimands import table2x2
    assert table2x2([1, 1, 0, 0], [1, 0, 1, 0]) == {"n11": 1, "n10": 1, "n01": 1, "n00": 1}


def test_nonpairwise_extremes():
    from dcorr.analysis.estimands import nonpairwise
    same = np.array([[1, 1, 0, 0], [1, 1, 0, 0], [1, 1, 0, 0]])   # identical members
    r = nonpairwise(same)
    assert r["entropy"] == pytest.approx(0.0) and r["kw_variance"] == pytest.approx(0.0)
    assert r["kappa"] == pytest.approx(1.0)
    assert r["theta"] == pytest.approx(0.25)          # items all-correct or all-wrong
    # two members always disagreeing: maximal entropy, kappa -1
    r = nonpairwise(np.array([[1, 0, 1, 0], [0, 1, 0, 1]]))
    assert r["entropy"] == pytest.approx(1.0) and r["kappa"] == pytest.approx(-1.0)
    assert r["theta"] == pytest.approx(0.0)


def test_auroc_and_best_of():
    from dcorr.analysis.estimands import auroc, best_of_strategies
    assert auroc([0.9, 0.8, 0.1, 0.2], [1, 1, 0, 0]) == pytest.approx(1.0)
    assert auroc([0.5, 0.5], [1, 0]) == pytest.approx(0.5)
    out = best_of_strategies({"a": np.array([1, 0, 0]), "b": np.array([0, 0, 1])})
    assert list(out) == [1, 0, 1]


def test_heldout_selection_prefers_complementary_members():
    from dcorr.analysis.estimands import heldout_selection
    n = 40
    a = np.array([1] * (n // 2) + [0] * (n // 2))     # fails on the first half
    b = a.copy()                                        # redundant with a (same row)
    c = 1 - a                                           # complementary (another row)
    out = heldout_selection({"a": a, "b": b, "c": c}, {"a": "X", "b": "X", "c": "Y"},
                            k_max=2, splits=50)
    assert out["greedy"][2] == pytest.approx(0.0)       # picks a (or b) + c
    assert out["row"][2] == pytest.approx(0.0)
    assert out["random"][2] == pytest.approx(1 / 6, abs=0.03)   # {a,b} leaves 0.5
    assert out["single_best"] == pytest.approx(0.5, abs=0.1)
