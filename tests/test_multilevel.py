"""Validate the difficulty decomposition on simulated data where the truth is known."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dcorr.analysis.estimands import Vectors  # noqa: E402
from dcorr.analysis.multilevel import difficulty_decomposition, predicted_phi  # noqa: E402

ALPHA = {"undefended": 0.8, "a": 0.4, "b": 0.2, "c": -0.3}
N, SEEDS, SD = 300, (0, 1, 2), 1.5


def _simulate(shared_ab: float = 0.0, seed: int = 0) -> Vectors:
    rng = np.random.default_rng(seed)
    u = rng.normal(0, SD, N)                       # behaviour difficulty, shared by all
    w = rng.normal(0, shared_ab, N) if shared_ab else np.zeros(N)   # extra a-b mechanism
    vec = {}
    for s in SEEDS:
        vec[s] = {}
        for d, al in ALPHA.items():
            eta = al + u + (w if d in ("a", "b") else 0.0)
            vec[s][d] = (rng.random(N) < 1 / (1 + np.exp(-eta))).astype(int)
    return Vectors(behaviours=[f"b{i}" for i in range(N)], vec=vec)


def test_predicted_phi_limits():
    assert predicted_phi(0.0, 0.0, 0.0) == pytest.approx(0.0, abs=1e-9)   # no difficulty
    assert predicted_phi(0.0, 0.0, 3.0) > predicted_phi(0.0, 0.0, 1.0) > 0


def test_difficulty_only_data_shows_no_excess():
    out = difficulty_decomposition(_simulate())
    assert out["fit"]["sd_behaviour"] == pytest.approx(SD, rel=0.25)
    for pair, v in out["pairs"].items():
        assert v["observed"] > 0.15, pair                     # difficulty alone correlates
        assert abs(v["excess"]) < 0.08, (pair, v)             # ...and is fully explained


def test_shared_mechanism_shows_up_as_excess():
    out = difficulty_decomposition(_simulate(shared_ab=1.5))
    assert out["pairs"]["a|b"]["excess"] > 0.10, out["pairs"]["a|b"]
    assert out["pairs"]["a|b"]["excess"] > out["pairs"]["a|c"]["excess"] + 0.08
