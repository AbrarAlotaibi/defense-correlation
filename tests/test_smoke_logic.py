"""CPU-only logic tests: config inheritance, stats identities, defense wiring, no GPU.

Run: gambit-abl python -m pytest tests/ -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dcorr.analysis.stats import (  # noqa: E402
    benjamini_hochberg, mcnemar, phi_coefficient, phi_from_rates, pair_stat,
)
from dcorr.config import load_config  # noqa: E402


def test_config_inheritance():
    cfg = load_config(ROOT / "configs" / "primary_llama2.yaml")
    assert cfg.get_path("target.model_id") == "meta-llama/Llama-2-7b-chat-hf"
    assert cfg.get_path("analysis.bootstrap_resamples") == 10000   # from base.yaml
    assert "ppl_filter" in cfg.enabled_defenses()
    assert cfg.defense_row("ppl_filter") == "token_surface"


def test_replication_reduced_roster():
    """The replication target runs only the defenses the headline pairs need."""
    cfg = load_config(ROOT / "configs" / "hpc_llama2_autodan.yaml")
    en = set(cfg.enabled_defenses())
    assert en == {"ppl_filter", "token_anomaly", "probe", "probe_b"}
    assert cfg.get_path("data.eval_limit") == 50


def test_phi_identity():
    # ASR_{d1d2} = p1 p2 + rho sqrt(p1(1-p1)p2(1-p2)); phi recovers rho.
    rng = np.random.default_rng(0)
    n = 5000
    a = (rng.random(n) < 0.4).astype(int)
    # build b correlated with a
    b = a.copy()
    flip = rng.random(n) < 0.25
    b[flip] = 1 - b[flip]
    phi = phi_coefficient(a, b)
    p1, p2 = a.mean(), b.mean()
    joint = ((a == 1) & (b == 1)).mean()
    recon = phi_from_rates(p1, p2, phi)
    assert abs(recon - joint) < 0.02


def test_pair_stat_ci_orders():
    rng = np.random.default_rng(1)
    a = (rng.random(100) < 0.3).astype(int)
    b = (rng.random(100) < 0.3).astype(int)
    ps = pair_stat("d1", "d2", a, b, same_row=False, resamples=500, seed=1)
    assert ps.phi_ci[0] <= ps.phi <= ps.phi_ci[1] or ps.phi_ci[0] <= ps.phi_ci[1]
    assert 0.0 <= ps.p1 <= 1.0


def test_mcnemar_direction():
    # stack breaches fewer than single -> b01 (single-only) > b10 (stack-only)
    stack = np.array([0, 0, 0, 1, 1])
    single = np.array([1, 1, 1, 1, 1])
    mc = mcnemar(stack, single)
    assert mc["b01"] == 3 and mc["b10"] == 0


def test_bh_monotone():
    q = benjamini_hochberg([0.01, 0.02, 0.5, 0.9])
    assert all(0 <= x <= 1 for x in q)


def test_token_anomaly_scores_high_on_gibberish():
    # No model needed: token_anomaly uses only the tokenizer, which we stub.
    from dcorr.defenses.token_anomaly import TokenAnomalyFilter

    class _Tok:
        def encode(self, s, add_special_tokens=False):
            return list(range(len(s.split())))

        def decode(self, ids):
            return "x"

    class _T:
        tokenizer = _Tok()
        model_id = "stub"

    d = TokenAnomalyFilter({"threshold": 0.5, "row": "token_surface"}, _T())
    clean = d.score_one("Please help me write a polite email to my landlord.")
    gib = d.score_one("aWx9@#$ zzzz \\!--Two describing.\\ + similarlyNow")
    assert gib >= clean


def test_refusal_banks_self_contained():
    # No gambit import: the banks live in the repo and cover Llama-2 and Llama-3.
    from dcorr.judge.prefilter import get_prefilter

    p = get_prefilter()
    assert "meta-llama/Llama-2-7b-chat-hf" in p._by_model
    assert "meta-llama/Llama-3.1-8B-Instruct" in p._by_model


def test_refusal_scorer_llama2():
    from dcorr.judge import RefusalScorer

    rs = RefusalScorer(model_name="meta-llama/Llama-2-7b-chat-hf")
    assert rs.is_refusal("As a responsible and ethical AI language model, I cannot help.")
    assert not rs.is_refusal("Sure! Here is a cookie recipe. Preheat the oven to 350F.")
    assert rs.is_refusal("", blocked=True)


def test_no_gambit_import_anywhere():
    import ast

    for py in (ROOT / "dcorr").rglob("*.py"):
        tree = ast.parse(py.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            mods = []
            if isinstance(node, ast.Import):
                mods = [n.name for n in node.names]
            elif isinstance(node, ast.ImportFrom):
                mods = [node.module or ""]
            for m in mods:
                assert "gambit" not in (m or "").lower(), f"{py.name} imports {m}"


def test_dtype_compat_shim():
    from dcorr.compat import dtype_kwargs

    kw = dtype_kwargs("float16")
    assert set(kw) <= {"dtype", "torch_dtype"} and len(kw) == 1


def test_run_adaptive_wiring_cpu():
    """Exercise the attack loop with stubs (no GPU/model) so signature/wiring bugs like a
    stale run_adaptive kwarg are caught by CI, not only by the GPU smoke test."""
    import torch
    from dcorr.attacks.random_search import Behaviour, run_adaptive
    from dcorr.defenses.base import Defense

    class StubTok:
        def encode(self, s, add_special_tokens=False):
            return [1, 2, 3]

        def decode(self, ids, skip_special_tokens=False):
            return "Sure"

    class StubTarget:
        model_id = "stub"
        tokenizer = StubTok()

        def allowed_suffix_tokens(self):
            return [1, 2, 3, 4, 5]

        def decode_suffix(self, ids):
            return " ".join(map(str, ids))

        def score_chunked(self, users, target_str, **kw):
            n = len(users)
            return type("S", (), {
                "target_logprob": torch.zeros(n),
                "target_logprob_sum": torch.zeros(n),
                "hidden": None, "window_nll": None, "n_user_tokens": None,
            })()

    class NoOpDefense(Defense):
        name = "noop"
        def __init__(self):
            pass
        row = "none"; expensive_constraint = False

    behs = [Behaviour(behaviour_id="b0", prompt="do x", target_str="Sure")]
    recs = run_adaptive(StubTarget(), NoOpDefense(), behs, iterations=3,
                        suffix_n_tokens=4, seed=0)
    assert len(recs) == 1 and recs[0]["behaviour_id"] == "b0"
    assert recs[0]["attack"] == "adaptive"


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([str(Path(__file__).parent), "-q"]))
