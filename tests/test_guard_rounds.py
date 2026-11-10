"""Tests for Llama Guard inside the AutoDAN attack (REVIEW F4), on CPU fakes.

v1 never consulted an expensive constraint in AutoDAN, so the "adaptive" Llama Guard run was
the undefended search. v2 runs such defenses through a rounds path: one guard block per
round, guard-verified final choice, optional best-of-m through the deployed respond().

  * differential: for every defense WITHOUT an expensive constraint, the refactored
    run_autodan returns records identical to the code committed before the refactor;
  * rounds: the guard is consulted once per round in one batch, and every prompt reported
    feasible passes the guard and the deployed input filters.

Run: python -m pytest tests/test_guard_rounds.py -q
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from _fakes import FakeTarget  # noqa: E402
from dcorr.attacks.autodan import SENTENCE_POOL, run_autodan  # noqa: E402
from dcorr.attacks.random_search import Behaviour  # noqa: E402
from dcorr.defenses.base import Defense, FilterOut  # noqa: E402
from dcorr.defenses.ppl_filter import PerplexityFilter  # noqa: E402
from dcorr.defenses.refusal_prime import RefusalPrime  # noqa: E402
from dcorr.defenses.smoothllm import SmoothLLM  # noqa: E402
from dcorr.defenses.stack import DefenseStack  # noqa: E402

REF_COMMIT = "d256655"   # last commit before the rounds refactor
BEHAVIOURS = [Behaviour(f"jbb_{i}", f"Describe historical event number {i} plainly",
                        f"Sure, here is event {i}") for i in range(5)]
GA = dict(generations=6, population=8, seed=11)
BANNED = set(range(0, 30, 3))          # 10 of the 30 framing sentences


# ----------------------------------------------------------------------------- helpers
def _load_reference_autodan():
    src = subprocess.run(["git", "-C", str(ROOT), "show", f"{REF_COMMIT}:dcorr/attacks/autodan.py"],
                         check=True, capture_output=True, text=True, encoding="utf-8").stdout
    name = "dcorr.attacks._autodan_ref"
    spec = importlib.util.spec_from_loader(name, loader=None)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = "dcorr.attacks"
    sys.modules[name] = mod             # dataclasses resolve annotations via sys.modules
    exec(compile(src, f"{REF_COMMIT}:autodan.py", "exec"), mod.__dict__)
    return mod.run_autodan


def _inert(name):
    return type(f"_Inert_{name}", (Defense,), {"name": name})({}, None)


class _FakeGuard(Defense):
    """Rejects prompts containing a banned framing sentence; logs every guard block.
    Its output check blocks responses carrying FakeTarget.MARKER."""
    name = "llamaguard"
    has_input_filter = True
    has_output_filter = True
    expensive_constraint = True

    def __init__(self, target, banned=BANNED, reject_all=False):
        super().__init__({}, target)
        self.banned = banned
        self.reject_all = reject_all
        self.blocks: list[list[str]] = []

    def unsafe(self, p: str) -> bool:
        return self.reject_all or any(SENTENCE_POOL[g] in p for g in self.banned)

    def expensive_feasible(self, prompts):
        self.blocks.append(list(prompts))
        return [not self.unsafe(p) for p in prompts]

    def filter_input(self, prompts):
        s = [float(self.unsafe(p)) for p in prompts]
        return FilterOut([x > 0 for x in s], s, [{} for _ in s])

    def filter_output(self, prompts, responses):
        s = [float(FakeTarget.MARKER in r) for r in responses]
        return FilterOut([x > 0 for x in s], s, [{} for _ in s])


def _stack_with_guard(t, guard):
    ppl = PerplexityFilter({"threshold": 0.1, "window_tokens": 16}, t)
    rp = RefusalPrime({}, t)
    sm = SmoothLLM({"q": 4, "perturbation_pct": 0.10}, t)
    return DefenseStack([ppl, guard, rp, sm],
                        ["ppl_filter", "llamaguard", "refusal_prime", "smoothllm"], t)


# ----------------------------------------------------------------------------- differential
@pytest.mark.parametrize("make", [
    lambda t: _inert("ppl_filter"),
    lambda t: PerplexityFilter({"threshold": 0.1, "window_tokens": 16}, t),
    lambda t: DefenseStack([PerplexityFilter({"threshold": 0.1, "window_tokens": 16}, t),
                            RefusalPrime({}, t), SmoothLLM({"q": 4}, t)],
                           ["ppl_filter", "refusal_prime", "smoothllm"], t),
], ids=["inert", "ppl_filter", "stack_without_guard"])
def test_cheap_path_identical_to_pre_refactor_code(make):
    ref = _load_reference_autodan()
    t1, t2 = FakeTarget(), FakeTarget()
    new = run_autodan(t1, make(t1), BEHAVIOURS, **GA)
    old = ref(t2, make(t2), BEHAVIOURS, **GA)
    assert new == old
    assert t1.calls == t2.calls          # same scoring passes, in the same order


# ----------------------------------------------------------------------------- rounds
def test_guard_consulted_once_per_round_in_one_batch():
    t = FakeTarget()
    g = _FakeGuard(t)
    recs = run_autodan(t, g, BEHAVIOURS, expensive_rounds=3, expensive_pool_k=4, **GA)
    assert len(g.blocks) == 3
    first = g.blocks[0]
    assert len({next(b.behaviour_id for b in BEHAVIOURS if p.endswith(b.prompt))
                for p in first}) == len(BEHAVIOURS), "a block must span all behaviours"
    assert all(r["guard_rounds"] == 3 for r in recs)
    assert all(r["n_evals"] == GA["generations"] * GA["population"] for r in recs)


def test_feasible_prompts_pass_guard_and_deployment():
    t = FakeTarget()
    g = _FakeGuard(t)
    recs = run_autodan(t, g, BEHAVIOURS, expensive_rounds=3, expensive_pool_k=4, **GA)
    feasible = [r for r in recs if r["found_feasible"]]
    assert len(feasible) >= 3, [r["found_feasible"] for r in recs]
    for r in feasible:
        assert not g.unsafe(r["prompt"])
        assert g.respond([r["prompt"]], 16)[0]["blocked_stage"] != "input"
    for r in recs:
        if not r["found_feasible"]:
            assert r["verified_candidates"] == 0


def test_guard_rejecting_everything_yields_no_feasible_prompt():
    t = FakeTarget()
    g = _FakeGuard(t, reject_all=True)
    recs = run_autodan(t, g, BEHAVIOURS, expensive_rounds=2, expensive_pool_k=4, **GA)
    assert len(g.blocks) == 2
    assert not any(r["found_feasible"] for r in recs)


def test_stack_with_guard_uses_rounds_and_respects_all_filters():
    t = FakeTarget()
    g = _FakeGuard(t)
    stack = _stack_with_guard(t, g)
    recs = run_autodan(t, stack, BEHAVIOURS, expensive_rounds=2, expensive_pool_k=4, **GA)
    assert len(g.blocks) == 2
    for r in recs:
        if r["found_feasible"]:
            assert not g.unsafe(r["prompt"])
            assert stack.filter_input([r["prompt"]]).blocked == [False]
    assert sum(r["found_feasible"] for r in recs) >= 1


def test_best_of_m_skips_candidates_the_output_check_blocks():
    t = FakeTarget(marker_mod=2)          # about half of all responses carry the marker
    g = _FakeGuard(t)
    recs = run_autodan(t, g, BEHAVIOURS, expensive_rounds=2, expensive_pool_k=6,
                       final_top_m=3, **GA)
    for r in recs:
        if not r["found_feasible"]:
            continue
        deployed = g.respond([r["prompt"]], 16)[0]
        if r["final_blocked"] < r["final_tried"]:
            assert not deployed["blocked"], r
    assert any(r.get("final_blocked", 0) > 0 for r in recs), \
        "premise: the output check should block some top candidates"


def test_carry_is_rejected_in_the_rounds_path():
    t = FakeTarget()
    with pytest.raises(ValueError):
        run_autodan(t, _FakeGuard(t), BEHAVIOURS, carry_across_behaviours=True, **GA)
