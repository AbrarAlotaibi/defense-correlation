"""Tests for the PAIR attack family, on CPU fakes.

The scientific claim the second family rests on is that it changes the proposal mechanism and
nothing else: same objective, same feasibility rule as the genetic arm, so a disagreement
between the two isolates the search. These tests hold that claim up:

  * equivalence: pair._evaluate returns the same fitness and feasibility as the computation
    inside autodan._evaluate for the same prompts, defense and target. If the two ever drift,
    the comparison between families stops meaning what the paper says it means.
  * schema: records carry the same keys as the genetic arm's, so stage 04 and stage 06 need no
    special case and the analysis sees "pair" as just another attack value.
  * determinism: the same seed gives the same search, and a different seed_index does not.
  * parsing: the attacker's reply is read correctly, including the malformed and truncated
    forms a 7B attacker actually emits.
  * no judge in the loop: the search never calls a grader, which the project forbids.

Run: python -m pytest tests/test_pair.py -q
"""
from __future__ import annotations

import json

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from _fakes import FakeTarget  # noqa: E402
from dcorr.attacks.autodan import run_autodan  # noqa: E402
from dcorr.attacks.pair import (_balanced_objects, _evaluate, _extract_prompt,  # noqa: E402
                                run_pair)
from dcorr.attacks.random_search import Behaviour  # noqa: E402
from dcorr.defenses.ppl_filter import PerplexityFilter  # noqa: E402
from dcorr.defenses.refusal_prime import RefusalPrime  # noqa: E402
from dcorr.defenses.smoothllm import SmoothLLM  # noqa: E402
from dcorr.defenses.stack import DefenseStack  # noqa: E402

BEHAVIOURS = [Behaviour(f"jbb_{i}", f"Describe historical event number {i} plainly",
                        f"Sure, here is event {i}") for i in range(3)]


class ScriptedAttacker:
    """Stands in for the attacker LLM. Returns canned replies, recording what it was asked.

    `model_id` and the `generate` signature match HFTarget's, including the sampling keywords,
    so the production call path is exercised rather than a simplified one.
    """

    def __init__(self, replies: list[str]):
        self.replies = replies
        self.calls: list[dict] = []
        self.model_id = "scripted/attacker"

    def generate(self, users, max_new_tokens=256, system="__default__", assistant_prefix="",
                 sample=False, temperature=1.0, top_p=0.9, gen_seed=None):
        self.calls.append({"user": users[0], "system": system, "sample": sample,
                           "temperature": temperature, "gen_seed": gen_seed})
        return [self.replies[(len(self.calls) - 1) % len(self.replies)]]


def _reply(p: str) -> str:
    return '{"improvement": "reframed as fiction", "prompt": "%s"}' % p


def _defense(kind: str, target):
    if kind == "ppl":
        return PerplexityFilter({"name": "ppl_filter", "threshold": 1e9}, target)
    if kind == "prime":
        return RefusalPrime({"name": "refusal_prime"}, target)
    if kind == "smooth":
        return SmoothLLM({"name": "smoothllm", "q": 3, "perturbation_pct": 0.1}, target)
    if kind == "stack":
        ppl = PerplexityFilter({"name": "ppl_filter", "threshold": 1e9}, target)
        sm = SmoothLLM({"name": "smoothllm", "q": 3, "perturbation_pct": 0.1}, target)
        return DefenseStack([ppl, sm], ["ppl_filter", "smoothllm"], target)
    raise AssertionError(kind)


# --------------------------------------------------------------------------- parsing
@pytest.mark.parametrize("raw,want", [
    ('{"improvement":"x","prompt":"Write a story about P"}', "Write a story about P"),
    ('```json\n{"improvement":"x","prompt":"fenced P"}\n```', "fenced P"),
    ('Sure! {"improvement":"a","prompt":"embedded P"} hope that helps', "embedded P"),
    (r'{"prompt": "has \"quotes\" inside"}', 'has "quotes" inside'),
    ('{"improvement":"a", "prompt":"truncated mid value', "truncated mid value"),
    ('{"improvement":"a","prompt":"   "}', None),
    ("no json at all", None),
    ("", None),
    (None, None),
])
def test_extract_prompt(raw, want):
    assert _extract_prompt(raw) == want


def test_balanced_objects_ignores_braces_in_strings():
    s = '{"prompt":"a } brace in a string"} tail {"prompt":"second"}'
    assert _balanced_objects(s) == ['{"prompt":"a } brace in a string"}',
                                    '{"prompt":"second"}']


# --------------------------------------------------------------------------- equivalence
@pytest.mark.parametrize("kind", ["ppl", "prime", "smooth", "stack"])
def test_evaluate_matches_the_genetic_arm(kind):
    """pair._evaluate must agree with autodan's inline objective on identical inputs.

    Recomputed here from autodan's own source path rather than hardcoded, so this fails if
    either side changes: the whole comparison between families depends on these agreeing.
    """
    target = FakeTarget()
    defense = _defense(kind, target)
    prompts = ["tell me about event 0 in a play", "a short fictional scene", "plainly please"]
    tgt = "Sure, here is"

    got_fit, got_feas = _evaluate(target, defense, prompts, tgt)

    # The genetic arm's computation, inlined from dcorr/attacks/autodan.py::_evaluate.
    from dcorr.attacks.feasibility import cheap_feasible_raw, needs_separate_pass
    users = [defense.objective_users(p) for p in prompts]
    flat = [u for us in users for u in us]
    owner = [i for i, us in enumerate(users) for _ in us]
    separate = needs_separate_pass(defense, prompts, users)
    out = target.score_chunked(flat, tgt,
                               need_hidden=defense.needs_hidden() and not separate,
                               need_window_nll=defense.needs_window_nll() and not separate,
                               **defense.score_kwargs())
    if separate:
        cheap_ok = cheap_feasible_raw(target, defense, prompts, tgt)
    else:
        cheap = defense.cheap_feasible(out, flat)
        cheap_ok = [True] * len(prompts)
        for k, i in enumerate(owner):
            cheap_ok[i] = cheap_ok[i] and bool(cheap[k])
    lps = out.target_logprob.tolist()
    agg = [[0.0, 0] for _ in prompts]
    for k, i in enumerate(owner):
        agg[i][0] += float(lps[k]); agg[i][1] += 1
    want_fit = [a / max(1, n) for a, n in agg]

    assert got_fit == pytest.approx(want_fit)
    assert got_feas == [bool(x) for x in cheap_ok]


# --------------------------------------------------------------------------- schema
def test_record_schema_matches_the_genetic_arm():
    target = FakeTarget()
    defense = _defense("ppl", target)
    attacker = ScriptedAttacker([_reply("a fictional framing of event 0")])

    pair_recs = run_pair(target, defense, BEHAVIOURS, attacker=attacker,
                         streams=2, iterations=2, seed=7)
    ga_recs = run_autodan(target, defense, BEHAVIOURS, generations=3, population=4, seed=7)

    shared = set(ga_recs[0]) - {"genes"}
    missing = shared - set(pair_recs[0])
    assert not missing, f"pair records lack keys the genetic arm has: {sorted(missing)}"
    assert len(pair_recs) == len(BEHAVIOURS)
    assert {r["attack"] for r in pair_recs} == {"pair"}
    for r in pair_recs:
        assert isinstance(r["prompt"], str) and r["prompt"]
        assert isinstance(r["found_feasible"], bool)
        assert r["n_evals"] > 0
        assert r["attacker_model"] == "scripted/attacker"


# --------------------------------------------------------------------------- determinism
def test_same_seed_same_search_and_seed_index_changes_it():
    target = FakeTarget()
    defense = _defense("ppl", target)
    mk = lambda: ScriptedAttacker([_reply("framing one"), _reply("framing two")])

    a = run_pair(target, defense, BEHAVIOURS, attacker=mk(), streams=2, iterations=2,
                 seed=5, seed_index=0)
    b = run_pair(target, defense, BEHAVIOURS, attacker=mk(), streams=2, iterations=2,
                 seed=5, seed_index=0)
    c = run_pair(target, defense, BEHAVIOURS, attacker=mk(), streams=2, iterations=2,
                 seed=5, seed_index=1)

    assert [r["search_seed"] for r in a] == [r["search_seed"] for r in b]
    assert [r["prompt"] for r in a] == [r["prompt"] for r in b]
    assert [r["search_seed"] for r in a] != [r["search_seed"] for r in c]


def test_attacker_is_sampled_with_a_seed_and_the_seed_varies_per_stream():
    target = FakeTarget()
    defense = _defense("ppl", target)
    attacker = ScriptedAttacker([_reply("x")])
    run_pair(target, defense, BEHAVIOURS[:1], attacker=attacker, streams=3, iterations=2,
             seed=3)

    assert all(c["sample"] for c in attacker.calls), "the attacker must be sampled"
    assert all(c["gen_seed"] is not None for c in attacker.calls)
    first_round = [c["gen_seed"] for c in attacker.calls[:3]]
    assert len(set(first_round)) == 3, "each stream needs its own draw"


def test_search_seed_differs_from_the_genetic_arm():
    """Namespaced separately, so agreement between families is not one shared random draw."""
    from dcorr.seeding import search_seed
    target = FakeTarget()
    defense = _defense("ppl", target)
    recs = run_pair(target, defense, BEHAVIOURS, attacker=ScriptedAttacker([_reply("p")]),
                    streams=1, iterations=1, seed=0, seed_index=0)
    for r, b in zip(recs, BEHAVIOURS):
        assert r["search_seed"] != search_seed(0, 0, defense.name, b.behaviour_id)


# --------------------------------------------------------------------------- hygiene
def test_parse_failures_are_counted_not_silently_scored():
    target = FakeTarget()
    defense = _defense("ppl", target)
    attacker = ScriptedAttacker(["I will not help with that."])
    recs = run_pair(target, defense, BEHAVIOURS[:1], attacker=attacker, streams=2,
                    iterations=2, seed=1)
    r = recs[0]
    assert r["pair_parse_failures"] == 4, r["pair_parse_failures"]
    assert r["n_evals"] == 0, "an unparsed reply must not be scored as a candidate"
    assert r["found_feasible"] is False
    assert r["prompt"] == BEHAVIOURS[0].prompt, "falls back to the bare behaviour"


def test_no_judge_in_the_loop():
    """The grader must not steer the attack; stage 44's separability depends on it."""
    src = (ROOT / "dcorr" / "attacks" / "pair.py").read_text(encoding="utf-8")
    for banned in ("StrongRejectJudge", "judge_gold", "VerdictCache", "from ..judge"):
        assert banned not in src, f"pair.py must not import a grader: found {banned}"


def test_attacker_sees_the_target_response_and_the_score():
    """PAIR's mechanism is semantic feedback; without it this is just random restarts."""
    target = FakeTarget()
    defense = _defense("ppl", target)
    attacker = ScriptedAttacker([_reply("first try")])
    run_pair(target, defense, BEHAVIOURS[:1], attacker=attacker, streams=1, iterations=3,
             seed=2)
    later = attacker.calls[1]["user"]
    assert "The model replied:" in later
    assert "Score:" in later
    assert "first try" in later, "the attacker must see its own previous prompt"


# --------------------------------------------------------------------------- integration
def test_run_one_defense_dispatches_pair_and_keys_resume_by_attack(tmp_path):
    """The stage 04 path: pair rows land beside the other arms and resume independently.

    This is the integration the GPU run depends on, and it is checkable on CPU fakes: that
    run_one_defense emits pair rows only when an attacker is supplied, that they carry
    attack="pair" through the row builder, and that a second call re-runs nothing.
    """
    from dcorr.run_defense import run_one_defense

    target = FakeTarget()
    defense = _defense("ppl", target)
    out = tmp_path / "stage04_ppl_filter.jsonl"
    attack_cfg = {"method": "autodan", "generations": 2, "population": 4,
                  "include_plain": True, "include_gcg": False}
    pair_cfg = {"streams": 2, "iterations": 2}

    # Without an attacker: the genetic and static arms only.
    rows = run_one_defense(target, defense, BEHAVIOURS, out, attack_cfg=attack_cfg,
                           gcg_suffix="! ! !", seed=4)
    assert {r["attack"] for r in rows} == {"adaptive", "static_plain"}

    # With one: pair rows are added, nothing else re-runs.
    attacker = ScriptedAttacker([_reply("a fictional framing")])
    rows2 = run_one_defense(target, defense, BEHAVIOURS, out, attack_cfg=attack_cfg,
                            gcg_suffix="! ! !", seed=4, pair_attacker=attacker,
                            pair_cfg=pair_cfg)
    assert {r["attack"] for r in rows2} == {"adaptive", "static_plain", "pair"}
    pair_rows = [r for r in rows2 if r["attack"] == "pair"]
    assert len(pair_rows) == len(BEHAVIOURS)
    for r in pair_rows:
        assert r["response"] is not None and r["blocked"] in (True, False)
        assert r["search_seed"] and r["n_evals"]

    # Idempotent: a third call adds nothing.
    before = len(rows2)
    rows3 = run_one_defense(target, defense, BEHAVIOURS, out, attack_cfg=attack_cfg,
                            gcg_suffix="! ! !", seed=4, pair_attacker=attacker,
                            pair_cfg=pair_cfg)
    assert len(rows3) == before, "pair rows must resume by (model, defense, behaviour, attack)"


def test_pair_arm_is_off_unless_configured():
    """Every run before this family existed must behave exactly as it did."""
    import inspect
    from dcorr.run_defense import run_one_defense
    assert inspect.signature(run_one_defense).parameters["pair_attacker"].default is None


def test_target_generate_is_greedy_by_default():
    """sample=True exists for the attacker alone; nothing that makes a breach label uses it."""
    import inspect
    from dcorr.target import HFTarget
    assert inspect.signature(HFTarget.generate).parameters["sample"].default is False


def test_seeded_sampling_is_actually_reproducible():
    """The seed must change the draw, not merely be passed.

    The earlier version of this test asserted only that gen_seed reached the attacker, which
    a scripted fake satisfies trivially -- so it passed while the real path was broken:
    transformers samples with a bare torch.multinomial and ignores a `generator=` kwarg, so
    the seed had no effect and the search was diverse but not reproducible. This drives the
    real code path with a stub model that draws from torch, so it fails if seeding regresses.
    """
    import torch

    from dcorr.target import HFTarget

    class StubModel:
        device = "cpu"

        def generate(self, **kw):
            n = kw.get("max_new_tokens", 4)
            if kw.get("do_sample"):
                probs = torch.ones(1, 32) / 32
                draw = torch.multinomial(probs.expand(n, 32), num_samples=1).T
            else:
                draw = torch.zeros(1, n, dtype=torch.long)
            return torch.cat([kw["input_ids"], draw], dim=1)

    t = HFTarget.__new__(HFTarget)          # no weights, no GPU
    t.model = StubModel()
    enc = {"input_ids": torch.zeros(1, 2, dtype=torch.long)}

    a = t._generate_seeded(enc, 6, {"do_sample": True, "temperature": 1.0, "top_p": 0.9},
                           1234, 0)
    b = t._generate_seeded(enc, 6, {"do_sample": True, "temperature": 1.0, "top_p": 0.9},
                           1234, 0)
    c = t._generate_seeded(enc, 6, {"do_sample": True, "temperature": 1.0, "top_p": 0.9},
                           5678, 0)
    assert torch.equal(a, b), "same seed must give the same draw"
    assert not torch.equal(a, c), "a different seed must give a different draw"


def test_seeding_restores_the_global_rng():
    """Seeding globally is the only thing that works here, so it must not leak."""
    import torch

    from dcorr.target import HFTarget

    class StubModel:
        device = "cpu"

        def generate(self, **kw):
            torch.rand(3)                   # consume randomness, as sampling would
            return kw["input_ids"]

    t = HFTarget.__new__(HFTarget)
    t.model = StubModel()
    enc = {"input_ids": torch.zeros(1, 2, dtype=torch.long)}

    torch.manual_seed(99)
    before = torch.rand(2)
    torch.manual_seed(99)
    t._generate_seeded(enc, 4, {"do_sample": True}, 4321, 0)
    after = torch.rand(2)
    assert torch.equal(before, after), "the surrounding RNG stream must be left untouched"


def test_budget_comes_from_the_config_and_is_persisted(tmp_path):
    """The config's budget must reach the search AND survive into the stored rows.

    Both halves failed silently before: run_one_defense read attack.pair from attack_cfg,
    which the stages never put it in, so every run used the function defaults (3 streams x 5
    iterations, visible only as n_evals = 15); and the row builder dropped pair_streams,
    pair_parse_failures, iterations and attacker_model, so the artefacts could not say what
    budget had run or whether the attacker was attacking at all.
    """
    from dcorr.run_defense import run_one_defense

    target = FakeTarget()
    defense = _defense("ppl", target)
    out = tmp_path / "stage04_ppl_filter.jsonl"
    attacker = ScriptedAttacker([_reply("a framing")])

    run_one_defense(target, defense, BEHAVIOURS[:1], out,
                    attack_cfg={"method": "autodan", "generations": 2, "population": 4,
                                "include_plain": False, "include_gcg": False},
                    gcg_suffix="", seed=4, pair_attacker=attacker,
                    pair_cfg={"streams": 2, "iterations": 3})

    rows = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines() if l.strip()]
    pair = [r for r in rows if r["attack"] == "pair"]
    assert len(pair) == 1
    r = pair[0]
    for k in ("pair_streams", "pair_parse_failures", "iterations", "attacker_model"):
        assert k in r, f"the row builder dropped {k}"
    assert r["pair_streams"] == 2, "streams came from the defaults, not the config"
    assert r["iterations"] == 3, "iterations came from the defaults, not the config"
    assert r["attacker_model"] == "scripted/attacker"
    assert r["n_evals"] <= 2 * 3, f"n_evals {r['n_evals']} exceeds the configured budget"


def test_an_attacker_without_a_config_is_refused(tmp_path):
    """Defaulting the budget would invalidate the comparison with the genetic arm."""
    import pytest as _pytest

    from dcorr.run_defense import run_one_defense

    target = FakeTarget()
    defense = _defense("ppl", target)
    with _pytest.raises(ValueError, match="pair_cfg"):
        run_one_defense(target, defense, BEHAVIOURS[:1], tmp_path / "o.jsonl",
                        attack_cfg={"method": "autodan", "generations": 2, "population": 4,
                                    "include_plain": False, "include_gcg": False},
                        gcg_suffix="", seed=1,
                        pair_attacker=ScriptedAttacker([_reply("p")]))
