"""CPU-only tests for v2 search seeding (dcorr/seeding.py; REVIEW F2).

v1 gave every defense the same RNG stream, reset at each behaviour chunk, so two defenses
that neither changed the objective nor rejected a candidate replayed one identical search.
These tests pin the v2 contract with a fake target, so no model is downloaded:

  * different defenses get different searches; common_random_numbers=True restores sharing;
  * a behaviour's result does not depend on chunking or on the behaviours before it;
  * seeds are identical across processes (no reliance on Python's salted hash()).

Run: python -m pytest tests/test_seeding.py -q
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dcorr.attacks.autodan import run_autodan  # noqa: E402
from dcorr.attacks.random_search import Behaviour  # noqa: E402
from dcorr.config import apply_seed_index, load_config  # noqa: E402
from dcorr.defenses.base import Defense  # noqa: E402
from dcorr.seeding import search_seed, stable_seed  # noqa: E402


# ----------------------------------------------------------------------------- fakes
class _CharTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]

    def decode(self, ids):
        return "".join(chr(i) for i in ids)


class _Scored:
    def __init__(self, lp):
        self.target_logprob = torch.tensor(lp)
        self.hidden = None
        self.window_nll = None


class _FakeTarget:
    """Fitness is a fixed pseudo-random function of the prompt text, so the GA has a real
    landscape to climb and any difference between two runs comes from their RNG alone."""
    model_id = "fake/target"
    tokenizer = _CharTokenizer()

    def score_chunked(self, users, target_str, **_kw):
        return _Scored([-(stable_seed("fit", u) % 10_000) / 2_000.0 for u in users])


def _inert(name: str) -> Defense:
    """A defense that changes neither the objective nor feasibility (like ppl_filter or
    probe_b in v1, which never rejected a fluent candidate)."""
    cls = type(f"_Inert_{name}", (Defense,), {"name": name})
    return cls({}, _FakeTarget())


BEHAVIOURS = [Behaviour(f"jbb_{i}", f"Explain topic number {i} in detail",
                        f"Sure, here is topic {i}") for i in range(6)]
GA = dict(generations=4, population=8, seed=20260727)


def _genes(records):
    return {r["behaviour_id"]: tuple(r["genes"]) for r in records}


# ----------------------------------------------------------------------------- tests
def test_stable_seed_is_deterministic_and_distinct():
    assert stable_seed("a", 1) == stable_seed("a", 1)
    assert stable_seed("a", 1) != stable_seed("a", 2)
    assert stable_seed("ab", "c") != stable_seed("a", "bc")   # separator prevents collisions
    assert 0 <= stable_seed("x") < 2**63


def test_search_seed_depends_on_defense_unless_crn():
    a = search_seed(1, 0, "ppl_filter", "jbb_0")
    b = search_seed(1, 0, "probe_b", "jbb_0")
    assert a != b
    assert (search_seed(1, 0, "ppl_filter", "jbb_0", common_random_numbers=True)
            == search_seed(1, 0, "probe_b", "jbb_0", common_random_numbers=True))
    assert search_seed(1, 0, "ppl_filter", "jbb_0") != search_seed(1, 1, "ppl_filter", "jbb_0")


def test_inert_defenses_no_longer_replay_one_search():
    """The v1 failure: two inert defenses produced byte-identical prompts on 100/100."""
    g1 = _genes(run_autodan(_FakeTarget(), _inert("ppl_filter"), BEHAVIOURS, **GA))
    g2 = _genes(run_autodan(_FakeTarget(), _inert("probe_b"), BEHAVIOURS, **GA))
    differing = sum(g1[k] != g2[k] for k in g1)
    assert differing >= 4, f"only {differing}/6 searches differ between defenses"


def test_common_random_numbers_restores_shared_search():
    kw = dict(GA, common_random_numbers=True)
    g1 = _genes(run_autodan(_FakeTarget(), _inert("ppl_filter"), BEHAVIOURS, **kw))
    g2 = _genes(run_autodan(_FakeTarget(), _inert("probe_b"), BEHAVIOURS, **kw))
    assert g1 == g2


def test_results_do_not_depend_on_chunking():
    """v1 reset the RNG per 10-behaviour chunk in stage 04 but not in stage 03."""
    d = _inert("token_anomaly")
    whole = _genes(run_autodan(_FakeTarget(), d, BEHAVIOURS, **GA))
    chunked = {}
    for c0 in range(0, len(BEHAVIOURS), 2):
        chunked.update(_genes(run_autodan(_FakeTarget(), d, BEHAVIOURS[c0:c0 + 2], **GA)))
    assert whole == chunked


def test_behaviour_result_independent_of_predecessors():
    d = _inert("probe")
    in_list = _genes(run_autodan(_FakeTarget(), d, BEHAVIOURS, **GA))["jbb_4"]
    alone = _genes(run_autodan(_FakeTarget(), d, BEHAVIOURS[4:5], **GA))["jbb_4"]
    assert in_list == alone


def test_seed_index_changes_search_and_is_recorded():
    d = _inert("smoothllm_like")
    r0 = run_autodan(_FakeTarget(), d, BEHAVIOURS, seed_index=0, **GA)
    r1 = run_autodan(_FakeTarget(), d, BEHAVIOURS, seed_index=1, **GA)
    assert all(r["seed_index"] == 0 for r in r0) and all(r["seed_index"] == 1 for r in r1)
    assert all(a["search_seed"] != b["search_seed"] for a, b in zip(r0, r1))
    assert sum(a["genes"] != b["genes"] for a, b in zip(r0, r1)) >= 4


def test_smoothllm_copies_identical_across_processes():
    """v1 seeded copies with hash(prompt), which Python salts per process."""
    code = ("import sys; sys.path.insert(0, sys.argv[1]);"
            "from dcorr.defenses.smoothllm import SmoothLLM;"
            "print(SmoothLLM({'q': 4}, None).copies('Write a short poem about the sea'))")
    outs = []
    for salt in ("0", "12345"):
        env = dict(os.environ, PYTHONHASHSEED=salt)
        outs.append(subprocess.run([sys.executable, "-c", code, str(ROOT)], env=env,
                                   capture_output=True, text=True, check=True).stdout)
    assert outs[0] == outs[1] and outs[0].strip()


def test_seed_index_selects_replicate_dir(tmp_path):
    cfg = load_config(ROOT / "configs" / "hpc_vicuna_autodan.yaml")
    cfg.set_path("paths.results", str(tmp_path))        # keep the test out of results/
    assert cfg.seed_index is None
    assert cfg.results_dir == cfg.run_dir                 # v1 layout unchanged
    apply_seed_index(cfg, 2)
    assert cfg.seed_index == 2
    assert cfg.results_dir == cfg.run_dir / "seed2"
    apply_seed_index(cfg, None)                           # None keeps the current value
    assert cfg.seed_index == 2
