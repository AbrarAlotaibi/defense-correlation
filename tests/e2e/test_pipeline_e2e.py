"""End-to-end checks of the v2 harness fixes on a real model, through the real stage scripts.

Nothing here is mocked. Each run invokes scripts/03, 04 or 05 as a subprocess (a fresh
Python process, so per-process hash salting is in play), on meta-llama/Llama-3.2-3B-Instruct
in fp16 with the real defenses and that model's real 1%-FPR thresholds, and the assertions
read the JSONL the pipeline writes. Four behaviours and a small search budget keep it to
minutes on one GPU.

What must hold (REVIEW F1, F2):
  1. defenses no longer replay one search; seed indices give different searches;
  2. a defense's search does not depend on chunking or on the process it ran in;
  3. a stack's in-loop feasibility equals what its deployed input filters admit, for the
     fluent attack and for GCG (an equivalence, so it cannot pass vacuously), and on stack B
     (see STACK_B) the fluent search finds prompts the deployed filters admit;
  4. the same stack-B run on the v1 code (tag v1-infus-submitted) finds none - proof that
     check 3 detects the bug.

Run: DCORR_E2E=1 python -m pytest tests/e2e -q -s
"""
from __future__ import annotations

import itertools
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (CALIBRATION, N_BEHAVIOURS, REPO, SINGLE, adaptive_rows,  # noqa: E402
                     config, extract_v1, feasibility_mismatches, run)

STACK_ORDER = ["ppl_filter", "token_anomaly", "probe", "refusal_prime", "smoothllm"]
# Stack B drops token-anomaly. At its 1%-FPR threshold on this model no fluent prompt within
# the test budget passes it (its own single-defense run admits 0/4; REVIEW F5), so a stack
# containing it has an empty feasible region and cannot tell v1 from v2. Stack B keeps the
# SmoothLLM + refusal-prime composition that triggered the v1 bug, and its other members each
# admit fluent prompts on their own, so "found a feasible prompt" is achievable and testable.
STACK_B = ["ppl_filter", "probe", "refusal_prime", "smoothllm"]


@pytest.fixture(scope="session")
def runs(tmp_path_factory):
    if not CALIBRATION.exists():
        pytest.skip(f"missing {CALIBRATION}")
    try:
        import torch
        if not torch.cuda.is_available():
            pytest.skip("no CUDA device")
    except ImportError:
        pytest.skip("torch not installed")

    tmp = tmp_path_factory.mktemp("e2e")
    results = tmp / "results"
    logs = tmp / "logs"
    logs.mkdir()

    def make(run_name: str, **kw) -> Path:
        (results / run_name).mkdir(parents=True, exist_ok=True)
        shutil.copy(CALIBRATION, results / run_name / "calibration.json")
        p = tmp / f"{run_name}.yaml"
        p.write_text(config(results, run_name, **kw), encoding="utf-8")
        return p

    main_cfg = make("e2e", stack_order=STACK_ORDER)
    chunk_cfg = make("e2e_chunk4", chunk=4, stack_order=STACK_ORDER)
    gcg_cfg = make("e2e_gcg", method="gcg", stack_order=STACK_ORDER)
    stackb_cfg = make("e2e_stackB", stack_order=STACK_B)
    v1_cfg = make("e2e_v1", stack_order=STACK_B)

    steps = [
        ("03_positive_control.py", main_cfg, ["--seed-index", "0"]),
        ("04_run_attacks.py", main_cfg, ["--seed-index", "0", "--no-benign", "--only", *SINGLE]),
        ("24_cross_eval.py", main_cfg, ["--seed-index", "0"]),
        ("12_undefended_benign.py", main_cfg, ["--seed-index", "0", "--benign-set", "orbench"]),
        ("04_run_attacks.py", main_cfg, ["--seed-index", "1", "--benign-only",
                                         "--benign-set", "orbench", "--only", "refusal_prime"]),
        ("04_run_attacks.py", main_cfg, ["--seed-index", "1", "--no-benign",
                                         "--only", "ppl_filter", "probe"]),
        ("05_run_stack.py", main_cfg, ["--seed-index", "0", "--no-benign"]),
        ("04_run_attacks.py", chunk_cfg, ["--seed-index", "0", "--no-benign",
                                          "--only", "ppl_filter", "smoothllm"]),
        ("05_run_stack.py", gcg_cfg, ["--seed-index", "0", "--no-benign"]),
        ("05_run_stack.py", stackb_cfg, ["--seed-index", "0", "--no-benign"]),
        ("05_run_stack.py", main_cfg, ["--seed-index", "0", "--no-benign",
                                       "--layers", "probe", "ppl_filter"]),
    ]
    for i, (script, cfg, args) in enumerate(steps):
        print(f"[e2e] {script} {cfg.name} {' '.join(args)}", flush=True)
        run(REPO, script, cfg, args, logs / f"{i:02d}_{script}.log")

    v1_code = extract_v1(tmp / "v1code")
    print("[e2e] v1 05_run_stack.py e2e_v1", flush=True)
    run(v1_code, "05_run_stack.py", v1_cfg, ["--no-benign"], logs / "v1_05_run_stack.log")

    return {"results": results, "logs": logs}


# ------------------------------------------------------------------------------- 0
def test_every_run_covered_every_behaviour(runs):
    r = runs["results"]
    files = [r / "e2e/seed0" / f"stage04_{d}.jsonl" for d in ["undefended", *SINGLE]]
    files += [r / "e2e/seed1/stage04_ppl_filter.jsonl", r / "e2e/seed0/stage05_stack.jsonl",
              r / "e2e_chunk4/seed0/stage04_smoothllm.jsonl",
              r / "e2e_gcg/seed0/stage05_stack.jsonl", r / "e2e_stackB/seed0/stage05_stack.jsonl",
              r / "e2e_v1/stage05_stack.jsonl"]
    for f in files:
        rows = adaptive_rows(f)
        assert len(rows) == N_BEHAVIOURS, f"{f}: {len(rows)} adaptive rows"
        if "e2e_v1" not in str(f):
            assert all(x.get("search_seed") is not None for x in rows.values()), f


# ------------------------------------------------------------------------------- 1
def test_defenses_do_not_replay_one_search(runs):
    d = runs["results"] / "e2e" / "seed0"
    names = ["undefended", "ppl_filter", "token_anomaly", "probe"]
    R = {n: adaptive_rows(d / f"stage04_{n}.jsonl") for n in names}
    for a, b in itertools.combinations(names, 2):
        same = sum(R[a][i]["prompt"] == R[b][i]["prompt"] for i in R[a])
        seeds_differ = all(R[a][i]["search_seed"] != R[b][i]["search_seed"] for i in R[a])
        assert seeds_differ, f"{a} x {b} share search seeds"
        assert same <= 1, f"{a} x {b}: identical final prompt on {same}/{N_BEHAVIOURS}"


def test_seed_index_gives_a_different_search(runs):
    r = runs["results"] / "e2e"
    for d in ["ppl_filter", "probe"]:
        s0 = adaptive_rows(r / "seed0" / f"stage04_{d}.jsonl")
        s1 = adaptive_rows(r / "seed1" / f"stage04_{d}.jsonl")
        assert all(s0[i]["seed_index"] == 0 and s1[i]["seed_index"] == 1 for i in s0)
        same = sum(s0[i]["prompt"] == s1[i]["prompt"] for i in s0)
        assert same <= 1, f"{d}: seed 0 and seed 1 identical on {same}/{N_BEHAVIOURS}"


# ------------------------------------------------------------------------------- 2
def test_search_independent_of_chunking_and_process(runs):
    r = runs["results"]
    for d in ["ppl_filter", "smoothllm"]:
        a = adaptive_rows(r / "e2e/seed0" / f"stage04_{d}.jsonl")          # chunks of 2
        b = adaptive_rows(r / "e2e_chunk4/seed0" / f"stage04_{d}.jsonl")   # chunks of 4
        diff = [i for i in a if a[i]["prompt"] != b[i]["prompt"]]
        assert not diff, f"{d}: chunking/process changed the search for {diff}"


# ------------------------------------------------------------------------------- 3
def test_fluent_stack_feasibility_equals_deployment(runs):
    for name in ["e2e", "e2e_stackB"]:
        rows = adaptive_rows(runs["results"] / name / "seed0" / "stage05_stack.jsonl")
        bad = feasibility_mismatches(rows)
        assert not bad, f"{name}: in-loop feasibility disagrees with the deployed stack:\n" \
            + "\n".join(bad)


def test_stack_b_members_admit_fluent_prompts(runs):
    """Precondition for the v1/v2 comparison: stack B's filtering members each admit
    fluent prompts alone, so its joint feasible region is not empty by construction."""
    d = runs["results"] / "e2e" / "seed0"
    for name in ["ppl_filter", "probe"]:
        rows = adaptive_rows(d / f"stage04_{name}.jsonl")
        assert sum(bool(x["found_feasible"]) for x in rows.values()) >= 1, name


def test_fluent_search_finds_prompts_stack_b_admits(runs):
    rows = adaptive_rows(runs["results"] / "e2e_stackB/seed0/stage05_stack.jsonl")
    assert sum(bool(x["found_feasible"]) for x in rows.values()) >= 1, \
        "the fluent search found no prompt stack B's deployed filters admit"


def test_gcg_stack_feasibility_equals_deployment(runs):
    rows = adaptive_rows(runs["results"] / "e2e_gcg/seed0/stage05_stack.jsonl")
    bad = feasibility_mismatches(rows)
    assert not bad, "GCG in-loop feasibility disagrees with the deployed stack:\n" + "\n".join(bad)


# ------------------------------------------------------------------------------- 5
CROSS_DEFS = ["undefended", *SINGLE]
PLAIN_GROUP = ["undefended", "ppl_filter", "token_anomaly", "probe"]


def _cross(runs, d):
    import json
    p = runs["results"] / "e2e/seed0/cross" / f"stage04_{d}.jsonl"
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def test_cross_eval_covers_every_source_and_defense(runs):
    for d in CROSS_DEFS:
        rows = _cross(runs, d)
        pairs = {(r["source"], r["behaviour_id"]) for r in rows}
        assert len(rows) == len(pairs) == len(CROSS_DEFS) * N_BEHAVIOURS, (d, len(rows))


def test_cross_eval_diagonal_reproduces_stage04(runs):
    """A defense on its own search's prompts must give stage 04's deployed response."""
    d0 = runs["results"] / "e2e" / "seed0"
    for d in CROSS_DEFS:
        own = adaptive_rows(d0 / f"stage04_{d}.jsonl")
        diag = [r for r in _cross(runs, d) if r["source"] == d]
        bad = [r["behaviour_id"] for r in diag
               if (r["response"], r["blocked_stage"]) != (own[r["behaviour_id"]]["response"],
                                                         own[r["behaviour_id"]]["blocked_stage"])]
        assert not bad, f"{d}: cross-eval disagrees with stage 04 on {bad}"


def test_cross_eval_shares_generation_within_the_plain_group(runs):
    """Same prompt, same deployed generation: admitted responses are byte-identical."""
    by = {d: {(r["source"], r["behaviour_id"]): r for r in _cross(runs, d)} for d in PLAIN_GROUP}
    for key in by["undefended"]:
        admitted = {by[d][key]["response"] for d in PLAIN_GROUP if not by[d][key]["blocked"]}
        assert len(admitted) <= 1, f"{key}: plain-group responses differ"


# ------------------------------------------------------------------------------- 4
def test_v1_code_reproduces_the_stack_bug(runs):
    """The check above must be able to fail: on v1 code the same stack finds nothing."""
    v1 = adaptive_rows(runs["results"] / "e2e_v1/stage05_stack.jsonl")
    v2 = adaptive_rows(runs["results"] / "e2e_stackB/seed0/stage05_stack.jsonl")
    v1_feasible = sum(bool(x["found_feasible"]) for x in v1.values())
    v2_feasible = sum(bool(x["found_feasible"]) for x in v2.values())
    print(f"[e2e] stack feasible prompts: v1={v1_feasible}/{N_BEHAVIOURS} v2={v2_feasible}/{N_BEHAVIOURS}")
    assert v1_feasible == 0, f"v1 found {v1_feasible} feasible stack prompts; bug not reproduced"
    assert v2_feasible > v1_feasible


# ------------------------------------------------------------------------------- 6
def test_orbench_benign_passes(runs):
    """Run D: the held-out OR-Bench-Hard set flows through stages 12 and 04 (benign only)."""
    import json
    r = runs["results"] / "e2e"
    for f in [r / "seed0/stage04_undefended_benign_orbench.jsonl",
              r / "seed1/stage04_refusal_prime_benign_orbench.jsonl"]:
        rows = [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert [x["benign_id"] for x in rows] == [f"orb_{i}" for i in range(N_BEHAVIOURS)], f
        assert all(x["response"] for x in rows)
    # --benign-only ran no attack: refusal_prime was never attacked under seed 1
    assert not (r / "seed1/stage04_refusal_prime.jsonl").exists()


# ------------------------------------------------------------------------------- 7
def test_pair_stack_is_a_joint_attack_on_just_that_pair(runs):
    """Plan 4.7: --layers attacks a subset jointly, in deployment order, under its own name."""
    import json
    d = runs["results"] / "e2e" / "seed0"
    f = d / "stage05_stack_ppl_filter+probe.jsonl"        # named in stack.order, not CLI order
    assert f.exists(), sorted(p.name for p in d.glob("stage05_*.jsonl"))
    rows = [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
    adaptive = {r["behaviour_id"]: r for r in rows if r["attack"] == "adaptive"}
    assert len(adaptive) == N_BEHAVIOURS
    assert {r["defense"] for r in rows} == {"stack_ppl_filter+probe"}, "rows must carry the subset"
    # in-loop feasibility still equals what the pair's deployed filters admit
    assert not feasibility_mismatches(adaptive)
    # the pair is a different system from the full stack: at least one prompt differs
    full = adaptive_rows(d / "stage05_stack.jsonl")
    assert any(adaptive[i]["prompt"] != full[i]["prompt"] for i in adaptive)
