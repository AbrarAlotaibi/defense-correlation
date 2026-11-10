"""End-to-end check that Llama Guard is inside the attack loop (REVIEW F4), on real models.

Runs the real stage scripts on meta-llama/Llama-3.2-3B-Instruct with the real
meta-llama/Llama-Guard-3-8B. v2 uses the `reload` residency (fp16, loaded per guard block
with the target evicted, layers beyond `max_gpu_memory` on CPU) - the configuration the 16 GB
card will use for the paper. v1 is run with the nf4-resident guard its code supports.

What must hold:
  1. v2: for Llama Guard alone and for a stack containing it, "found a feasible prompt"
     equals "the deployed system did not block it at input" - the guard is now part of
     feasibility;
  2. v2: the guard is loaded a handful of times per run (once per round plus the final
     checks), not once per behaviour;
  3. v1: the same equivalence FAILS - prompts reported feasible are blocked by the guard at
     deployment - proof that check 1 detects the bug.

Run: DCORR_E2E=1 python -m pytest tests/e2e/test_guard_e2e.py -q -s
"""
from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (CALIBRATION, N_BEHAVIOURS, REPO, adaptive_rows, config,  # noqa: E402
                     extract_v1, feasibility_mismatches, run)

ROUNDS = 2
BUDGET = {"generations": 4, "population": 6, "expensive_rounds": ROUNDS,
          "expensive_pool_k": 4, "final_top_m": 2}
GUARD_V2 = {"enabled": True, "residency": "reload", "load_in_4bit": False,
            "max_gpu_memory": "12GiB", "check_input": True, "check_output": True}
GUARD_V1 = {"enabled": True, "residency": "resident_gpu", "load_in_4bit": True,
            "check_input": True, "check_output": True}
STACK_LG = ["ppl_filter", "probe", "llamaguard", "refusal_prime", "smoothllm"]
# Per run: one load per round, plus at most two (input, output) for the final best-of-m
# respond() and two for run_one_defense's deployed respond().
MAX_LOADS = ROUNDS + 4


@pytest.fixture(scope="session")
def guard_runs(tmp_path_factory):
    if not CALIBRATION.exists():
        pytest.skip(f"missing {CALIBRATION}")
    try:
        import torch
        if not torch.cuda.is_available():
            pytest.skip("no CUDA device")
    except ImportError:
        pytest.skip("torch not installed")

    tmp = tmp_path_factory.mktemp("guard")
    results, logs = tmp / "results", tmp / "logs"
    logs.mkdir()

    def make(run_name: str, **kw) -> Path:
        (results / run_name).mkdir(parents=True, exist_ok=True)
        shutil.copy(CALIBRATION, results / run_name / "calibration.json")
        p = tmp / f"{run_name}.yaml"
        p.write_text(config(results, run_name, **kw), encoding="utf-8")
        return p

    lg_cfg = make("g_lg", adaptive=BUDGET, defenses={"llamaguard": GUARD_V2})
    stack_cfg = make("g_stack", adaptive=BUDGET, defenses={"llamaguard": GUARD_V2},
                     stack_order=STACK_LG)
    v1_cfg = make("g_v1", adaptive=BUDGET, defenses={"llamaguard": GUARD_V1})

    out = {"results": results}
    print("[guard-e2e] v2 04 llamaguard (reload, fp16)", flush=True)
    out["lg_log"] = run(REPO, "04_run_attacks.py", lg_cfg,
                        ["--seed-index", "0", "--no-benign", "--only", "llamaguard"],
                        logs / "v2_04_llamaguard.log")
    print("[guard-e2e] v2 05 stack with llamaguard", flush=True)
    out["stack_log"] = run(REPO, "05_run_stack.py", stack_cfg,
                           ["--seed-index", "0", "--no-benign"], logs / "v2_05_stack.log")
    v1_code = extract_v1(tmp / "v1code")
    print("[guard-e2e] v1 04 llamaguard (nf4)", flush=True)
    run(v1_code, "04_run_attacks.py", v1_cfg, ["--no-benign", "--only", "llamaguard"],
        logs / "v1_04_llamaguard.log")
    return out


def _loads(log: str) -> int:
    m = re.findall(r"llamaguard: guard loads = (\d+)", log)
    assert m, "the run did not report guard loads"
    return int(m[-1])


def test_v2_guard_alone_feasibility_equals_deployment(guard_runs):
    rows = adaptive_rows(guard_runs["results"] / "g_lg/seed0/stage04_llamaguard.jsonl")
    assert len(rows) == N_BEHAVIOURS
    assert all(r["guard_rounds"] == ROUNDS for r in rows.values())
    bad = feasibility_mismatches(rows)
    assert not bad, "Llama Guard feasibility disagrees with deployment:\n" + "\n".join(bad)
    print(f"[guard-e2e] v2 guard-feasible: {sum(r['found_feasible'] for r in rows.values())}"
          f"/{N_BEHAVIOURS}; checked per behaviour: "
          f"{[r['guard_checked'] for r in rows.values()]}")


def test_v2_stack_with_guard_feasibility_equals_deployment(guard_runs):
    rows = adaptive_rows(guard_runs["results"] / "g_stack/seed0/stage05_stack.jsonl")
    assert len(rows) == N_BEHAVIOURS
    assert all(r["guard_rounds"] == ROUNDS for r in rows.values())
    bad = feasibility_mismatches(rows)
    assert not bad, "stack-with-guard feasibility disagrees with deployment:\n" + "\n".join(bad)


def test_v2_guard_loaded_per_block_not_per_behaviour(guard_runs):
    for key in ["lg_log", "stack_log"]:
        n = _loads(guard_runs[key])
        print(f"[guard-e2e] {key}: guard loads = {n}")
        assert 1 <= n <= MAX_LOADS, f"{key}: {n} guard loads (expected 1..{MAX_LOADS})"


def test_v1_guard_runs_show_the_bug(guard_runs):
    """v1 never consulted the guard while searching: prompts it reports feasible are
    blocked by the guard at deployment, so the equivalence of test 1 fails on v1."""
    rows = adaptive_rows(guard_runs["results"] / "g_v1/stage04_llamaguard.jsonl")
    bad = feasibility_mismatches(rows)
    print(f"[guard-e2e] v1 mismatches: {len(bad)}/{N_BEHAVIOURS}")
    assert bad, "v1 showed no feasible-but-blocked prompt; the check would not detect F4"


def test_v2_search_speed_unchanged_after_guard_blocks(guard_runs):
    """Regression for the Reloadable leak: if the guard stayed resident after a block, the
    target spilled into shared system memory and later rounds ran several times slower."""
    for run, f in [("g_lg", "stage04_llamaguard.jsonl"), ("g_stack", "stage05_stack.jsonl")]:
        rows = adaptive_rows(guard_runs["results"] / run / "seed0" / f)
        secs = next(iter(rows.values()))["round_seconds"]
        print(f"[guard-e2e] {run}: round seconds {secs}")
        assert len(secs) == ROUNDS
        assert max(secs) <= 3 * min(secs), f"{run}: a round ran >3x slower than another: {secs}"
