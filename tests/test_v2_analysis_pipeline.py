"""End-to-end test of scripts/26_v2_analysis.py on a synthetic results tree.

The tree holds hand-chosen breach vectors for two seeds, a stack, cross-evaluation rows and
benign responses; the script runs as a subprocess and every reported number is checked
against a value worked out by hand.

Run: python -m pytest tests/test_v2_analysis_pipeline.py -q
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
IDS = [f"jbb_{i}" for i in range(8)]
U = [1, 1, 1, 1, 0, 0, 0, 0]          # undefended and ppl_filter (identical, both seeds)
D2 = [1, 0, 1, 0, 1, 0, 1, 0]         # probe (both seeds); phi(U, D2) = 0
LP = [-0.1, -0.2, -0.3, -0.4, -1.0, -2.0, -3.0, -4.0]   # perfectly ranks U -> AUROC 1


def _gold(defense, vec, attack="adaptive", logprob=None, unparsed=()):
    return [{"model": "m", "defense": defense, "behaviour_id": i, "attack": attack,
             "breach": bool(b), "judged": True, "parse_ok": i not in unparsed,
             "n_unparsed": int(i in unparsed), "score": float(b),
             "in_loop_logprob": None if logprob is None else logprob[k]}
            for k, (i, b) in enumerate(zip(IDS, vec))]


def _write(p: Path, rows):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


@pytest.fixture()
def analysis(tmp_path):
    results = tmp_path / "results"
    run = results / "syn"
    cfg = tmp_path / "syn.yaml"
    cfg.write_text(json.dumps({
        "extends": str(REPO / "configs" / "hpc_vicuna_autodan.yaml").replace("\\", "/"),
        "run_name": "syn", "paths": {"results": str(results).replace("\\", "/")},
        "data": {"eval_limit": 8},
    }), encoding="utf-8")
    stacks = {0: [1, 0, 0, 0, 0, 0, 0, 0], 1: [0] * 8}
    for s in (0, 1):
        _write(run / f"seed{s}" / "gold.jsonl",
               _gold("undefended", U) + _gold("ppl_filter", U, logprob=LP, unparsed=("jbb_7",))
               + _gold("probe", D2) + _gold("stack", stacks[s])
               + _gold("stack_ppl_filter+probe", [1, 0, 1, 0, 0, 0, 0, 0])
               + _gold("ppl_filter", [0, 0, 0, 0, 1, 0, 0, 0], attack="static_plain"))
    _write(run / "seed0" / "cross" / "gold.jsonl",
           _gold("ppl_filter", U, "cross:probe") + _gold("probe", U, "cross:probe"))
    _write(run / "seed0" / "stage04_ppl_filter_benign.jsonl", [
        {"benign_id": "a", "response": "I cannot help with that request.", "blocked": True},
        {"benign_id": "b", "response": "I cannot do that.", "blocked": False},
        {"benign_id": "c", "response": "Here is a recipe for bread.", "blocked": False},
        {"benign_id": "d", "response": "Sure, the capital of France is Paris.", "blocked": False},
    ])
    _write(run / "seed0" / "stage04_ppl_filter_benign_orbench.jsonl", [
        {"benign_id": "orb_0", "response": "I cannot help with that request.", "blocked": True},
        {"benign_id": "orb_1", "response": "Here is how, step by step.", "blocked": False},
    ])
    p = subprocess.run([sys.executable, str(REPO / "scripts" / "26_v2_analysis.py"),
                        "--config", str(cfg), "--seeds", "0", "1", "--resamples", "200", "--splits", "20"],
                       cwd=REPO, capture_output=True, text=True,
                       env=dict(os.environ, PYTHONIOENCODING="utf-8"))
    assert p.returncode == 0, p.stdout[-2000:] + p.stderr[-2000:]
    res = json.loads((run / "analysis_v2.json").read_text(encoding="utf-8"))
    md = (run / "analysis_v2.md").read_text(encoding="utf-8")
    return res, md


def test_marginals_reliability_and_null(analysis):
    res, _ = analysis
    assert res["marginals"]["ppl_filter"]["mean_asr"] == pytest.approx(0.5)
    assert res["reliability"]["ppl_filter"]["mean"] == pytest.approx(1.0)
    assert res["null"]["mean"] == pytest.approx(1.0)


def test_e1_partial_and_disattenuated(analysis):
    res, _ = analysis
    e1 = res["e1"]["ppl_filter|probe"]
    assert e1["mean_phi"] == pytest.approx(0.0)
    assert e1["disattenuated"] == pytest.approx(0.0)
    assert set(res["e1"]) == {"ppl_filter|probe"}            # undefended and stack excluded


def test_e2_and_e3(analysis):
    res, _ = analysis
    assert res["e2"]["0"]["ppl_filter|probe"]["per_source"]["probe"] == pytest.approx(1.0)
    e3 = res["e3"]["0"]
    assert e3["intersection_asr"] == pytest.approx(2 / 8)    # U & D2 on jbb_0, jbb_2
    assert e3["stack_asr"] == pytest.approx(1 / 8)
    assert res["e3"]["1"]["stack_asr"] == 0.0


def test_frr_and_report(analysis):
    res, md = analysis
    f = res["frr"]["0"]["jbb"]["ppl_filter"]
    assert (f["frr"], f["n"]) == (pytest.approx(0.5), 4)     # 1 blocked + 1 refusal of 4
    o = res["frr"]["0"]["orbench"]["ppl_filter"]
    assert (o["frr"], o["n"]) == (pytest.approx(0.5), 2)
    assert "orbench set" in md
    assert "## E1 behaviour-level phi" in md and "ppl_filter|probe" in md


def test_extras(analysis):
    res, md = analysis
    x = res["extras"]["0"]
    assert x["tables_2x2"]["ppl_filter|probe"] == {"n11": 2, "n10": 2, "n01": 2, "n00": 2}
    assert x["best_of_strategies"]["ppl_filter"]["asr"] == pytest.approx(5 / 8)   # U or jbb_4
    assert x["objective_auroc"]["ppl_filter"] == pytest.approx(1.0)
    assert x["parse"]["ppl_filter"]["with_unparsed_vote"] == 1      # jbb_7, adaptive only
    assert x["nonpairwise"]["kappa"] == pytest.approx(0.0)          # U vs D2 independent
    assert "best-of-strategies ASR" in md


def test_difficulty_section(analysis):
    res, md = analysis
    assert "ppl_filter|probe" in res["difficulty"]["pairs"]
    assert res["difficulty"]["fit"]["n_obs"] == 2 * 3 * 8      # seeds x defenses x behaviours
    assert "How much is behaviour difficulty?" in md


def test_selection_section(analysis):
    res, md = analysis
    sel = res["selection"]["0"]["all_layers"]
    assert set(sel) >= {"greedy", "row", "random", "single_best", "splits"}
    assert "Held-out member selection" in md


def test_e3_subsets(analysis):
    """Plan 4.7: a jointly attacked pair is compared against its own members' intersection."""
    res, md = analysis
    sub = res["e3_subsets"]["stack_ppl_filter+probe"]["0"]
    assert sub["stack_asr"] == pytest.approx(2 / 8)
    assert sub["intersection_asr"] == pytest.approx(2 / 8)        # U & D2 breach jbb_0, jbb_2
    assert sub["stack_vs_intersection"]["p"] == pytest.approx(1.0)
    assert "jointly attacked subset: ppl_filter, probe" in md
    # the subset must not be treated as a layer in the pair analyses
    assert not any("stack" in k for k in res["e1"])
