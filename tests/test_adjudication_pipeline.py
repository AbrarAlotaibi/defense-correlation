"""End-to-end test of scripts/28_adjudication.py (sample, then score) on a synthetic run."""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


def _write(p: Path, rows):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def _run(*args):
    p = subprocess.run([sys.executable, str(REPO / "scripts" / "28_adjudication.py"), *args],
                       cwd=REPO, capture_output=True, text=True,
                       env=dict(os.environ, PYTHONIOENCODING="utf-8"))
    assert p.returncode == 0, p.stdout[-2000:] + p.stderr[-2000:]


@pytest.fixture()
def run(tmp_path):
    results = tmp_path / "results"
    cfg = tmp_path / "adj.yaml"
    cfg.write_text(json.dumps({
        "extends": str(REPO / "configs" / "hpc_vicuna_autodan.yaml").replace("\\", "/"),
        "run_name": "adj", "paths": {"results": str(results).replace("\\", "/")}}),
        encoding="utf-8")
    sd = results / "adj" / "seed0"
    # (behaviour, response, verdict_key, votes of 5, unparsed, breach)
    spec = [("jbb_0", "split answer", "k_split", 3, 0, True),
            ("jbb_1", "unparsed answer", "k_unp", 0, 1, False),
            ("jbb_2", "clear breach A", "k_b1", 5, 0, True),
            ("jbb_3", "clear breach B", "k_b2", 5, 0, True),
            ("jbb_4", "clear refusal A", "k_r1", 0, 0, False),
            ("jbb_5", "clear refusal B", "k_r2", 0, 0, False)]
    stage, gold = [], []
    for b, resp, k, votes, unp, br in spec:
        for d in ("ppl_filter", "probe"):              # same text under two defenses
            stage.append({"defense": d, "behaviour_id": b, "attack": "adaptive", "response": resp})
            gold.append({"defense": d, "behaviour_id": b, "attack": "adaptive", "judged": True,
                         "verdict_key": k, "reps": 5, "breach_votes": votes, "n_unparsed": unp,
                         "breach": br, "score": 0.9 if br else 0.0})
    _write(sd / "stage04_ppl_filter.jsonl", [r for r in stage if r["defense"] == "ppl_filter"])
    _write(sd / "stage04_probe.jsonl", [r for r in stage if r["defense"] == "probe"])
    _write(sd / "gold.jsonl", gold)
    return cfg, results / "adj" / "adjudication"


def _read(p: Path):
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_sample_is_blind_unique_and_stratified(run):
    cfg, out = run
    _run("sample", "--config", str(cfg), "--seeds", "0", "--n-random", "2")
    sheet, key = _read(out / "sheet.csv"), _read(out / "key.csv")
    assert len(sheet) == 4                                  # split + unparsed + 2 random
    # stratum lives in the key, never in the sheet: an annotator who sees that a row came
    # from the split or unparsed stratum knows the grader found it hard.
    assert "stratum" not in sheet[0]
    assert sorted(k["stratum"] for k in key) == ["random", "random", "split", "unparsed"]
    assert len({r["behaviour_id"] for r in sheet}) == 4      # duplicate texts appear once
    assert not any("grader" in c or "score" in c or "vote" in c for c in sheet[0])
    assert {k["item_id"] for k in key} == {r["item_id"] for r in sheet}
    rnd = [k for k in key if k["stratum"] == "random"]
    assert sorted(k["grader_breach"] for k in rnd) == ["0", "1"]   # half breach, half not
    assert all(r["response"] and r["behaviour"] for r in sheet)


def test_score_reports_hand_computed_values(run):
    cfg, out = run
    _run("sample", "--config", str(cfg), "--seeds", "0", "--n-random", "2")
    sheet = _read(out / "sheet.csv")
    key = {k["item_id"]: k for k in _read(out / "key.csv")}
    for r in sheet:                                         # A1 = grader; A2 flips the split item
        g = int(key[r["item_id"]]["grader_breach"])
        r["annotator_1_breach"] = str(g)
        stratum = key[r["item_id"]]["stratum"]
        r["annotator_2_breach"] = str(1 - g if stratum == "split" else g)
    with open(out / "sheet.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(sheet[0]))
        w.writeheader()
        w.writerows(sheet)
    _run("score", "--config", str(cfg))
    res = json.loads((out / "adjudication.json").read_text(encoding="utf-8"))
    o = res["overall"]
    # labels: A1 = grader = [1 (split), 0 (unparsed), 1, 0] in some order; A2 differs once
    assert o["n"] == 4 and o["agreement_annotators"] == pytest.approx(0.75)
    assert o["kappa_annotators"] == pytest.approx(0.5)      # po .75, pe .5
    assert o["agreement_a1_grader"] == pytest.approx(1.0)
    assert o["n_consensus"] == 3 and o["grader_vs_consensus"]["accuracy"] == pytest.approx(1.0)
    assert res["by_stratum"]["split"]["n_consensus"] == 0
