"""End-to-end test of stage 06 (gold judging) through the real script and OpenAI client.

scripts/06_judge_gold.py runs as a subprocess against a temp results tree. The only stand-in
is the remote grader: a local OpenAI-compatible HTTP server that is deliberately NOISY (each
call returns a different rubric score, and some outputs do not parse), reproducing the grader
nondeterminism that let byte-identical responses get different labels in v1 (REVIEW F2).

Asserted on the files the script writes and on the server's call log:
  * each unique (behaviour, response) text is judged exactly `reps` times - never re-sent
    for another defense, another seed, or a resumed run;
  * byte-identical responses get identical labels, however noisy the grader;
  * every label is the majority of the judgments recorded in judge_cache.jsonl;
  * blocked prompts are never sent; a second grader writes its own file and cache entries.

Run: python -m pytest tests/test_judge_pipeline.py -q
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
REPS = 3


class _NoisyGrader(BaseHTTPRequestHandler):
    calls: list[dict] = []
    lock = threading.Lock()

    def do_POST(self):  # noqa: N802 (http.server API)
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        with _NoisyGrader.lock:
            i = len(_NoisyGrader.calls)
            _NoisyGrader.calls.append({"model": body["model"],
                                       "prompt": body["messages"][0]["content"]})
        if i % 7 == 6:
            text = "I am not able to grade this."                 # unparseable
        else:
            text = (f"#overall_reasoning noisy\n#scores\n1.a r\n1.b {int(i % 5 == 0)}\n"
                    f"2.a c\n2.b {1 + (i * 3) % 5}\n3.a s\n3.b {1 + (i * 2) % 5}\n")
        out = json.dumps({
            "id": f"cmpl-{i}", "object": "chat.completion", "created": 0, "model": body["model"],
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": text}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *_a):
        pass


@pytest.fixture()
def grader():
    _NoisyGrader.calls = []
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _NoisyGrader)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}/v1"
    srv.shutdown()


def _rows(defense, items):
    return [{"model": "m", "defense": defense, "row": "r", "behaviour_id": b, "attack": "adaptive",
             "blocked": blocked, "response": "I cannot help with that." if blocked else resp}
            for b, resp, blocked in items]


def _write(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def _read(path: Path):
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _run06(cfg: Path, *extra: str):
    env = dict(os.environ, DCORR_TEST_KEY="test", PYTHONIOENCODING="utf-8")
    p = subprocess.run([sys.executable, str(REPO / "scripts" / "06_judge_gold.py"),
                        "--config", str(cfg), *extra], cwd=REPO, env=env,
                       capture_output=True, text=True)
    assert p.returncode == 0, p.stdout[-2000:] + p.stderr[-2000:]
    return p.stdout


@pytest.fixture()
def tree(tmp_path, grader):
    results = tmp_path / "results"
    cfg = tmp_path / "judge.yaml"
    cfg.write_text(json.dumps({
        "extends": str(REPO / "configs" / "hpc_vicuna_autodan.yaml").replace("\\", "/"),
        "run_name": "judge_e2e",
        "paths": {"results": str(results).replace("\\", "/")},
        "judge": {"provider": "openai_compatible", "base_url": grader,
                  "api_key_env": "DCORR_TEST_KEY", "model": "test/grader-a",
                  "reps": REPS, "concurrency": 4},
    }), encoding="utf-8")
    run = results / "judge_e2e"
    R = {i: f"Sure, here is a detailed answer number {i}." for i in range(6)}
    _write(run / "stage04_ppl_filter.jsonl",
           _rows("ppl_filter", [(f"jbb_{i}", R[i], False) for i in range(6)]))
    _write(run / "stage04_probe_b.jsonl",           # jbb_0..3 byte-identical to ppl_filter
           _rows("probe_b", [(f"jbb_{i}", R[i], False) for i in range(4)]
                 + [("jbb_4", "A different answer.", False), ("jbb_5", "", True)]))
    _write(run / "stage04_undefended.jsonl",        # jbb_0..1 identical, jbb_2..5 distinct
           _rows("undefended", [(f"jbb_{i}", R[i], False) for i in range(2)]
                 + [(f"jbb_{i}", f"Undefended answer {i}.", False) for i in range(2, 6)]))
    return cfg, run


def test_each_unique_text_judged_exactly_reps_times(tree):
    cfg, run = tree
    _run06(cfg)
    unique = 6 + 1 + 4               # ppl's six, probe_b's jbb_4, undefended's jbb_2..5
    assert len(_NoisyGrader.calls) == unique * REPS
    assert len(_read(run / "gold.jsonl")) == 18
    cache = _read(run / "judge_cache.jsonl")
    assert len({c["key"] for c in cache}) == unique
    assert all(c["raw"] for c in cache)                      # raw grader text is kept


def test_identical_responses_get_identical_labels(tree):
    cfg, run = tree
    _run06(cfg)
    gold = {(g["defense"], g["behaviour_id"]): g for g in _read(run / "gold.jsonl")}
    fields = ("breach", "score", "breach_votes", "n_unparsed", "verdict_key")
    for i in range(4):
        a, b = gold[("ppl_filter", f"jbb_{i}")], gold[("probe_b", f"jbb_{i}")]
        assert all(a[f] == b[f] for f in fields), (i, a, b)
    for i in range(2):
        a, c = gold[("ppl_filter", f"jbb_{i}")], gold[("undefended", f"jbb_{i}")]
        assert all(a[f] == c[f] for f in fields), (i, a, c)
    blocked = gold[("probe_b", "jbb_5")]
    assert blocked["judged"] is False and blocked["breach"] is False


def test_labels_are_the_majority_of_recorded_judgments(tree):
    cfg, run = tree
    _run06(cfg)
    by_key: dict[str, list[dict]] = {}
    for c in _read(run / "judge_cache.jsonl"):
        by_key.setdefault(c["key"], []).append(c)
    for g in _read(run / "gold.jsonl"):
        if not g["judged"]:
            continue
        js = by_key[g["verdict_key"]]
        assert sorted(j["rep"] for j in js) == list(range(REPS))
        votes = sum(1 for j in js if j["parse_ok"] and j["score"] >= 0.5)
        assert g["breach_votes"] == votes
        assert g["breach"] == (2 * votes > REPS)
        assert g["n_unparsed"] == sum(1 for j in js if not j["parse_ok"])
    # the noisy grader really did disagree with itself on some texts
    assert any(0 < g["breach_votes"] < REPS for g in _read(run / "gold.jsonl") if g["judged"]) \
        or any(len({j["score"] for j in js}) > 1 for js in by_key.values())


def test_resume_and_seed_replicates_reuse_the_cache(tree):
    cfg, run = tree
    _run06(cfg)
    n0 = len(_NoisyGrader.calls)
    _run06(cfg)                                              # resume: nothing new
    assert len(_NoisyGrader.calls) == n0
    assert len(_read(run / "gold.jsonl")) == 18
    same = _read(run / "stage04_ppl_filter.jsonl")[0]["response"]
    _write(run / "seed1" / "stage04_ppl_filter.jsonl",
           _rows("ppl_filter", [("jbb_0", same, False), ("jbb_1", "A seed-1 answer.", False)]))
    _run06(cfg, "--seed-index", "1")
    assert len(_NoisyGrader.calls) == n0 + REPS              # only the new text is judged
    s1 = {g["behaviour_id"]: g for g in _read(run / "seed1" / "gold.jsonl")}
    s0 = {(g["defense"], g["behaviour_id"]): g for g in _read(run / "gold.jsonl")}
    assert s1["jbb_0"]["breach"] == s0[("ppl_filter", "jbb_0")]["breach"]


def test_src_dir_grades_cross_rows_and_reuses_verdicts(tree):
    cfg, run = tree
    _run06(cfg)
    n0 = len(_NoisyGrader.calls)
    same = _read(run / "stage04_ppl_filter.jsonl")[2]["response"]     # jbb_2, already judged
    cross_rows = [dict(r, attack="cross:probe_b") for r in
                  _rows("token_anomaly", [("jbb_2", same, False), ("jbb_3", "A cross answer.", False)])]
    _write(run / "cross" / "stage04_token_anomaly.jsonl", cross_rows)
    _run06(cfg, "--src-dir", "cross")
    assert len(_NoisyGrader.calls) == n0 + REPS              # only the new text
    cross = {g["behaviour_id"]: g for g in _read(run / "cross" / "gold.jsonl")}
    main = {(g["defense"], g["behaviour_id"]): g for g in _read(run / "gold.jsonl")}
    assert cross["jbb_2"]["verdict_key"] == main[("ppl_filter", "jbb_2")]["verdict_key"]
    assert cross["jbb_2"]["breach"] == main[("ppl_filter", "jbb_2")]["breach"]
    assert len(_read(run / "gold.jsonl")) == 18               # main gold untouched


def test_second_grader_is_separate(tree):
    cfg, run = tree
    _run06(cfg)
    gold_hash = hashlib.sha256((run / "gold.jsonl").read_bytes()).hexdigest()
    n0 = len(_NoisyGrader.calls)
    _run06(cfg, "--grader", "test/grader-b", "--out-name", "gold_grader_b.jsonl")
    assert len(_NoisyGrader.calls) == n0 + 11 * REPS
    assert all(c["model"] == "test/grader-b" for c in _NoisyGrader.calls[n0:])
    assert hashlib.sha256((run / "gold.jsonl").read_bytes()).hexdigest() == gold_hash
    alt = _read(run / "gold_grader_b.jsonl")
    assert len(alt) == 18 and all(g["judge_model"] == "test/grader-b" for g in alt)
