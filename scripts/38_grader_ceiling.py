"""The grader ceiling: phi between two independent gradings of the SAME responses.

Reviewer R2-1/R5-1.1 asked for this to be reported beside the pairwise phi, so a reader can
see what phi = 1 looks like under this grader. Any measured phi has to be read against it: a
pair cannot exceed the agreement the grader has with itself.

Seed 0 was accidentally graded twice (two stage-06 processes ran concurrently), which left up
to ten judgments per unique text instead of five. Split them into two disjoint sets of five,
take the majority within each, and the two resulting breach vectors differ ONLY by grader
variability, because the responses are byte-identical.
"""
import json
import math
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
R = REPO / "results" / "v2_vicuna"
THR = 0.5


def phi(x, y):
    n = len(x)
    a = sum(1 for i in range(n) if x[i] and y[i])
    b = sum(1 for i in range(n) if x[i] and not y[i])
    c = sum(1 for i in range(n) if not x[i] and y[i])
    d = n - a - b - c
    den = math.sqrt((a + b) * (c + d) * (a + c) * (b + d))
    return float("nan") if den == 0 else (a * d - b * c) / den


# every judgment, including the duplicates
by_key = defaultdict(list)
for line in (R / "judge_cache_paired_raw.jsonl").read_text(encoding="utf-8").splitlines():
    if line.strip():
        r = json.loads(line)
        by_key[r["key"]].append(r)

both = {k: v for k, v in by_key.items() if len(v) >= 10}
print(f"unique texts with >=10 judgments: {len(both)} of {len(by_key)}")


def majority(js):
    votes = sum(1 for j in js if j["parse_ok"] and j["score"] >= THR)
    return votes * 2 > len(js)


# map each (defense, behaviour) of seed 0 to its verdict key, via the response text
import sys  # noqa: E402
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
from _bootstrap import load_env  # noqa: E402

load_env()
from dcorr.config import load_config                        # noqa: E402
from dcorr.judge.cache import verdict_key                   # noqa: E402
from dcorr.judge.strongreject import StrongRejectJudge      # noqa: E402

cfg = load_config(str(REPO / "configs" / "v2_vicuna.yaml"))
j = StrongRejectJudge(dict(cfg.get_path("judge")))
harmful = {}
for line in ((REPO / "data" / "eval_harmful.jsonl")
             .read_text(encoding="utf-8").splitlines()):
    if line.strip():
        r = json.loads(line)
        harmful[r["behaviour_id"]] = r["prompt"]

vecA, vecB = defaultdict(dict), defaultdict(dict)
missing = 0
for p in sorted((R / "seed0").glob("stage04_*.jsonl")):
    if p.name.endswith("_benign.jsonl"):
        continue
    dn = p.name[len("stage04_"):-len(".jsonl")]
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row["attack"] != "adaptive" or row.get("blocked"):
            continue
        k = verdict_key(model=j.model, temperature=j.temperature, max_tokens=j.max_tokens,
                        rubric_sha256=j.prompt_sha256,
                        behaviour=harmful.get(row["behaviour_id"], ""),
                        response=row.get("response") or "")
        js = both.get(k)
        if not js:
            missing += 1
            continue
        js = sorted(js, key=lambda r: (int(r["rep"]), r.get("raw", "")[:40]))
        vecA[dn][row["behaviour_id"]] = majority(js[:5])
        vecB[dn][row["behaviour_id"]] = majority(js[5:10])

print(f"rows without a doubly-graded key: {missing}\n")
print(f"{'defense':<16}{'n':>5}{'phi(grading A, grading B)':>28}{'disagree':>10}")
vals = []
for dn in sorted(vecA):
    ids = sorted(set(vecA[dn]) & set(vecB[dn]))
    if len(ids) < 20:
        continue
    a = [vecA[dn][i] for i in ids]
    b = [vecB[dn][i] for i in ids]
    v = phi(a, b)
    dis = sum(1 for i in range(len(a)) if a[i] != b[i])
    vals.append(v)
    print(f"{dn:<16}{len(ids):>5}{v:>28.3f}{dis/len(ids):>10.1%}")

import statistics  # noqa: E402
print(f"\nGRADER CEILING: mean phi over defenses = {statistics.mean(vals):.3f}")
print("A measured pair phi cannot be read as approaching 1; this is the practical maximum.")
