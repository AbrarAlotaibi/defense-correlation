"""Task B: judge every response REPS times to measure grader variance.

The paper's intervals resample behaviours and therefore say nothing about the grader, yet a
single re-judge of 300 responses already flipped 19 verdicts. This runs the full corpus REPS
times so that (a) a majority vote gives a more stable label set and (b) the spread across
repetitions gives an interval that actually covers grader noise.

Mirrors scripts/06_judge_gold.py exactly - same source rows, same StrongRejectJudge, same
prompt, same temperature, same binarisation threshold. The ONLY change is that each response
is scored REPS times and every judgment is kept, tagged with `rep`.

Blocked rows are not sent to the judge (they cannot fail to parse and their verdict is fixed),
matching stage 06; they are written once with rep=0 and reused for every repetition.

Resumable: keyed on (model, defense, behaviour_id, attack, rep), so an interrupted run
continues without re-spending API calls.

Usage:
  python scripts/19_rejudge_multi.py --config configs/hpc_vicuna_autodan.yaml --reps 5
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from _bootstrap import load_env  # noqa: E402

from dcorr.config import load_config
from dcorr.io_utils import append_jsonl, read_jsonl
from dcorr.judge import StrongRejectJudge

KEY = ("model", "defense", "behaviour_id", "attack", "rep")


def _key(r: dict, rep: int) -> dict:
    return {"model": r["model"], "defense": r["defense"], "row": r.get("row"),
            "behaviour_id": r["behaviour_id"], "attack": r["attack"], "rep": rep}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--out-name", default="gold_multi.jsonl")
    a = ap.parse_args()

    load_env()
    cfg = load_config(a.config)
    rd = cfg.results_dir

    harmful = read_jsonl(Path(cfg.data_dir) / "eval_harmful.jsonl")
    by_id = {h["behaviour_id"]: h for h in harmful}

    src: list[dict] = []
    for p in sorted(rd.glob("stage04_*.jsonl")):
        if p.name.endswith("_benign.jsonl"):
            continue
        src.extend(read_jsonl(p))
    stack = rd / "stage05_stack.jsonl"
    if stack.is_file():
        src.extend(read_jsonl(stack))
    if not src:
        raise SystemExit(f"no stage04/05 rows under {rd}")

    out = rd / a.out_name
    done = set()
    if out.is_file():
        for r in read_jsonl(out):
            done.add(tuple(r.get(k) for k in KEY))

    judge = StrongRejectJudge(cfg.get_path("judge", {}))
    blocked = [r for r in src if r.get("blocked")]
    live = [r for r in src if not r.get("blocked")]
    print(f"[19] {len(src)} responses: {len(live)} to judge, {len(blocked)} blocked "
          f"(never sent to the grader)")
    print(f"[19] reps={a.reps} -> {len(live) * a.reps} judge calls at full cost, "
          f"model={judge.model}, threshold={judge.threshold}")

    # blocked rows: fixed verdict, written once
    for r in blocked:
        k = tuple(_key(r, 0)[x] for x in KEY)
        if k not in done:
            append_jsonl(out, {**_key(r, 0), "blocked": True, "score": 0.0, "breach": False,
                               "refused": None, "parse_ok": True, "judged": False})
            done.add(k)

    conc = int(cfg.get_path("judge.concurrency", 8))
    total_new = 0
    for rep in range(1, a.reps + 1):
        pending = [r for r in live
                   if tuple(_key(r, rep)[x] for x in KEY) not in done]
        print(f"[19] rep {rep}/{a.reps}: {len(pending)} pending")
        if not pending:
            continue

        def _judge(r: dict, _rep: int = rep) -> dict:
            beh = by_id[r["behaviour_id"]]["prompt"] if r["behaviour_id"] in by_id else ""
            v = judge.score_one(beh, r.get("response", ""))
            return {**_key(r, _rep), "blocked": False, "score": v.score,
                    "breach": bool(v.breach), "refused": v.refused,
                    "convincing": v.convincing, "specific": v.specific,
                    "parse_ok": v.parse_ok, "judged": True}

        n = 0
        with ThreadPoolExecutor(max_workers=conc) as ex:
            for verdict in ex.map(_judge, pending):
                append_jsonl(out, verdict)
                n += 1
                if n % 200 == 0:
                    print(f"[19]   rep {rep}: {n}/{len(pending)}", flush=True)
        total_new += n
        print(f"[19] rep {rep} done ({n} new verdicts)", flush=True)

    rows = read_jsonl(out)
    judged = [r for r in rows if r.get("judged")]
    print(f"[19] wrote {out}: {len(rows)} rows, {len(judged)} judged, "
          f"{total_new} new this session")
    print(json.dumps({"reps": a.reps, "judge_model": judge.model,
                      "threshold": judge.threshold}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
