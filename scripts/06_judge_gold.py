"""Stage 06: gold judging with StrongREJECT. POST HOC. Runs once over final responses.

Reads every stage04_*.jsonl and stage05_stack.jsonl, scores each final response with the
StrongREJECT autograder, and writes a per-response verdict keyed by
(model, defense, behaviour_id, attack). A blocked prompt is a non-breach without calling
the judge (there is nothing to score). Resumable and concurrent.

The gold judge is a DIFFERENT model from the target and is never reachable from the attack
loop - that separation is the whole point of the previous line of work's post-mortem.

Writes: results/<run>/gold.jsonl
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from _bootstrap import load_env  # noqa: E402

from dcorr.config import load_config
from dcorr.io_utils import append_jsonl, done_keys, read_jsonl
from dcorr.judge import StrongRejectJudge

KEY = ("model", "defense", "behaviour_id", "attack")


def _behaviour_text(harmful_by_id: dict, bid: str) -> str:
    r = harmful_by_id.get(bid, {})
    return r.get("prompt", "")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/primary_llama2.yaml")
    args = ap.parse_args()

    load_env()
    cfg = load_config(args.config)
    rd = cfg.results_dir

    harmful = read_jsonl(Path(cfg.data_dir) / "eval_harmful.jsonl")
    harmful_by_id = {h["behaviour_id"]: h for h in harmful}

    src_rows: list[dict] = []
    for p in sorted(rd.glob("stage04_*.jsonl")):
        if p.name.endswith("_benign.jsonl"):
            continue
        src_rows.extend(read_jsonl(p))
    src_rows.extend(read_jsonl(rd / "stage05_stack.jsonl"))
    if not src_rows:
        raise RuntimeError(f"no stage04/05 result rows under {rd} - run the attack stages first")

    gold_path = rd / "gold.jsonl"
    done = done_keys(gold_path, KEY)

    judge = StrongRejectJudge(cfg.get_path("judge", {}))

    pending = [r for r in src_rows
               if (r["model"], r["defense"], r["behaviour_id"], r["attack"]) not in done]
    print(f"[06] {len(src_rows)} responses, {len(pending)} to judge "
          f"(model={judge.model}, threshold={judge.threshold})")

    def _judge(r: dict) -> dict:
        if r.get("blocked"):
            # Blocked by a filter -> non-breach, no judge call.
            return {**_key(r), "blocked": True, "score": 0.0, "breach": False,
                    "refused": None, "parse_ok": True, "judged": False}
        beh = _behaviour_text(harmful_by_id, r["behaviour_id"])
        v = judge.score_one(beh, r.get("response", ""))
        return {**_key(r), "blocked": False, "score": v.score, "breach": bool(v.breach),
                "refused": v.refused, "convincing": v.convincing, "specific": v.specific,
                "parse_ok": v.parse_ok, "judged": True,
                "in_loop_logprob": r.get("in_loop_logprob")}

    conc = int(cfg.get_path("judge.concurrency", 8))
    with ThreadPoolExecutor(max_workers=conc) as ex:
        for verdict in ex.map(_judge, pending):
            append_jsonl(gold_path, verdict)

    all_gold = read_jsonl(gold_path)
    n_breach = sum(1 for g in all_gold if g.get("breach"))
    n_unparsed = sum(1 for g in all_gold if g.get("judged") and not g.get("parse_ok"))
    print(f"[06] gold done: {len(all_gold)} verdicts, {n_breach} breaches, "
          f"{n_unparsed} unparsed judge outputs -> {gold_path}")
    if n_unparsed:
        print("[06][WARN] some judge outputs did not parse; they count as non-breach. "
              "Inspect gold.jsonl rows with parse_ok=false.")


def _key(r: dict) -> dict:
    return {"model": r["model"], "defense": r["defense"],
            "behaviour_id": r["behaviour_id"], "attack": r["attack"], "row": r.get("row")}


if __name__ == "__main__":
    main()
