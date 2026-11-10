"""Stage 06: gold judging with StrongREJECT. POST HOC. Runs once over final responses.

Reads every stage04_*.jsonl and stage05_stack.jsonl, scores each final response with the
StrongREJECT autograder, and writes a per-response verdict keyed by
(model, defense, behaviour_id, attack). A blocked prompt is a non-breach without calling
the judge (there is nothing to score). Resumable and concurrent.

The gold judge is a DIFFERENT model from the target and is never reachable from the attack
loop - that separation is the whole point of the previous line of work's post-mortem.

v2: verdicts go through a cache keyed by the grader's identity and the exact text judged
(dcorr/judge/cache.py), so byte-identical responses get one label wherever they occur, and
each unique response is judged `judge.reps` times with the majority as the label. The cache
lives in results/<run>/ and is shared by every seed replicate. `--grader` and `--out-name`
run a second grader into its own file without touching gold.jsonl.

Writes: results/<run>[/seed<k>]/gold.jsonl (or --out-name)
        results/<run>/judge_cache.jsonl (every individual judgment, with raw grader text)
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from _bootstrap import load_env  # noqa: E402

from dcorr.config import apply_seed_index, load_config
from dcorr.io_utils import append_jsonl, done_keys, read_jsonl
from dcorr.judge import StrongRejectJudge
from dcorr.judge.cache import VerdictCache, majority, verdict_key

KEY = ("model", "defense", "behaviour_id", "attack")


def _behaviour_text(harmful_by_id: dict, bid: str) -> str:
    r = harmful_by_id.get(bid, {})
    return r.get("prompt", "")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/primary_llama2.yaml")
    ap.add_argument("--seed-index", type=int, default=None,
                    help="attack replicate k: outputs go to results/<run>/seed<k>")
    ap.add_argument("--grader", default=None,
                    help="override judge.model, e.g. for a second-grader robustness pass")
    ap.add_argument("--reps", type=int, default=None, help="override judge.reps")
    ap.add_argument("--out-name", default="gold.jsonl")
    ap.add_argument("--src-dir", default=None,
                    help="read stage rows from this subdirectory of the results dir and write "
                         "the gold file there (e.g. cross, for scripts/24_cross_eval.py)")
    args = ap.parse_args()

    load_env()
    cfg = load_config(args.config)
    apply_seed_index(cfg, args.seed_index)
    if args.grader:
        cfg.set_path("judge.model", args.grader)
    if args.reps is not None:
        cfg.set_path("judge.reps", args.reps)
    rd = cfg.results_dir / args.src_dir if args.src_dir else cfg.results_dir
    reps = int(cfg.get_path("judge.reps", 1))
    if reps < 1:
        raise ValueError("judge.reps must be >= 1")

    harmful = read_jsonl(Path(cfg.data_dir) / "eval_harmful.jsonl")
    harmful_by_id = {h["behaviour_id"]: h for h in harmful}

    src_rows: list[dict] = []
    for p in sorted(rd.glob("stage04_*.jsonl")):
        # Substring, not endswith("_benign.jsonl"): the held-out sets are written as
        # stage04_<member>_benign_<set>.jsonl, so an endswith test let the 400 OR-Bench rows
        # per member into the harmful grading set. They carry benign_id rather than
        # behaviour_id, so this raised a KeyError instead of silently grading 3,200 benign
        # prompts as attacks -- but only because of that field name.
        if "_benign" in p.name:
            continue
        src_rows.extend(read_jsonl(p))
    # stage05_stack.jsonl is the configured stack; stage05_stack_<a>+<b>.jsonl are the subsets
    # that 05_run_stack.py --layers writes (plan item 4.7). Globbing rather than naming the one
    # file, because a pair run that is never graded is invisible: the analysis simply does not
    # see that subset and reports nothing for it.
    for p in sorted(rd.glob("stage05_stack*.jsonl")):
        if "_benign" in p.name:
            continue
        src_rows.extend(read_jsonl(p))
    if not src_rows:
        raise RuntimeError(f"no stage04/05 result rows under {rd} - run the attack stages first")

    gold_path = rd / args.out_name
    done = done_keys(gold_path, KEY)

    judge = StrongRejectJudge(cfg.get_path("judge", {}))
    cache = VerdictCache(cfg.run_dir / "judge_cache.jsonl")

    pending = [r for r in src_rows
               if (r["model"], r["defense"], r["behaviour_id"], r["attack"]) not in done]

    def _key_of(r: dict) -> str:
        return verdict_key(model=judge.model, temperature=judge.temperature,
                           max_tokens=judge.max_tokens, rubric_sha256=judge.prompt_sha256,
                           behaviour=_behaviour_text(harmful_by_id, r["behaviour_id"]),
                           response=r.get("response", "") or "")

    # One grader call per (unique text, missing rep). Duplicates across defenses, seeds and
    # resumed runs are never re-sent.
    tasks: dict[tuple[str, int], dict] = {}
    for r in pending:
        if r.get("blocked"):
            continue
        k = _key_of(r)
        for rep in cache.missing_reps(k, reps):
            tasks.setdefault((k, rep), r)
    n_unique = len({_key_of(r) for r in pending if not r.get("blocked")})
    print(f"[06] {len(src_rows)} responses, {len(pending)} to label, {n_unique} unique texts, "
          f"{len(tasks)} grader calls (model={judge.model}, reps={reps}, "
          f"threshold={judge.threshold})")

    def _call(item: tuple[tuple[str, int], dict]) -> dict:
        (k, rep), r = item
        v = judge.score_one(_behaviour_text(harmful_by_id, r["behaviour_id"]),
                            r.get("response", "") or "")
        return {"key": k, "rep": rep, "model": judge.model, "score": v.score,
                "refused": v.refused, "convincing": v.convincing, "specific": v.specific,
                "parse_ok": v.parse_ok, "raw": v.raw}

    conc = int(cfg.get_path("judge.concurrency", 8))
    with ThreadPoolExecutor(max_workers=conc) as ex:
        for j in ex.map(_call, tasks.items()):
            cache.add(j)          # main thread only: the cache file has one writer

    for r in pending:
        if r.get("blocked"):
            # Blocked by a filter -> non-breach, no judge call.
            append_jsonl(gold_path, {**_key(r), "blocked": True, "score": 0.0, "breach": False,
                                     "refused": None, "parse_ok": True, "judged": False,
                                     "judge_model": judge.model})
            continue
        k = _key_of(r)
        js = [cache.reps_for(k)[i] for i in range(reps)]
        m = majority(js, judge.threshold)
        first = js[0]
        append_jsonl(gold_path, {
            **_key(r), "blocked": False, "score": m.score, "breach": m.breach,
            "breach_votes": m.breach_votes, "reps": m.reps, "n_unparsed": m.n_unparsed,
            "refused": first["refused"], "convincing": first["convincing"],
            "specific": first["specific"], "parse_ok": m.n_unparsed == 0, "judged": True,
            "judge_model": judge.model, "verdict_key": k,
            "in_loop_logprob": r.get("in_loop_logprob"),
        })

    all_gold = read_jsonl(gold_path)
    n_breach = sum(1 for g in all_gold if g.get("breach"))
    n_unparsed = sum(1 for g in all_gold if g.get("judged") and not g.get("parse_ok"))
    print(f"[06] gold done: {len(all_gold)} verdicts, {n_breach} breaches, "
          f"{n_unparsed} labels with at least one unparsed judgment -> {gold_path}")
    if n_unparsed:
        print("[06][WARN] some judge outputs did not parse; they count as non-breach votes. "
              "Inspect judge_cache.jsonl rows with parse_ok=false.")


def _key(r: dict) -> dict:
    return {"model": r["model"], "defense": r["defense"],
            "behaviour_id": r["behaviour_id"], "attack": r["attack"], "row": r.get("row")}


if __name__ == "__main__":
    main()
