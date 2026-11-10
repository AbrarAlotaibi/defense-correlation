"""Stage 05: direct adaptive attack on the assembled stack.

This is the second, independent measurement. Stage 04 attacks each layer separately and
stage 07 intersects; stage 05 attacks the whole stack at once, which is what composition
rule 6 actually says to measure. Running both is deliberate: agreement corroborates the
independence decomposition, disagreement is a finding in its own right (07 reports it).

The attacker knows every layer: cheap constraints (perplexity, token anomaly, probe,
Prompt Guard) restrict the feasible set every iteration; the expensive one (Llama Guard)
is checked in round blocks; generation-modifying layers (refusal_prime, smoothllm) enter
the objective. All of that composition lives in dcorr/defenses/stack.py.

`--layers d1 d2 ...` attacks a SUBSET jointly instead of the configured stack, writing
stage05_stack_<d1>+<d2>.jsonl. That gives the end-to-end estimand at the pair level (plan
4.7; REVIEW R2-1, R5-1.2): the residual of a jointly attacked pair, against the residual the
same pair's separately measured breach vectors predict by intersection.

Writes: results/<run>/stage05_stack.jsonl (+ _benign for H3 on the stack).
"""
from __future__ import annotations

import argparse

from _bootstrap import load_env  # noqa: E402

from dcorr.gpulock import acquire_for_process
from dcorr.config import apply_seed_index, load_config
from dcorr.defenses import build_defenses, build_stack
from dcorr.io_utils import append_jsonl, done_keys, read_json, set_all_seeds
from dcorr.run_defense import run_one_defense
from dcorr.runtime import (benign_suffix, build_target, behaviours_from, load_benign, load_eval,
                           load_gcg_suffix)


def _merge_calibration(cfg) -> None:
    calib = read_json(cfg.run_dir / "calibration.json")
    if not calib:
        return
    for name, thr in (calib.get("thresholds") or {}).items():
        if cfg.get_path(f"defenses.{name}.enabled"):
            cfg.set_path(f"defenses.{name}.threshold", thr)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/primary_llama2.yaml")
    ap.add_argument("--seed-index", type=int, default=None,
                    help="attack replicate k: outputs go to results/<run>/seed<k>")
    ap.add_argument("--no-benign", action="store_true")
    ap.add_argument("--benign-set", default="jbb", choices=["jbb", "orbench"],
                    help="benign prompts for false-refusal rates (default: JBB benign)")
    ap.add_argument("--benign-only", action="store_true",
                    help="skip the attacks; only run the benign pass")
    ap.add_argument("--layers", nargs="+", default=None,
                    help="attack this subset jointly instead of the configured stack; "
                         "outputs stage05_stack_<layer>+<layer>.jsonl")
    args = ap.parse_args()

    load_env()

    acquire_for_process("stage 05 stack")   # one GPU job at a time
    cfg = load_config(args.config)
    apply_seed_index(cfg, args.seed_index)
    set_all_seeds(int(cfg.get("seed", 0)))
    _merge_calibration(cfg)

    harmful, _ = load_eval(cfg)
    benign = load_benign(cfg, args.benign_set)
    behaviours = behaviours_from(harmful)
    max_new = int(cfg.get_path("target.max_new_tokens", 256))
    attack_cfg = dict(cfg.get_path("attack.adaptive", {}), **cfg.get_path("attack.static", {}))

    tag = ""
    if args.layers:
        order = [d for d in cfg.get_path("stack.order", []) if d in set(args.layers)]
        missing = sorted(set(args.layers) - set(order))
        if missing:
            raise SystemExit(f"--layers names defenses absent from stack.order: {missing}")
        for name in list(cfg.get_path("defenses", {})):
            cfg.set_path(f"defenses.{name}.enabled", name in order)
        cfg.set_path("stack.order", order)
        tag = "_" + "+".join(order)       # deployment order, so the name is canonical

    target = build_target(cfg)
    defenses = build_defenses(cfg, target)
    stack = build_stack(cfg, target, defenses)
    # The rows (and therefore the grader's keys) carry the subset, so pair runs written into
    # one seed directory never collide with each other or with the full stack.
    stack.name = f"stack{tag}"
    print(f"[05] stack order: {[d.name for d in stack.layers]}", flush=True)

    out = cfg.results_dir / f"stage05_stack{tag}.jsonl"
    transfer = cfg.results_dir / "stage04_undefended.jsonl"
    if not args.benign_only:
        run_one_defense(
            target, stack, behaviours, out,
            attack_cfg=attack_cfg, gcg_suffix=load_gcg_suffix(cfg), max_new_tokens=max_new,
            seed=int(cfg.get("seed", 0)),
            seed_index=cfg.seed_index or 0,
            transfer_from=transfer if transfer.exists() else None,
            progress=_mk_progress(),
        )

    if not args.no_benign:
        bout = cfg.results_dir / f"stage05_stack{tag}{benign_suffix(args.benign_set)}.jsonl"
        model = target.model_id
        done = done_keys(bout, ("model", "defense", "benign_id"))
        pend = [b for b in benign if (model, "stack", b["benign_id"]) not in done]
        if pend:
            resp = stack.respond([b["prompt"] for b in pend], max_new_tokens=max_new)
            for b, r in zip(pend, resp):
                append_jsonl(bout, {
                    "model": model, "defense": f"stack{tag}", "row": "stack",
                    "benign_id": b["benign_id"], "prompt": b["prompt"],
                    "response": r["response"], "blocked": r["blocked"],
                    "blocked_stage": r["blocked_stage"],
                })

    for d in stack.layers:
        if getattr(d, "n_loads", None) is not None:
            print(f"[05] {d.name}: guard loads = {d.n_loads}", flush=True)
    print("[05] done.", flush=True)
    target.free()


def _mk_progress():
    state = {"last": -1}

    def _p(t, total, states):
        pct = int(100 * t / total)
        if pct >= state["last"] + 20:
            state["last"] = pct
            feas = sum(1 for s in states if s.feas_suffix is not None)
            print(f"[05]   stack adaptive {t}/{total}  feasible={feas}/{len(states)}",
                  flush=True)
    return _p


if __name__ == "__main__":
    main()
