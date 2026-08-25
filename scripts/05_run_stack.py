"""Stage 05: direct adaptive attack on the assembled stack.

This is the second, independent measurement. Stage 04 attacks each layer separately and
stage 07 intersects; stage 05 attacks the whole stack at once, which is what composition
rule 6 actually says to measure. Running both is deliberate: agreement corroborates the
independence decomposition, disagreement is a finding in its own right (07 reports it).

The attacker knows every layer: cheap constraints (perplexity, token anomaly, probe,
Prompt Guard) restrict the feasible set every iteration; the expensive one (Llama Guard)
is checked in round blocks; generation-modifying layers (refusal_prime, smoothllm) enter
the objective. All of that composition lives in dcorr/defenses/stack.py.

Writes: results/<run>/stage05_stack.jsonl (+ _benign for H3 on the stack).
"""
from __future__ import annotations

import argparse

from _bootstrap import load_env  # noqa: E402

from dcorr.config import load_config
from dcorr.defenses import build_defenses, build_stack
from dcorr.io_utils import append_jsonl, done_keys, read_json, set_all_seeds
from dcorr.run_defense import run_one_defense
from dcorr.runtime import build_target, behaviours_from, load_eval, load_gcg_suffix


def _merge_calibration(cfg) -> None:
    calib = read_json(cfg.results_dir / "calibration.json")
    if not calib:
        return
    for name, thr in (calib.get("thresholds") or {}).items():
        if cfg.get_path(f"defenses.{name}.enabled"):
            cfg.set_path(f"defenses.{name}.threshold", thr)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/primary_llama2.yaml")
    ap.add_argument("--no-benign", action="store_true")
    args = ap.parse_args()

    load_env()
    cfg = load_config(args.config)
    set_all_seeds(int(cfg.get("seed", 0)))
    _merge_calibration(cfg)

    harmful, benign = load_eval(cfg)
    behaviours = behaviours_from(harmful)
    max_new = int(cfg.get_path("target.max_new_tokens", 256))
    attack_cfg = dict(cfg.get_path("attack.adaptive", {}), **cfg.get_path("attack.static", {}))

    target = build_target(cfg)
    defenses = build_defenses(cfg, target)
    stack = build_stack(cfg, target, defenses)
    print(f"[05] stack order: {[d.name for d in stack.layers]}")

    out = cfg.results_dir / "stage05_stack.jsonl"
    transfer = cfg.results_dir / "stage04_undefended.jsonl"
    run_one_defense(
        target, stack, behaviours, out,
        attack_cfg=attack_cfg, gcg_suffix=load_gcg_suffix(cfg), max_new_tokens=max_new,
        seed=int(cfg.get("seed", 0)),
        transfer_from=transfer if transfer.exists() else None,
        progress=_mk_progress(),
    )

    if not args.no_benign:
        bout = cfg.results_dir / "stage05_stack_benign.jsonl"
        model = target.model_id
        done = done_keys(bout, ("model", "defense", "benign_id"))
        pend = [b for b in benign if (model, "stack", b["benign_id"]) not in done]
        if pend:
            resp = stack.respond([b["prompt"] for b in pend], max_new_tokens=max_new)
            for b, r in zip(pend, resp):
                append_jsonl(bout, {
                    "model": model, "defense": "stack", "row": "stack",
                    "benign_id": b["benign_id"], "prompt": b["prompt"],
                    "response": r["response"], "blocked": r["blocked"],
                    "blocked_stage": r["blocked_stage"],
                })

    print("[05] done.")
    target.free()


def _mk_progress():
    state = {"last": -1}

    def _p(t, total, states):
        pct = int(100 * t / total)
        if pct >= state["last"] + 20:
            state["last"] = pct
            feas = sum(1 for s in states if s.feas_suffix is not None)
            print(f"[05]   stack adaptive {t}/{total}  feasible={feas}/{len(states)}")
    return _p


if __name__ == "__main__":
    main()
