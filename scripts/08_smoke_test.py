"""Stage 08: shape check. 3 behaviours, 5 iterations, every enabled defense, tiny gen.

Runs the whole per-defense path (attack -> constraint -> deployed response) plus a stack
attack, on a handful of behaviours with a 5-iteration budget, so wiring, chat templating,
threshold merging, residency swaps, and the stack composition are all exercised in a
couple of minutes before the real grid is spent. Does NOT judge gold and writes to a
throwaway results dir (run_name + "_smoke").

Run this first. It should finish without error and report a feasible suffix for at least
the cheap defenses.
"""
from __future__ import annotations

import argparse

from _bootstrap import load_env  # noqa: E402

from dcorr.config import load_config
from dcorr.defenses import build_defenses, build_stack
from dcorr.io_utils import read_json, set_all_seeds
from dcorr.run_defense import run_one_defense
from dcorr.runtime import build_target, behaviours_from, load_eval, load_gcg_suffix


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/primary_llama2.yaml")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--iters", type=int, default=5)
    args = ap.parse_args()

    load_env()
    cfg = load_config(args.config)
    cfg["run_name"] = f"{cfg.get('run_name', 'run')}_smoke"
    set_all_seeds(int(cfg.get("seed", 0)))

    # Merge any real calibration if present; otherwise inject permissive thresholds so the
    # smoke test does not require stage 01 to have run.
    calib = read_json((cfg.results_dir.parent / cfg.get("run_name").replace("_smoke", "")
                       / "calibration.json"))
    for name in ("ppl_filter", "token_anomaly", "probe"):
        if cfg.get_path(f"defenses.{name}.enabled"):
            thr = None
            if calib:
                thr = (calib.get("thresholds") or {}).get(name)
            cfg.set_path(f"defenses.{name}.threshold", thr if thr is not None else 1e9)

    harmful, _ = load_eval(cfg)
    behaviours = behaviours_from(harmful)[: args.n]
    max_new = 48
    attack_cfg = dict(cfg.get_path("attack.adaptive", {}), **cfg.get_path("attack.static", {}))
    attack_cfg["iterations"] = args.iters
    attack_cfg["steps"] = args.iters   # GCG uses `steps`, not `iterations`; keep smoke fast

    target = build_target(cfg)
    defenses = build_defenses(cfg, target)
    gcg = load_gcg_suffix(cfg)

    for name, d in defenses.items():
        if name == "probe":
            try:
                d.load()
            except RuntimeError as e:
                print(f"[08] skip probe (no weights yet): {e}")
                continue
        print(f"[08] === {name} ===")
        rows = run_one_defense(
            target, d, behaviours, cfg.results_dir / f"smoke_{name}.jsonl",
            attack_cfg=attack_cfg, gcg_suffix=gcg, max_new_tokens=max_new,
            seed=int(cfg.get("seed", 0)),
        )
        adir = [r for r in rows if r["defense"] == name and r["attack"] == "adaptive"]
        feasible = sum(1 for r in adir if r.get("found_feasible"))
        blocked = sum(1 for r in adir if r.get("blocked"))
        print(f"[08]   {name}: {len(adir)} adaptive rows, {feasible} feasible, {blocked} blocked")

    # Stack path
    try:
        stack = build_stack(cfg, target, defenses)
        run_one_defense(
            target, stack, behaviours, cfg.results_dir / "smoke_stack.jsonl",
            attack_cfg=attack_cfg, gcg_suffix=gcg, max_new_tokens=max_new,
            seed=int(cfg.get("seed", 0)),
        )
        print(f"[08] stack path ok ({[d.name for d in stack.layers]})")
    except Exception as e:
        print(f"[08][WARN] stack path failed: {e}")

    print(f"[08] smoke complete -> {cfg.results_dir}")
    target.free()


if __name__ == "__main__":
    main()
