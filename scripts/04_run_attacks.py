"""Stage 04: the grid. Every enabled defense, adaptive + static, harmful + benign.

For each defense: attack it (with knowledge of it) over the 100 harmful behaviours, and
also run the benign set through the deployed defense so H3 has per-defense false-refusal
data. Thresholds from stage 01 are merged over the config; probe weights from stage 02
are required for the probe defense.

Per-defense self-transfer: the adaptive attack seeds from the undefended suffixes found in
stage 03, so a defended run starts from a known-strong point rather than "! ! !".

Writes: results/<run>/stage04_<defense>.jsonl (harmful) and
        results/<run>/stage04_<defense>_benign.jsonl (benign)
Undefended reuses stage 03's file. NO gold judging here.
"""
from __future__ import annotations

import argparse

from _bootstrap import load_env  # noqa: E402

from dcorr.config import load_config, save_resolved
from dcorr.defenses import build_defenses
from dcorr.io_utils import append_jsonl, done_keys, read_json, set_all_seeds
from dcorr.run_defense import KEY_FIELDS, run_one_defense
from dcorr.runtime import build_target, behaviours_from, load_eval, load_gcg_suffix


def _merge_calibration(cfg) -> None:
    calib = read_json(cfg.results_dir / "calibration.json")
    if not calib:
        print("[04][WARN] no calibration.json - thresholded filters will error. Run stage 01.")
        return
    for name, thr in (calib.get("thresholds") or {}).items():
        if cfg.get_path(f"defenses.{name}.enabled"):
            cfg.set_path(f"defenses.{name}.threshold", thr)
    print(f"[04] merged calibrated thresholds: {calib.get('thresholds')}")


def _run_benign(target, defense, benign, out_path, max_new_tokens):
    """Deployed responses on the benign set (plain prompts) for H3 false-refusal rates."""
    model = target.model_id
    done = done_keys(out_path, ("model", "defense", "benign_id"))
    pending = [b for b in benign if (model, defense.name, b["benign_id"]) not in done]
    if not pending:
        return
    resp = defense.respond([b["prompt"] for b in pending], max_new_tokens=max_new_tokens)
    for b, r in zip(pending, resp):
        append_jsonl(out_path, {
            "model": model, "defense": defense.name, "row": defense.row,
            "benign_id": b["benign_id"], "prompt": b["prompt"],
            "response": r["response"], "raw_response": r["raw_response"],
            "blocked": r["blocked"], "blocked_stage": r["blocked_stage"],
            "input_score": r["input_score"], "output_score": r["output_score"],
        })


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/primary_llama2.yaml")
    ap.add_argument("--only", nargs="*", help="restrict to these defense names")
    ap.add_argument("--no-benign", action="store_true")
    args = ap.parse_args()

    load_env()
    cfg = load_config(args.config)
    set_all_seeds(int(cfg.get("seed", 0)))
    _merge_calibration(cfg)

    harmful, benign = load_eval(cfg)
    behaviours = behaviours_from(harmful)
    gcg = load_gcg_suffix(cfg)
    max_new = int(cfg.get_path("target.max_new_tokens", 256))
    attack_cfg = dict(cfg.get_path("attack.adaptive", {}), **cfg.get_path("attack.static", {}))

    target = build_target(cfg)
    defenses = build_defenses(cfg, target)
    transfer = cfg.results_dir / "stage04_undefended.jsonl"

    names = list(defenses)
    if args.only:
        names = [n for n in names if n in set(args.only)]

    # run_one_defense only persists rows once every behaviour it was given has finished, so a
    # crash near the end of a defense destroys the whole defense's work (we lost 40/50
    # behaviours to a late OOM this way). Feeding it the behaviour list in chunks makes the
    # unit of loss one chunk instead of one defense; resume already skips completed keys, so
    # chunking changes nothing about the results.
    chunk = int(cfg.get_path("attack.behaviour_chunk", 10) or len(behaviours))

    for name in names:
        d = defenses[name]
        out = cfg.results_dir / f"stage04_{name}.jsonl"
        print(f"[04] === {name} (row={d.row}) ===")
        for c0 in range(0, len(behaviours), chunk):
            run_one_defense(
                target, d, behaviours[c0 : c0 + chunk], out,
                attack_cfg=attack_cfg, gcg_suffix=gcg, max_new_tokens=max_new,
                seed=int(cfg.get("seed", 0)),
                transfer_from=transfer if transfer.exists() else None,
                progress=_mk_progress(name),
            )
            print(f"[04] {name}: persisted through behaviour {min(c0 + chunk, len(behaviours))}"
                  f"/{len(behaviours)}", flush=True)
        if not args.no_benign:
            _run_benign(target, d, benign, cfg.results_dir / f"stage04_{name}_benign.jsonl",
                        max_new)
        if hasattr(d, "free"):
            try:
                d.free()
            except Exception:
                pass

    save_resolved(cfg, cfg.results_dir / "resolved_config.yaml")
    print("[04] done.")
    target.free()


def _mk_progress(name):
    state = {"last": -1}

    def _p(t, total, states):
        pct = int(100 * t / total)
        if pct >= state["last"] + 20:
            state["last"] = pct
            feas = sum(1 for s in states if s.feas_suffix is not None)
            print(f"[04]   {name} adaptive {t}/{total}  feasible={feas}/{len(states)}")
    return _p


if __name__ == "__main__":
    main()
