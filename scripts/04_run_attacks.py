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
import json
import time

from _bootstrap import load_env  # noqa: E402

from dcorr.gpulock import acquire_for_process
from dcorr.config import apply_seed_index, load_config, save_resolved
from dcorr.defenses import build_defenses
from dcorr.io_utils import append_jsonl, done_keys, read_json, set_all_seeds
from dcorr.run_defense import KEY_FIELDS, run_one_defense
from dcorr.runtime import (benign_suffix, build_target, behaviours_from, host_provenance,
                           load_benign, load_eval, load_gcg_suffix)
from dcorr.target import HFTarget


def _merge_calibration(cfg) -> None:
    calib = read_json(cfg.run_dir / "calibration.json")
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
    ap.add_argument("--seed-index", type=int, default=None,
                    help="attack replicate k: outputs go to results/<run>/seed<k>")
    ap.add_argument("--only", nargs="*", help="restrict to these defense names")
    ap.add_argument("--no-benign", action="store_true")
    ap.add_argument("--benign-set", default="jbb", choices=["jbb", "orbench"],
                    help="benign prompts for false-refusal rates (default: JBB benign)")
    ap.add_argument("--benign-only", action="store_true",
                    help="skip the attacks; only run the benign pass")
    args = ap.parse_args()

    load_env()

    acquire_for_process("stage 04 attacks")   # one GPU job at a time
    cfg = load_config(args.config)
    apply_seed_index(cfg, args.seed_index)
    set_all_seeds(int(cfg.get("seed", 0)))
    _merge_calibration(cfg)

    harmful, _ = load_eval(cfg)
    benign = load_benign(cfg, args.benign_set)
    behaviours = behaviours_from(harmful)
    gcg = load_gcg_suffix(cfg)
    max_new = int(cfg.get_path("target.max_new_tokens", 256))
    attack_cfg = dict(cfg.get_path("attack.adaptive", {}), **cfg.get_path("attack.static", {}))

    target = build_target(cfg)
    defenses = build_defenses(cfg, target)
    transfer = cfg.results_dir / "stage04_undefended.jsonl"

    # The second attack family's attacker model, built ONCE for the whole stage: it is a
    # 7B model, and loading it per defense (let alone per chunk) would cost more than the
    # searches. None unless attack.pair.enabled, so every configuration that predates this
    # behaves exactly as before.
    pair_attacker = None
    pair_cfg = dict(cfg.get_path("attack.pair", {}) or {})
    if pair_cfg.get("enabled") and not args.benign_only:
        attacker_id = pair_cfg.get("attacker_model_id")
        if not attacker_id:
            raise SystemExit("[04] attack.pair.enabled needs attack.pair.attacker_model_id")
        if attacker_id == cfg.get_path("target.model_id"):
            # An attacker reading its own refusals is a different and much easier problem,
            # and would make the family comparison a comparison of two different setups.
            raise SystemExit(
                f"[04] attack.pair.attacker_model_id is the target ({attacker_id}). The "
                "attacker must be a different model, or the arm is self-attack.")
        print(f"[04] loading the PAIR attacker {attacker_id}", flush=True)
        pair_attacker = HFTarget(
            attacker_id,
            dtype=pair_cfg.get("attacker_dtype", "float16"),
            system_prompt=None,
            gen_batch_size=int(pair_cfg.get("attacker_gen_batch_size", 1)),
        )

    names = list(defenses)
    if args.only:
        names = [n for n in names if n in set(args.only)]

    # run_one_defense only persists rows once every behaviour it was given has finished, so a
    # crash near the end of a defense destroys the whole defense's work (we lost 40/50
    # behaviours to a late OOM this way). Feeding it the behaviour list in chunks makes the
    # unit of loss one chunk instead of one defense; resume already skips completed keys, so
    # chunking changes nothing about the results.
    chunk = int(cfg.get_path("attack.behaviour_chunk", 10) or len(behaviours))
    # A defense with an expensive constraint (Llama Guard) is searched in lockstep rounds,
    # loading the guard once per round PER CALL; chunks of 10 would multiply the loads by 10.
    # It gets its own, larger chunk (the trade: more work at risk if a chunk crashes).
    expensive_chunk = int(cfg.get_path("attack.expensive_behaviour_chunk", 50)
                          or len(behaviours))

    for name in names:
        d = defenses[name]
        out = cfg.results_dir / f"stage04_{name}.jsonl"
        print(f"[04] === {name} (row={d.row}) ===")
        t_defense, had_rows = time.time(), out.exists()
        step = expensive_chunk if d.expensive_constraint else chunk
        for c0 in (range(0, len(behaviours), step) if not args.benign_only else []):
            run_one_defense(
                target, d, behaviours[c0 : c0 + step], out,
                attack_cfg=attack_cfg, gcg_suffix=gcg, max_new_tokens=max_new,
                seed=int(cfg.get("seed", 0)),
                seed_index=cfg.seed_index or 0,
                transfer_from=transfer if transfer.exists() else None,
                pair_attacker=pair_attacker,
                pair_cfg=pair_cfg if pair_attacker is not None else None,
                progress=_mk_progress(name),
            )
            print(f"[04] {name}: persisted through behaviour {min(c0 + step, len(behaviours))}"
                  f"/{len(behaviours)}", flush=True)
        if not args.no_benign:
            _run_benign(target, d, benign, cfg.results_dir / f"stage04_{name}{benign_suffix(args.benign_set)}.jsonl",
                        max_new)
        # Per-defense wall clock, appended as its own record: file timestamps cannot be used
        # for this, because a resumed run leaves the earlier defenses' files untouched.
        rec = {"stage": "04", "defense": name, "seed_index": cfg.seed_index or 0,
               "seconds": round(time.time() - t_defense, 1),
               "n_behaviours": len(behaviours), "resumed": had_rows,
               "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
               # Which machine and card produced these rows. A resumed run appends a record
               # per chunk, so a result set assembled across nodes shows every node it
               # touched rather than only the last.
               **host_provenance()}
        if getattr(d, "n_loads", None) is not None:
            rec["guard_loads"] = d.n_loads
            print(f"[04] {name}: guard loads = {d.n_loads}", flush=True)
        with open(cfg.run_dir / "defense_timings.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + chr(10))
        print(f"[04] {name}: {rec['seconds'] / 60:.1f} min", flush=True)
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
