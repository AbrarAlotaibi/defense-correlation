"""Stage 12 (B3): the undefended model on the benign set.

Stage 03 runs the undefended control on the HARMFUL split only, so no run has ever recorded
what the bare model refuses on the benign prompts. Without that floor, the per-defense
false-refusal rates of H3 are raw rather than attributable: a defense showing 0.08 has not
been separated from a model that already refuses some of those prompts unprompted.

This writes results/<run>/stage04_undefended_benign.jsonl in exactly the schema
scripts/04_run_attacks.py::_run_benign produces, so scripts/07_analyze.py's frr() reads it
without change and the number is defined identically to every other FRR in the paper.

No attack search: 100 benign prompts, one greedy decode each.

Usage:
  python scripts/12_undefended_benign.py --config configs/hpc_vicuna_autodan.yaml
"""
from __future__ import annotations

import argparse

from _bootstrap import load_env  # noqa: E402

from dcorr.config import load_config
from dcorr.defenses import Undefended
from dcorr.io_utils import append_jsonl, done_keys, set_all_seeds
from dcorr.runtime import build_target, load_eval


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    load_env()
    cfg = load_config(args.config)
    set_all_seeds(int(cfg.get("seed", 0)))

    _, benign = load_eval(cfg)
    target = build_target(cfg)
    undef = Undefended({}, target, cfg)
    out = cfg.results_dir / "stage04_undefended_benign.jsonl"
    max_new = int(cfg.get_path("target.max_new_tokens", 256))

    model = target.model_id
    done = done_keys(out, ("model", "defense", "benign_id"))
    pending = [b for b in benign if (model, undef.name, b["benign_id"]) not in done]
    print(f"[12] undefended on benign: {len(pending)} pending of {len(benign)}")
    if pending:
        resp = undef.respond([b["prompt"] for b in pending], max_new_tokens=max_new)
        for b, r in zip(pending, resp):
            append_jsonl(out, {
                "model": model, "defense": undef.name, "row": undef.row,
                "benign_id": b["benign_id"], "prompt": b["prompt"],
                "response": r["response"], "raw_response": r["raw_response"],
                "blocked": r["blocked"], "blocked_stage": r["blocked_stage"],
                "input_score": r["input_score"], "output_score": r["output_score"],
            })
    target.free()
    print(f"[12] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
