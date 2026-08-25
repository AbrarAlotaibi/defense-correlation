"""Stage 15 (D3): recalibrate the four thresholded filters on an EXTERNAL benign corpus.

Identical to scripts/01_calibrate.py in every respect except where the benign scores come
from: the threshold is still the (1 - target_fpr) quantile of the filter's benign score
distribution, computed with the same raw_scores() call on the same defense objects. Only the
prompt list differs, so any change in the thresholds is attributable to the corpus and not to
the procedure.

It also scores the in-sample evaluation benign set with the SAME defense objects in the same
process, so the two score distributions are directly comparable and the realised block rate
of the external threshold on the held-out JailbreakBench benign set is reported without a
second model load.

Writes results/<run>/calibration_external.json - deliberately NOT calibration.json, so
nothing downstream silently picks up the new thresholds and the reported runs stay
reproducible.

Usage:
  python scripts/15_calibrate_external.py --config configs/hpc_vicuna_autodan.yaml \
      --external data/external_benign_alpacaeval.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from _bootstrap import load_env  # noqa: E402

from dcorr.config import load_config
from dcorr.defenses import build_defense
from dcorr.io_utils import set_all_seeds, write_json
from dcorr.runtime import build_target, load_eval

FILTERS = ("ppl_filter", "token_anomaly", "probe", "probe_b")


def read_jsonl(p: Path) -> list[dict]:
    with open(p, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def quantile_threshold(scores, fpr: float) -> float:
    return float(np.quantile(np.asarray(scores, dtype=float), 1.0 - fpr))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--external", required=True, type=Path)
    ap.add_argument("--out-name", default="calibration_external.json")
    args = ap.parse_args()

    load_env()
    cfg = load_config(args.config)
    set_all_seeds(int(cfg.get("seed", 0)))
    fpr = float(cfg.get_path("calibration.target_fpr", 0.01))

    ext = read_jsonl(args.external)
    ext_prompts = [r["prompt"] for r in ext]
    _, benign = load_eval(cfg)
    in_prompts = [b["prompt"] for b in benign]
    print(f"[15] external corpus: {len(ext_prompts)} prompts from {args.external.name}")
    print(f"[15] in-sample set   : {len(in_prompts)} prompts (held out for evaluation)")
    print(f"[15] target FPR      : {fpr}  -> {fpr * len(ext_prompts):.0f} external prompts "
          f"vs {fpr * len(in_prompts):.0f} in-sample prompt(s)")

    target = build_target(cfg)
    old = json.loads((cfg.results_dir / "calibration.json").read_text(encoding="utf-8"))
    res = {"target_fpr": fpr, "model_id": cfg.get_path("target.model_id"),
           "external_corpus": str(args.external), "n_external": len(ext_prompts),
           "n_in_sample": len(in_prompts), "thresholds": {}, "comparison": {}}

    for name in FILTERS:
        if not cfg.get_path(f"defenses.{name}.enabled"):
            continue
        spec = dict(cfg.get_path(f"defenses.{name}", {}), threshold=0.5)
        d = build_defense(name, spec, target, cfg)
        if name in ("probe", "probe_b"):
            try:
                d.load()
            except RuntimeError as e:
                print(f"[15] skip {name}: {e}")
                continue

        print(f"[15] scoring {len(ext_prompts)} external prompts for {name} ...")
        s_ext = np.asarray(d.raw_scores(ext_prompts), dtype=float)
        print(f"[15] scoring {len(in_prompts)} in-sample prompts for {name} ...")
        s_in = np.asarray(d.raw_scores(in_prompts), dtype=float)

        thr_new = quantile_threshold(s_ext, fpr)
        thr_old = float(old["thresholds"][name])
        res["thresholds"][name] = thr_new
        res["comparison"][name] = {
            "threshold_external": thr_new,
            "threshold_in_sample": thr_old,
            "shift": thr_new - thr_old,
            # what each threshold rejects on each corpus
            "external_block_rate_at_external_thr": float(np.mean(s_ext > thr_new)),
            "heldout_block_rate_at_external_thr": float(np.mean(s_in > thr_new)),
            "heldout_block_rate_at_in_sample_thr": float(np.mean(s_in > thr_old)),
            "external_block_rate_at_in_sample_thr": float(np.mean(s_ext > thr_old)),
            "score_summary": {
                "external": {"mean": float(s_ext.mean()), "p50": float(np.quantile(s_ext, .5)),
                             "p99": float(np.quantile(s_ext, .99)), "max": float(s_ext.max())},
                "in_sample": {"mean": float(s_in.mean()), "p50": float(np.quantile(s_in, .5)),
                              "p99": float(np.quantile(s_in, .99)), "max": float(s_in.max())},
            },
        }
        c = res["comparison"][name]
        print(f"[15] {name}: external thr={thr_new:.5f} vs in-sample {thr_old:.5f} "
              f"(shift {c['shift']:+.5f}); held-out block rate "
              f"{c['heldout_block_rate_at_in_sample_thr']:.3f} -> "
              f"{c['heldout_block_rate_at_external_thr']:.3f}")

    out = cfg.results_dir / args.out_name
    write_json(out, res)
    target.free()
    print(f"[15] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
