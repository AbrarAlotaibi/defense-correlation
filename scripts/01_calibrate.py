"""Stage 01: calibrate every thresholded input filter to 1% FPR on the benign set.

Threshold = the (1 - target_fpr) quantile of the filter's benign score distribution, so
by construction it rejects target_fpr of benign prompts. Filters covered:
  * ppl_filter     windowed NLL under the target
  * token_anomaly  surface-statistics score (no model call)
  * probe          sigmoid(probe) at the mid layer (requires stage 02 weights)

Writes the thresholds back into a per-run calibration file that later stages load and
merge over the config. The config's own threshold: null is left as documentation of what
must be calibrated.
"""
from __future__ import annotations

import argparse

from _bootstrap import load_env  # noqa: E402

from dcorr.config import load_config, save_resolved
from dcorr.defenses import build_defense
from dcorr.io_utils import read_json, set_all_seeds, write_json
from dcorr.runtime import build_target, load_eval

import numpy as np

# Filters that need only the tokenizer (no model forward pass), so they calibrate on CPU.
_CPU_ONLY = {"token_anomaly"}
# Filters that need a GPU model forward pass / hidden states.
_NEEDS_MODEL = {"ppl_filter", "probe", "probe_b"}


def quantile_threshold(scores: list[float], fpr: float) -> float:
    if not scores:
        raise RuntimeError("no benign scores to calibrate on")
    return float(np.quantile(np.asarray(scores, dtype=float), 1.0 - fpr))


class _TokenizerOnlyTarget:
    """Minimal stand-in exposing just `.tokenizer` and `.model_id`, for CPU-only filters.

    Loads only the tokenizer (fast, CPU), never the 13 GB model, so token_anomaly can be
    calibrated while the GPU is busy with other work.
    """

    def __init__(self, model_id: str):
        from transformers import AutoTokenizer

        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        self.probe_layer = 0
        self.ppl_window_tokens = 16


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/primary_llama2.yaml")
    ap.add_argument("--only", nargs="*", default=None,
                    help="calibrate only these filters (e.g. token_anomaly for a CPU-only run)")
    args = ap.parse_args()

    load_env()
    cfg = load_config(args.config)
    set_all_seeds(int(cfg.get("seed", 0)))
    fpr = float(cfg.get_path("calibration.target_fpr", 0.01))

    _, benign = load_eval(cfg)
    benign_prompts = [b["prompt"] for b in benign]

    wanted = set(args.only) if args.only else {"ppl_filter", "token_anomaly", "probe", "probe_b"}
    model_id = cfg.get_path("target.model_id")

    # Only load the GPU model if a model-dependent filter is actually being calibrated.
    if wanted & _NEEDS_MODEL:
        target = build_target(cfg)
    else:
        print("[01] CPU-only calibration (tokenizer only, model not loaded).")
        target = _TokenizerOnlyTarget(model_id)

    # Accumulate into any existing calibration.json so partial (e.g. CPU-only) runs and a
    # later GPU run compose rather than clobber each other.
    results = read_json(cfg.results_dir / "calibration.json") or {}
    results.setdefault("target_fpr", fpr)
    results["model_id"] = model_id
    results.setdefault("thresholds", {})
    results.setdefault("fpr_check", {})

    for name in ("ppl_filter", "token_anomaly", "probe", "probe_b"):
        if name not in wanted:
            continue
        if not cfg.get_path(f"defenses.{name}.enabled"):
            continue
        spec = dict(cfg.get_path(f"defenses.{name}", {}))
        # Probe needs trained weights but not a threshold to *score*; give a dummy so it
        # constructs, then read raw scores.
        spec = dict(spec, threshold=0.5)
        d = build_defense(name, spec, target, cfg)
        if name in ("probe", "probe_b"):
            try:
                d.load()
            except RuntimeError as e:
                print(f"[01] skip probe calibration: {e}")
                continue
        print(f"[01] scoring benign for {name} ...")
        scores = d.raw_scores(benign_prompts)
        thr = quantile_threshold(scores, fpr)
        realized = float(np.mean([s > thr for s in scores]))
        results["thresholds"][name] = thr
        results["fpr_check"][name] = {"threshold": thr, "realized_fpr": realized,
                                      "n_benign": len(scores)}
        print(f"[01] {name}: threshold={thr:.5f}  realized benign FPR={realized:.3f}")

    out = cfg.results_dir / "calibration.json"
    write_json(out, results)
    save_resolved(cfg, cfg.results_dir / "resolved_config.yaml")
    print(f"[01] wrote {out}")
    if hasattr(target, "free"):
        target.free()


if __name__ == "__main__":
    main()
