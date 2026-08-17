"""Row 1 (token surface), instance A: windowed perplexity filter.

Flags a prompt if ANY sliding window of `window_tokens` user tokens has mean NLL above
the threshold, which is calibrated to 1% FPR on the benign set (stage 01). Scored under
the target model itself, over the user-content span of the templated prompt, so the
constraint rides along on the forward pass the attack objective already needs.

Cheapest defense here and the row most attacks defeat - a fluent suffix passes it.
"""
from __future__ import annotations

from .base import Defense, FilterOut


class PerplexityFilter(Defense):
    name = "ppl_filter"
    row = "token_surface"
    overhead_class = "B"
    has_input_filter = True
    expensive_constraint = False

    def __init__(self, cfg, target, run_cfg=None):
        super().__init__(cfg, target, run_cfg)
        self.window = int(self.cfg.get("window_tokens", 16))
        self.threshold = self.cfg.get("threshold")
        self.target.ppl_window_tokens = self.window

    def _require_threshold(self) -> float:
        if self.threshold is None:
            raise RuntimeError(
                "ppl_filter.threshold is unset - run scripts/01_calibrate.py first"
            )
        return float(self.threshold)

    def raw_scores(self, prompts: list[str]) -> list[float]:
        out = self.target.score_chunked(
            prompts, target_str="Sure", need_window_nll=True
        )
        return [float(x) for x in out.window_nll.tolist()]

    def filter_input(self, prompts: list[str]) -> FilterOut:
        thr = self._require_threshold()
        scores = self.raw_scores(prompts)
        return FilterOut(
            blocked=[s > thr for s in scores],
            score=scores,
            detail=[{"window_nll": s, "threshold": thr} for s in scores],
        )

    # ---- attack side: cheap constraint, read straight off the shared forward pass
    def needs_window_nll(self) -> bool:
        return True

    def cheap_feasible(self, score_out, prompts: list[str]) -> list[bool]:
        thr = self._require_threshold()
        if score_out.window_nll is None:
            return [True] * len(prompts)
        return [float(s) <= thr for s in score_out.window_nll.tolist()]
