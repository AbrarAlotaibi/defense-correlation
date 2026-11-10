"""Row 1 (token surface), instance B: surface-statistics anomaly filter.

Exists so that H1 has a within-row pair to test. With one instance per row every pair is
cross-row and H1 is not testable at all.

Mechanically distinct from the perplexity filter - no model forward pass, pure surface
statistics - but drawing on the same signal (adversarial suffixes look unlike English at
the character/token level), which is exactly the dependency Table 6 row 1 asserts.

Score is the max of three normalised components, so a candidate must beat all three:
  * non-ASCII / non-printable character fraction
  * token-length entropy deficit (suffixes are dominated by rare short subword tokens)
  * repeated-token rate

Threshold calibrated to 1% FPR on the benign set (stage 01).
"""
from __future__ import annotations

import math
import re
from collections import Counter

from .base import Defense, FilterOut

_WORDish = re.compile(r"[A-Za-z]{2,}")


class TokenAnomalyFilter(Defense):
    name = "token_anomaly"
    row = "token_surface"
    overhead_class = "B"
    has_input_filter = True
    expensive_constraint = False

    def __init__(self, cfg, target, run_cfg=None):
        super().__init__(cfg, target, run_cfg)
        self.threshold = self.cfg.get("threshold")

    def _require_threshold(self) -> float:
        if self.threshold is None:
            raise RuntimeError(
                "token_anomaly.threshold is unset - run scripts/01_calibrate.py first"
            )
        return float(self.threshold)

    # ---- components ------------------------------------------------------
    def _components(self, prompt: str) -> dict:
        text = prompt or ""
        n_chars = max(1, len(text))
        nonascii = sum(1 for c in text if not (32 <= ord(c) < 127)) / n_chars

        ids = self.target.tokenizer.encode(text, add_special_tokens=False)
        n_tok = max(1, len(ids))
        pieces = [self.target.tokenizer.decode([i]) for i in ids]
        lens = [len(p.strip()) for p in pieces]
        # Entropy of the token-length distribution, normalised. Natural text spreads over
        # several lengths; a random-token suffix collapses onto short pieces.
        cnt = Counter(lens)
        ent = -sum((c / n_tok) * math.log((c / n_tok) + 1e-12) for c in cnt.values())
        max_ent = math.log(max(2, len(cnt)))
        entropy_deficit = 1.0 - (ent / max_ent if max_ent > 0 else 1.0)

        rep = 1.0 - (len(set(ids)) / n_tok)

        # Fraction of characters not inside a word-like run: high for symbol soup.
        wordchars = sum(len(m.group(0)) for m in _WORDish.finditer(text))
        nonword = 1.0 - (wordchars / n_chars)

        return {
            "nonascii": nonascii,
            "entropy_deficit": entropy_deficit,
            "repeat_rate": rep,
            "nonword_frac": nonword,
        }

    def raw_scores(self, prompts: list[str]) -> list[float]:
        return [self.score_one(p) for p in prompts]

    def score_one(self, prompt: str) -> float:
        c = self._components(prompt)
        # Max, not sum: the filter fires on the single most anomalous signal, so evading
        # it requires being unremarkable on all of them.
        return float(max(c["nonascii"], c["entropy_deficit"], c["repeat_rate"],
                         c["nonword_frac"]))

    def filter_input(self, prompts: list[str]) -> FilterOut:
        thr = self._require_threshold()
        scores = self.raw_scores(prompts)
        return FilterOut(
            blocked=[s > thr for s in scores],
            score=scores,
            detail=[dict(self._components(p), threshold=thr) for p in prompts],
        )

    # ---- attack side: pure CPU, so cheap even though it is not on the forward pass
    def cheap_feasible(self, score_out, prompts: list[str]) -> list[bool]:
        thr = self._require_threshold()
        return [self.score_one(p) <= thr for p in prompts]
