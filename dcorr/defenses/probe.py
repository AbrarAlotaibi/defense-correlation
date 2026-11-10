"""Row 5 (internal representations): linear probe on mid-layer hidden states.

The class-E instance. Requires self-hosted weights: there is no API through which a
mid-layer residual stream can be read, which is exactly why this row is expensive.

Feature: the layer-L hidden state at the LAST PROMPT TOKEN of the templated prompt.
Causal attention makes that identical to a probe run on the prompt alone, so the
constraint rides along on the forward pass the attack objective already needs - the probe
is nearly free in the loop despite being class E at deployment.

Training data must be disjoint from the evaluation behaviours or H2 is contaminated.
scripts/02_train_probe.py enforces that and writes data/probe_dedup_report.json.
"""
from __future__ import annotations

from pathlib import Path

import torch

from .base import Defense, FilterOut


class LinearProbe(Defense):
    name = "probe"
    row = "internal_representations"
    overhead_class = "E"
    has_input_filter = True
    expensive_constraint = False
    # True: read the mid-layer state off the forward pass the attack objective already needs
    # (free). False: take an own pass at this probe's own layer, so a SECOND probe instance
    # at a different layer can coexist with the first (see LinearProbeB).
    use_shared_pass = True

    def __init__(self, cfg, target, run_cfg=None):
        super().__init__(cfg, target, run_cfg)
        self.threshold = self.cfg.get("threshold")
        self.weights_path = self.cfg.get("weights")
        self._w = None
        self._b = None
        self._mu = None
        self._sd = None
        self.layer = self.cfg.get("layer")
        if self.layer is not None and self.use_shared_pass:
            self.target.probe_layer = int(self.layer)

    # ------------------------------------------------------------------ weights
    def load(self) -> None:
        if self._w is not None:
            return
        if not self.weights_path:
            raise RuntimeError("probe.weights is unset")
        p = Path(self.weights_path)
        if not p.is_absolute():
            from ..config import REPO_ROOT

            p = REPO_ROOT / p
        if not p.exists():
            raise RuntimeError(
                f"probe weights not found at {p} - run scripts/02_train_probe.py first"
            )
        ck = torch.load(p, map_location="cpu")
        self._w = ck["w"].float()
        self._b = float(ck["b"])
        self._mu = ck["mu"].float()
        self._sd = ck["sd"].float()
        layer = int(ck["layer"])
        if self.layer is not None and int(self.layer) != layer:
            raise RuntimeError(
                f"probe.layer={self.layer} but weights were trained at layer {layer}"
            )
        self.layer = layer
        if self.use_shared_pass:
            self.target.probe_layer = layer
        if ck.get("model_id") != self.target.model_id:
            raise RuntimeError(
                f"probe weights were trained on {ck.get('model_id')!r}, target is "
                f"{self.target.model_id!r} - probes do not transfer across models"
            )

    def _require_threshold(self) -> float:
        if self.threshold is None:
            raise RuntimeError("probe.threshold is unset - run scripts/01_calibrate.py first")
        return float(self.threshold)

    def scores_from_hidden(self, hidden: torch.Tensor) -> list[float]:
        # Score in LOGIT space, not sigmoid: a high-AUROC probe saturates the sigmoid (scores
        # pile at 0/1), so the 1%-FPR benign quantile collapses to 1.0 and the threshold
        # blocks nothing. The logit keeps resolution, so calibration yields a real threshold.
        self.load()
        h = (hidden.float() - self._mu) / self._sd
        logit = h @ self._w + self._b
        return [float(x) for x in logit.tolist()]

    def raw_scores(self, prompts: list[str]) -> list[float]:
        # Pin the target to THIS probe's layer for the duration of the pass, so a second
        # probe instance at another layer cannot silently read the wrong hidden state.
        self.load()
        prev = self.target.probe_layer
        self.target.probe_layer = int(self.layer)
        try:
            out = self.target.score_chunked(prompts, target_str="Sure", need_hidden=True)
            return self.scores_from_hidden(out.hidden)
        finally:
            self.target.probe_layer = prev

    def filter_input(self, prompts: list[str]) -> FilterOut:
        thr = self._require_threshold()
        s = self.raw_scores(prompts)
        return FilterOut([x > thr for x in s], s, [{"threshold": thr, "layer": self.layer}
                                                   for _ in s])

    # ---- attack side: cheap, read straight off the shared forward pass
    def needs_hidden(self) -> bool:
        return True

    def cheap_feasible(self, score_out, prompts: list[str]) -> list[bool]:
        thr = self._require_threshold()
        if score_out.hidden is None:
            return [True] * len(prompts)
        return [x <= thr for x in self.scores_from_hidden(score_out.hidden)]


class LinearProbeB(LinearProbe):
    """Second instance of the internal-representations row: a probe at a DIFFERENT layer.

    Exists so H1 (same-row failure correlation) is testable at all. The pre-registered
    same-row pair was ppl_filter x token_anomaly, but the perplexity filter blocks 100% of
    the GCG attack, so its marginal ASR is 0 and phi is undefined. The probe row is the one
    with a healthy non-zero marginal, so a second probe - same mechanism class (reading the
    residual stream), different depth - gives a within-row pair with two live marginals.
    See PREREGISTRATION.md amendment.

    Takes its own forward pass at its own layer (`use_shared_pass = False`) so it cannot
    collide with the primary probe's layer during a stacked run.
    """

    name = "probe_b"
    row = "internal_representations"
    overhead_class = "E"
    use_shared_pass = False

    def needs_hidden(self) -> bool:
        # Does not consume the shared pass; it takes its own at self.layer.
        return False

    def cheap_feasible(self, score_out, prompts: list[str]) -> list[bool]:
        thr = self._require_threshold()
        return [x <= thr for x in self.raw_scores(prompts)]
