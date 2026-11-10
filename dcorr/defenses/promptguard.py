"""Row 2 (semantic classification), instance B: Prompt Guard 2 86M jailbreak classifier.

The within-row partner for Llama Guard. 86M parameters (~0.35 GB fp16) so it stays
resident next to an fp16 7B target and its constraint is cheap enough to check inline
every iteration - unlike its row-mate, which is the interesting asymmetry: same row,
wildly different overhead, and H1 predicts they still fail together.
"""
from __future__ import annotations

import torch

from .base import Defense, FilterOut


class PromptGuard(Defense):
    name = "promptguard"
    row = "semantic_classification"
    overhead_class = "B"
    has_input_filter = True
    expensive_constraint = False

    def __init__(self, cfg, target, run_cfg=None):
        super().__init__(cfg, target, run_cfg)
        self.model_id = self.cfg.get("model_id", "meta-llama/Llama-Prompt-Guard-2-86M")
        self.threshold = float(self.cfg.get("threshold", 0.5))
        self.batch_size = int(self.cfg.get("batch_size", 32))
        self._model = None
        self._tok = None
        self._jailbreak_idx = None

    def _ensure(self):
        if self._model is not None:
            return
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        from ..compat import dtype_kwargs

        self._tok = AutoTokenizer.from_pretrained(self.model_id)
        m = AutoModelForSequenceClassification.from_pretrained(
            self.model_id, **dtype_kwargs("float16"))
        m.eval()
        m.requires_grad_(False)
        m.to("cuda" if torch.cuda.is_available() else "cpu")
        self._model = m
        # Prompt Guard 2 is binary: label 1 = malicious/jailbreak. Resolve by name when
        # the config provides id2label, else fall back to the last index.
        id2label = {int(k): str(v).lower() for k, v in (m.config.id2label or {}).items()}
        idx = next((k for k, v in id2label.items() if "jailbreak" in v or "malicious" in v
                    or v in ("label_1", "1")), None)
        self._jailbreak_idx = idx if idx is not None else (m.config.num_labels - 1)

    @torch.no_grad()
    def raw_scores(self, prompts: list[str]) -> list[float]:
        self._ensure()
        out: list[float] = []
        for i in range(0, len(prompts), self.batch_size):
            chunk = prompts[i : i + self.batch_size]
            enc = self._tok(chunk, return_tensors="pt", padding=True, truncation=True,
                            max_length=512).to(self._model.device)
            logits = self._model(**enc).logits.float()
            probs = torch.softmax(logits, dim=-1)[:, self._jailbreak_idx]
            out.extend([float(x) for x in probs.tolist()])
        return out

    def filter_input(self, prompts: list[str]) -> FilterOut:
        s = self.raw_scores(prompts)
        return FilterOut([x > self.threshold for x in s], s,
                         [{"threshold": self.threshold} for _ in s])

    def cheap_feasible(self, score_out, prompts: list[str]) -> list[bool]:
        return [x <= self.threshold for x in self.raw_scores(prompts)]

    def free(self):
        self._model = None
