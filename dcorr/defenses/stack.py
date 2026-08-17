"""The assembled stack: every enabled defense applied in the configured order.

Stage 05 attacks this directly. Stage 04 attacks each layer separately and stage 07
intersects the results. The intersection measures correlation; the direct attack measures
what composition rule 6 says you should measure. If they disagree, that disagreement is
itself a result and 07_analyze.py reports it as one rather than reconciling it away.

A stack is breached only if EVERY layer fails: any input filter blocking, any output
filter blocking, or the gold judge scoring below threshold, all mean non-breach.
"""
from __future__ import annotations

from .base import Defense, FilterOut, REFUSAL_TEXT


class DefenseStack(Defense):
    name = "stack"
    row = "stack"
    overhead_class = "mixed"

    def __init__(self, layers: list[Defense], order: list[str], target, run_cfg=None):
        super().__init__({"row": "stack"}, target, run_cfg)
        by_name = {d.name: d for d in layers}
        self.layers = [by_name[n] for n in order if n in by_name]
        missing = [d.name for d in layers if d.name not in {l.name for l in self.layers}]
        if missing:
            raise ValueError(f"defenses enabled but absent from stack.order: {missing}")

        self.input_layers = [d for d in self.layers if d.has_input_filter]
        self.output_layers = [d for d in self.layers if d.has_output_filter]
        self.gen_layers = [d for d in self.layers if d.modifies_generation]

        self.has_input_filter = bool(self.input_layers)
        self.has_output_filter = bool(self.output_layers)
        self.modifies_generation = bool(self.gen_layers)
        self.expensive_constraint = any(d.expensive_constraint for d in self.layers)

    # ------------------------------------------------------------------ deploy
    def filter_input(self, prompts: list[str]) -> FilterOut:
        blocked = [False] * len(prompts)
        detail = [{} for _ in prompts]
        for d in self.input_layers:
            fo = d.filter_input(prompts)
            for i, b in enumerate(fo.blocked):
                if b and not blocked[i]:
                    blocked[i] = True
                    detail[i] = {"blocked_by": d.name, "score": fo.score[i]}
        return FilterOut(blocked, [0.0] * len(prompts), detail)

    def filter_output(self, prompts: list[str], responses: list[str]) -> FilterOut:
        blocked = [False] * len(prompts)
        detail = [{} for _ in prompts]
        for d in self.output_layers:
            fo = d.filter_output(prompts, responses)
            for i, b in enumerate(fo.blocked):
                if b and not blocked[i]:
                    blocked[i] = True
                    detail[i] = {"blocked_by": d.name, "score": fo.score[i]}
        return FilterOut(blocked, [0.0] * len(prompts), detail)

    def system_for_generation(self):
        # Only one layer in the roster primes the system prompt; if that changes, the
        # composition of several system-prompt defenses has to be defined explicitly
        # rather than silently taking the last one.
        primers = [d for d in self.gen_layers if d.system_for_generation() != "__default__"]
        if len(primers) > 1:
            raise RuntimeError(
                f"more than one layer sets a system prompt: {[d.name for d in primers]} - "
                "define their composition explicitly before stacking them"
            )
        return primers[0].system_for_generation() if primers else "__default__"

    def transform_generate(self, prompts: list[str], max_new_tokens: int) -> list[str]:
        smoothers = [d for d in self.gen_layers if d.name == "smoothllm"]
        if smoothers:
            s = smoothers[0]
            # SmoothLLM's q copies must be generated under the stack's system prompt.
            prev = s.system_override
            s.system_override = self.system_for_generation()
            try:
                return s.transform_generate(prompts, max_new_tokens)
            finally:
                s.system_override = prev
        return self.target.generate(prompts, max_new_tokens=max_new_tokens,
                                    system=self.system_for_generation())

    # ------------------------------------------------------------------ attack
    def score_kwargs(self) -> dict:
        kw: dict = {}
        for d in self.gen_layers:
            kw.update(d.score_kwargs())
        return kw

    def objective_users(self, prompt: str) -> list[str]:
        for d in self.gen_layers:
            if d.name == "smoothllm":
                return d.objective_users(prompt)
        return [prompt]

    def needs_window_nll(self) -> bool:
        return any(d.needs_window_nll() for d in self.layers)

    def needs_hidden(self) -> bool:
        return any(d.needs_hidden() for d in self.layers)

    def cheap_feasible(self, score_out, prompts: list[str]) -> list[bool]:
        ok = [True] * len(prompts)
        for d in self.layers:
            if d.expensive_constraint:
                continue
            for i, f in enumerate(d.cheap_feasible(score_out, prompts)):
                ok[i] = ok[i] and f
        return ok

    def expensive_feasible(self, prompts: list[str]) -> list[bool]:
        ok = [True] * len(prompts)
        for d in self.layers:
            if not d.expensive_constraint:
                continue
            for i, f in enumerate(d.expensive_feasible(prompts)):
                ok[i] = ok[i] and f
        return ok

    def describe(self) -> dict:
        return {"name": "stack", "row": "stack", "overhead_class": "mixed",
                "layers": [d.describe() for d in self.layers]}
