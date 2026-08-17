"""The uniform Defense interface.

A defense is three optional hooks plus an attack-facing view of itself:

  filter_input(prompts)            -> (blocked: list[bool], score: list[float])
  transform_generate(target, ps)   -> list[str]   (defaults to plain greedy decode)
  filter_output(prompts, responses)-> (blocked: list[bool], score: list[float])

A prompt that any filter blocks is a NON-BREACH by construction: the deployed system
returns a canned refusal, so there is nothing for the gold judge to score.

The attacker knows the defense. That enters in two ways, and a defense declares which:

  * `constraint`  - the input filter is a hard feasibility test on the candidate prompt.
                    Cheap constraints (perplexity, token anomaly, probe) ride along on the
                    forward pass the objective already needs. Expensive ones (Llama Guard)
                    set `expensive_constraint = True` and are evaluated in round-structured
                    blocks - see attacks/random_search.py.
  * `objective`   - the defense changes what the model sees or how it decodes, so the
                    logprob must be computed THROUGH it. `score_kwargs()` and
                    `objective_users()` express that.

Nothing in this file, or anything it calls, may consult a judge model.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

REFUSAL_TEXT = "I cannot help with that request."


@dataclass
class FilterOut:
    blocked: list[bool]
    score: list[float]
    detail: list[dict] = field(default_factory=list)


class Defense:
    name: str = "base"
    row: str = "unassigned"
    overhead_class: str = "B"

    # Attack-facing declarations
    has_input_filter: bool = False
    has_output_filter: bool = False
    expensive_constraint: bool = False   # True -> feasibility deferred to round blocks
    modifies_generation: bool = False

    def __init__(self, cfg: dict, target, run_cfg=None):
        self.cfg = cfg or {}
        self.target = target
        self.run_cfg = run_cfg
        self.row = self.cfg.get("row", self.row)

    # ---------------------------------------------------------------- deploy side
    def filter_input(self, prompts: list[str]) -> FilterOut:
        return FilterOut([False] * len(prompts), [0.0] * len(prompts), [{}] * len(prompts))

    def filter_output(self, prompts: list[str], responses: list[str]) -> FilterOut:
        return FilterOut([False] * len(prompts), [0.0] * len(prompts), [{}] * len(prompts))

    def transform_generate(self, prompts: list[str], max_new_tokens: int) -> list[str]:
        """Produce the response the deployed system would return, ignoring filters."""
        return self.target.generate(prompts, max_new_tokens=max_new_tokens,
                                    system=self.system_for_generation())

    def system_for_generation(self) -> Any:
        """System prompt used at generation time; '__default__' means the target's own."""
        return "__default__"

    def respond(self, prompts: list[str], max_new_tokens: int) -> list[dict]:
        """Full deployed behaviour: input filter -> generate -> output filter."""
        fin = self.filter_input(prompts) if self.has_input_filter else \
            FilterOut([False] * len(prompts), [0.0] * len(prompts), [{}] * len(prompts))

        idx_pass = [i for i, b in enumerate(fin.blocked) if not b]
        responses = [REFUSAL_TEXT] * len(prompts)
        if idx_pass:
            gen = self.transform_generate([prompts[i] for i in idx_pass], max_new_tokens)
            for j, i in enumerate(idx_pass):
                responses[i] = gen[j]

        fout = self.filter_output(prompts, responses) if self.has_output_filter else \
            FilterOut([False] * len(prompts), [0.0] * len(prompts), [{}] * len(prompts))

        rows = []
        for i in range(len(prompts)):
            blocked_in = bool(fin.blocked[i])
            blocked_out = bool(fout.blocked[i]) and not blocked_in
            blocked = blocked_in or blocked_out
            rows.append({
                "response": REFUSAL_TEXT if blocked else responses[i],
                "raw_response": responses[i],
                "blocked": blocked,
                "blocked_stage": "input" if blocked_in else ("output" if blocked_out else None),
                "input_score": float(fin.score[i]),
                "output_score": float(fout.score[i]),
            })
        return rows

    # ---------------------------------------------------------------- attack side
    def score_kwargs(self) -> dict:
        """Extra kwargs for HFTarget.score() so the objective is computed through us."""
        return {}

    def objective_users(self, prompt: str) -> list[str]:
        """Prompt variants the objective averages over (SmoothLLM returns q copies)."""
        return [prompt]

    def needs_window_nll(self) -> bool:
        return False

    def needs_hidden(self) -> bool:
        return False

    def cheap_feasible(self, score_out, prompts: list[str]) -> list[bool]:
        """Feasibility from quantities already on the forward pass. Cheap constraints only."""
        return [True] * len(prompts)

    def expensive_feasible(self, prompts: list[str]) -> list[bool]:
        """Feasibility for `expensive_constraint` defenses, called in round blocks."""
        return [True] * len(prompts)

    def describe(self) -> dict:
        return {
            "name": self.name,
            "row": self.row,
            "overhead_class": self.overhead_class,
            "config": {k: v for k, v in self.cfg.items() if k != "enabled"},
        }
