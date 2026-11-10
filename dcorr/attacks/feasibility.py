"""In-loop cheap feasibility, computed exactly as the deployed input filters compute it.

A candidate is feasible when every cheap input filter would admit it at deployment. The
deployed path (`Defense.respond` -> `filter_input`) scores the RAW prompt under the target's
DEFAULT system prompt. v1's attack loops departed from that for composed defenses (REVIEW F1):

  * AutoDAN checked feasibility on `objective_users(p)`, which for a stack containing
    SmoothLLM are four character-perturbed copies. No fluent prompt survives that, so the
    stack search found 0/100 feasible prompts in every stack run;
  * both attacks scored the filters' signals (window NLL, probe hidden state) with the
    objective's `score_kwargs`, i.e. under refusal-prime's system prompt inside a stack,
    which changes both quantities relative to deployment;
  * GCG skipped cheap feasibility entirely whenever any layer was an expensive constraint,
    so the 7-layer stack (which contains Llama Guard) had no input constraint at all.

`cheap_feasible_raw` is the single deployment-faithful check both attacks now call.
Expensive constraints (Llama Guard) are handled separately and are not checked here.
"""
from __future__ import annotations

from ..defenses.base import Defense


class _NoScore:
    """Stand-in for a ScoreOut when no filter needs a forward pass."""
    hidden = None
    window_nll = None


def needs_separate_pass(defense: Defense, raw_prompts: list[str],
                        users_per_prompt: list[list[str]]) -> bool:
    """True when the objective pass does not see what deployment's filters see."""
    if not defense.has_input_filter:
        return False
    if defense.score_kwargs():
        return True
    return any(u != [p] for p, u in zip(raw_prompts, users_per_prompt))


def cheap_feasible_raw(target, defense: Defense, raw_prompts: list[str],
                       target_str: str) -> list[bool]:
    """Cheap-constraint feasibility of each raw prompt, as deployed (default system prompt).

    Always applies the cheap layers, even when the defense also has an expensive one.
    """
    if not defense.has_input_filter:
        return [True] * len(raw_prompts)
    need_hidden = defense.needs_hidden()
    need_nll = defense.needs_window_nll()
    if need_hidden or need_nll:
        # No score_kwargs: deployment filters run under the target's default system prompt.
        out = target.score_chunked(raw_prompts, target_str, need_hidden=need_hidden,
                                   need_window_nll=need_nll)
    else:
        out = _NoScore()
    return [bool(f) for f in defense.cheap_feasible(out, raw_prompts)]
