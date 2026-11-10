"""Static baselines, run against every defense.

(a) the unmodified behaviour prompt
(b) one precomputed GCG suffix, fixed for the whole run - a TRANSFER suffix, not one
    optimised against this target or against any defense

Reporting adaptive and static side by side gives the adaptive-vs-static gap, which is the
quantity Section 10 argues the field systematically ignores. The static attacker is
defense-oblivious by construction: that is the point of the comparison, so nothing here
consults the defense.
"""
from __future__ import annotations

from .random_search import Behaviour


def build_static(behaviours: list[Behaviour], gcg_suffix: str,
                 include_plain: bool = True, include_gcg: bool = True) -> list[dict]:
    rows: list[dict] = []
    for b in behaviours:
        if include_plain:
            rows.append({
                "behaviour_id": b.behaviour_id,
                "attack": "static_plain",
                "suffix": "",
                "prompt": b.prompt,
                "in_loop_logprob": None,
                "found_feasible": True,
            })
        if include_gcg:
            rows.append({
                "behaviour_id": b.behaviour_id,
                "attack": "static_gcg",
                "suffix": gcg_suffix,
                "prompt": f"{b.prompt} {gcg_suffix}",
                "in_loop_logprob": None,
                "found_feasible": True,
            })
    return rows
