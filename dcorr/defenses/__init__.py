"""Defense registry. Row assignments live in the config and in PREREGISTRATION.md."""
from __future__ import annotations

from .base import Defense, FilterOut, REFUSAL_TEXT
from .llamaguard import LlamaGuard
from .ppl_filter import PerplexityFilter
from .probe import LinearProbe, LinearProbeB
from .promptguard import PromptGuard
from .refusal_prime import RefusalPrime
from .smoothllm import SmoothLLM
from .stack import DefenseStack
from .token_anomaly import TokenAnomalyFilter

REGISTRY: dict[str, type[Defense]] = {
    "ppl_filter": PerplexityFilter,
    "token_anomaly": TokenAnomalyFilter,
    "llamaguard": LlamaGuard,
    "promptguard": PromptGuard,
    "refusal_prime": RefusalPrime,
    "smoothllm": SmoothLLM,
    "probe": LinearProbe,
    "probe_b": LinearProbeB,
}

# Row assignment of record. Must agree with configs/*.yaml and PREREGISTRATION.md;
# build_defenses() asserts that.
ROWS: dict[str, str] = {
    "ppl_filter": "token_surface",
    "token_anomaly": "token_surface",
    "llamaguard": "semantic_classification",
    "promptguard": "semantic_classification",
    "refusal_prime": "first_token_distribution",
    "smoothllm": "perturbation_stability",
    "probe": "internal_representations",
    "probe_b": "internal_representations",
}


class Undefended(Defense):
    """The positive control: no filters, plain greedy decode."""
    name = "undefended"
    row = "none"
    overhead_class = "A"


def build_defense(name: str, cfg: dict, target, run_cfg=None) -> Defense:
    if name == "undefended":
        return Undefended(cfg or {}, target, run_cfg)
    if name not in REGISTRY:
        raise KeyError(f"unknown defense {name!r}; known: {sorted(REGISTRY)}")
    d = REGISTRY[name](cfg or {}, target, run_cfg)
    declared = (cfg or {}).get("row")
    if declared and declared != ROWS[name]:
        raise ValueError(
            f"row mismatch for {name}: config says {declared!r}, registry of record says "
            f"{ROWS[name]!r}. Row assignments are pre-registered - fix the config, not this."
        )
    return d


def build_defenses(cfg, target) -> dict[str, Defense]:
    """Instantiate every enabled defense from a run config."""
    out: dict[str, Defense] = {}
    for name in cfg.enabled_defenses():
        out[name] = build_defense(name, cfg.get_path(f"defenses.{name}", {}), target, cfg)
    return out


def build_stack(cfg, target, defenses: dict[str, Defense]) -> DefenseStack:
    order = cfg.get_path("stack.order", list(defenses))
    return DefenseStack(list(defenses.values()), order, target, cfg)


__all__ = [
    "Defense", "FilterOut", "REFUSAL_TEXT", "REGISTRY", "ROWS", "Undefended",
    "DefenseStack", "build_defense", "build_defenses", "build_stack",
]
