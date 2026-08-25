"""Run one defense against the behaviour list: attacks -> deployed responses -> rows.

Shared by stages 03/04/05. This produces everything EXCEPT the gold judgement, which is a
separate post-hoc pass (stage 06). The only in-loop signal used here is the target
logprob inside run_adaptive; no judge model is touched.

For each (behaviour, attack) it records the crafted prompt, the deployed system's response
(after the defense's own input/generation/output handling), whether a filter blocked it,
and the in-loop logprob. Rows are appended to a JSONL keyed by
(model, defense, behaviour_id, attack) so a killed run resumes.
"""
from __future__ import annotations

from pathlib import Path

from .attacks import build_static, run_adaptive, run_autodan, run_gcg
from .attacks.random_search import ADV_TEMPLATE, Behaviour
from .defenses.base import Defense
from .io_utils import append_jsonl, done_keys, read_jsonl
from .target import HFTarget

KEY_FIELDS = ("model", "defense", "behaviour_id", "attack")


def _init_suffixes_from(path: Path, behaviour_ids: set[str]) -> dict[str, list[int]]:
    """Self-transfer seeds: best undefended adaptive suffix per behaviour, if available."""
    out: dict[str, list[int]] = {}
    for r in read_jsonl(path):
        if r.get("attack") == "adaptive" and r.get("behaviour_id") in behaviour_ids:
            sfx = r.get("suffix_token_ids")
            if sfx:
                out[r["behaviour_id"]] = sfx
    return out


def run_one_defense(
    target: HFTarget,
    defense: Defense,
    behaviours: list[Behaviour],
    out_path: Path,
    *,
    attack_cfg: dict,
    gcg_suffix: str,
    max_new_tokens: int = 256,
    seed: int = 0,
    transfer_from: Path | None = None,
    run_adaptive_attack: bool = True,
    run_static_attack: bool = True,
    progress=None,
) -> list[dict]:
    model = target.model_id
    already = done_keys(out_path, KEY_FIELDS)

    def _pending(attack: str) -> list[Behaviour]:
        return [b for b in behaviours
                if (model, defense.name, b.behaviour_id, attack) not in already]

    craft: list[dict] = []

    if run_adaptive_attack and _pending("adaptive"):
        pend = _pending("adaptive")
        seeds = {}
        if transfer_from is not None and attack_cfg.get("self_transfer", True):
            seeds = _init_suffixes_from(transfer_from, {b.behaviour_id for b in pend})
        method = str(attack_cfg.get("method", "gcg")).lower()
        tmpl = ADV_TEMPLATE if attack_cfg.get("use_template", True) else None
        if method == "autodan":
            # Fluency-preserving GA: the only attack in this project that can produce
            # low-perplexity prompts, so it is the one that can measure the token-surface row.
            def _ad_prog(bi, nb, g, gs, fit, _name=defense.name):
                if g == gs:
                    print(f"[{_name}] autodan behaviour {bi + 1}/{nb} done  fit={fit:.3f}",
                          flush=True)
            recs = run_autodan(
                target, defense, pend,
                generations=int(attack_cfg.get("generations", 30)),
                population=int(attack_cfg.get("population", 24)),
                elite_frac=float(attack_cfg.get("elite_frac", 0.25)),
                mutation_rate=float(attack_cfg.get("mutation_rate", 0.15)),
                target_prefix_tokens=int(attack_cfg.get("target_prefix_tokens", 12)),
                use_affirm_tail=bool(attack_cfg.get("use_affirm_tail", True)),
                seed=seed,
                progress=_ad_prog,
            )
        elif method == "template_only":
            # No suffix optimisation: the compliance template alone is the adaptive attack.
            # Cheapest and most robust path (generation only, no scoring) - used when the
            # optimiser is impractical on the hardware. Defense feasibility still applies at
            # respond() time (a blocked templated prompt is a non-breach).
            from .attacks.random_search import _compose
            recs = [{
                "behaviour_id": b.behaviour_id, "attack": "adaptive", "suffix": "",
                "prompt": _compose(target, b.prompt, [], tmpl, b.target_str),
                "in_loop_logprob": None, "found_feasible": True,
            } for b in pend]
        elif method == "gcg":
            # GCG has a different progress signature than the RS stage callbacks; log
            # per-behaviour completion here instead of threading the RS-style callback.
            def _gcg_prog(bi, nb, step, steps, loss, _name=defense.name):
                if step == steps:
                    print(f"[{_name}] gcg behaviour {bi + 1}/{nb} done  best_loss={loss:.3f}",
                          flush=True)
            recs = run_gcg(
                target, defense, pend,
                steps=int(attack_cfg.get("steps", attack_cfg.get("iterations", 150))),
                suffix_n_tokens=int(attack_cfg.get("suffix_n_tokens", 20)),
                top_k=int(attack_cfg.get("top_k", 96)),
                batch_size=int(attack_cfg.get("batch_size", 32)),
                target_prefix_tokens=int(attack_cfg.get("target_prefix_tokens", 12)),
                template=tmpl,
                eval_chunk=int(attack_cfg.get("eval_chunk", 32)),
                early_stop_loss=float(attack_cfg.get("early_stop_loss", 0.5)),
                init_suffixes=seeds,
                seed=seed,
                progress=_gcg_prog,
            )
        else:
            recs = run_adaptive(
                target, defense, pend,
                iterations=int(attack_cfg.get("iterations", 200)),
                suffix_n_tokens=int(attack_cfg.get("suffix_n_tokens", 25)),
                n_candidates=int(attack_cfg.get("n_candidates", 16)),
                target_prefix_tokens=int(attack_cfg.get("target_prefix_tokens", 10)),
                template=tmpl,
                restart_after_stall=int(attack_cfg.get("restart_after_stall", 60)),
                max_filter_rejections=int(attack_cfg.get("max_filter_rejections", 40)),
                init_suffixes=seeds,
                seed=seed,
                progress=progress,
            )
        craft.extend(recs)

    if run_static_attack:
        static_all = build_static(behaviours, gcg_suffix,
                                  include_plain=attack_cfg.get("include_plain", True),
                                  include_gcg=attack_cfg.get("include_gcg", True))
        craft.extend([r for r in static_all
                      if (model, defense.name, r["behaviour_id"], r["attack"]) not in already])

    if not craft:
        return read_jsonl(out_path)

    # Deployed response for each crafted prompt, batched by attack for locality.
    prompts = [c["prompt"] for c in craft]
    responses = defense.respond(prompts, max_new_tokens=max_new_tokens)

    for c, resp in zip(craft, responses):
        row = {
            "model": model,
            "defense": defense.name,
            "row": defense.row,
            "behaviour_id": c["behaviour_id"],
            "attack": c["attack"],
            "prompt": c["prompt"],
            "suffix": c.get("suffix", ""),
            "suffix_token_ids": c.get("suffix_token_ids"),
            "in_loop_logprob": c.get("in_loop_logprob"),
            "found_feasible": c.get("found_feasible", True),
            "n_evals": c.get("n_evals"),
            "n_infeasible": c.get("n_infeasible"),
            "response": resp["response"],
            "raw_response": resp["raw_response"],
            "blocked": resp["blocked"],
            "blocked_stage": resp["blocked_stage"],
            "input_score": resp["input_score"],
            "output_score": resp["output_score"],
        }
        append_jsonl(out_path, row)

    return read_jsonl(out_path)
