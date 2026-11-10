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

from .attacks import build_static, run_adaptive, run_autodan, run_gcg, run_pair
from .attacks.random_search import ADV_TEMPLATE, Behaviour
from .defenses.base import Defense
from .io_utils import append_jsonl, done_keys, read_jsonl
from .seeding import search_seed
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
    seed_index: int = 0,
    transfer_from: Path | None = None,
    run_adaptive_attack: bool = True,
    run_static_attack: bool = True,
    pair_attacker=None,
    pair_cfg: dict | None = None,
    progress=None,
) -> list[dict]:
    model = target.model_id
    already = done_keys(out_path, KEY_FIELDS)
    # Search seeding (dcorr/seeding.py): one RNG per (seed_index, defense, behaviour).
    seeding = dict(
        seed=seed,
        seed_index=int(seed_index),
        common_random_numbers=bool(attack_cfg.get("common_random_numbers", False)),
        carry_across_behaviours=bool(attack_cfg.get("carry_across_behaviours", False)),
    )

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
                **seeding,
                # rounds path, used only by defenses with an expensive constraint (Llama Guard)
                expensive_rounds=int(attack_cfg.get("expensive_rounds", 3)),
                expensive_pool_k=int(attack_cfg.get("expensive_pool_k", 8)),
                final_top_m=int(attack_cfg.get("final_top_m", 3)),
                max_new_tokens=max_new_tokens,
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
                **seeding,
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
                # Behaviours run in lockstep here, so one RNG per run rather than per
                # behaviour; still keyed by seed_index and defense, never shared by default.
                seed=search_seed(seed, seeding["seed_index"], defense.name, "__lockstep__",
                                 common_random_numbers=seeding["common_random_numbers"]),
                progress=progress,
            )
        craft.extend(recs)

    # A second attack family, recorded under its own `attack` value rather than as another
    # `adaptive` method, so both arms sit in one gold.jsonl and every estimand can be computed
    # per family without re-running anything. Opt-in: it needs a second model in memory, and
    # every run before this one predates it.
    if pair_attacker is not None and _pending("pair"):
        pend = _pending("pair")
        # Explicit, and required. This used to read attack_cfg["pair"], but stages 03 and 04
        # build attack_cfg from attack.adaptive + attack.static, so the key was never present
        # and every run silently used the defaults below instead of the configured budget.
        # Refusing is right: an attacker with no configuration is a wiring mistake, and a
        # search budget that does not match the config invalidates the comparison with the
        # genetic arm, which is the whole point of the family.
        if pair_cfg is None:
            raise ValueError(
                "run_one_defense got pair_attacker but no pair_cfg; pass "
                "cfg.get_path('attack.pair') so the budget comes from the config rather "
                "than from these defaults")
        pair_cfg = dict(pair_cfg)

        def _pair_prog(bi, nb, rec, _name=defense.name):
            print(f"[{_name}] pair behaviour {bi}/{nb} done  "
                  f"fit={rec['in_loop_logprob']:.3f}  feasible={rec['found_feasible']}  "
                  f"parse_fail={rec['pair_parse_failures']}", flush=True)

        craft.extend(run_pair(
            target, defense, pend,
            attacker=pair_attacker,
            streams=int(pair_cfg.get("streams", 3)),
            iterations=int(pair_cfg.get("iterations", 5)),
            temperature=float(pair_cfg.get("temperature", 1.0)),
            top_p=float(pair_cfg.get("top_p", 0.9)),
            # Same truncated target prefix as the genetic arm, so the objective is identical.
            target_prefix_tokens=int(attack_cfg.get("target_prefix_tokens", 12)),
            attacker_max_new_tokens=int(pair_cfg.get("attacker_max_new_tokens", 512)),
            history_turns=int(pair_cfg.get("history_turns", 2)),
            max_new_tokens=max_new_tokens,
            seed=seed,
            seed_index=int(seed_index),
            progress=_pair_prog,
        ))

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
            "seed_index": c.get("seed_index", seeding["seed_index"]),
            "search_seed": c.get("search_seed"),
            "genes": c.get("genes"),
            "loss_trace": c.get("loss_trace"),
            # rounds path (expensive constraints): guard diagnostics, None elsewhere
            **{k: c.get(k) for k in ("guard_rounds", "guard_checked", "verified_candidates",
                                     "final_tried", "final_blocked", "round_seconds")},
            # second attack family: its budget, how often the attacker's reply could not be
            # read, and which model produced the prompts. None for every other arm. Without
            # these the stored rows cannot say what budget the arm ran at or whether the
            # attacker was attacking rather than refusing, and `iterations` is what makes
            # n_evals interpretable.
            **{k: c.get(k) for k in ("iterations", "pair_streams", "pair_parse_failures",
                                     "attacker_model")},
            "response": resp["response"],
            "raw_response": resp["raw_response"],
            "blocked": resp["blocked"],
            "blocked_stage": resp["blocked_stage"],
            "input_score": resp["input_score"],
            "output_score": resp["output_score"],
        }
        append_jsonl(out_path, row)

    return read_jsonl(out_path)
