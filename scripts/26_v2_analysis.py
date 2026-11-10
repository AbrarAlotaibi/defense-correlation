"""Stage 26: the v2 analysis across seed replicates (plan Phase 3, decision gate).

Reads results/<run>/seed<k>/gold.jsonl (stage 06), seed<k>/cross/gold.jsonl (stage 24 + 06
--src-dir cross) and the benign responses, and writes results/<run>/analysis_v2.json plus a
readable results/<run>/analysis_v2.md with:

  * marginal ASR per defense with Wilson intervals, per seed and pooled;
  * test-retest reliability per defense, and the undefended seed-pair NULL;
  * E1: behaviour-level phi per pair (per seed, bootstrap CI), mean, disattenuated;
  * partial phi given the undefended breach rate across seeds (independent difficulty);
  * E2: prompt-level phi per pair and the transfer matrix, per seed;
  * E3: stack residual vs. per-layer intersection and strongest layer (exact McNemar);
  * benign false-refusal rate per configuration with Wilson intervals.

Usage: python scripts/26_v2_analysis.py --config configs/v2_vicuna.yaml --seeds 0 1 2
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from _bootstrap import load_env  # noqa: E402

from dcorr.analysis.estimands import (Vectors, auroc, best_of_strategies, e1_pairs,
                                      heldout_selection,
                                      e2_pairs, e3_stack, marginals, nonpairwise,
                                      null_from_undefended, partial_given_undefended,
                                      reliability, table2x2, transfer_matrix, wilson)
from dcorr.analysis.multilevel import difficulty_decomposition
from dcorr.config import load_config
from dcorr.defenses import ROWS
from dcorr.io_utils import read_jsonl, write_json
from dcorr.judge import RefusalScorer
from dcorr.runtime import load_eval


def _gold_vectors(path: Path, ids: list[str], attack: str = "adaptive") -> dict[str, np.ndarray]:
    """defense -> breach vector over `ids` for one attack; incomplete defenses dropped."""
    by: dict[str, dict[str, int]] = {}
    for g in read_jsonl(path):
        if g.get("attack") != attack:
            continue
        by.setdefault(g["defense"], {})[g["behaviour_id"]] = int(bool(g["breach"]))
    return {d: np.array([m[i] for i in ids]) for d, m in by.items() if all(i in m for i in ids)}


def _extras(path: Path, ids: list[str], vec: dict[str, np.ndarray]) -> dict:
    """2x2 tables, parse failures, non-pairwise diversity, best-of-strategies and the
    in-loop objective's AUROC for one seed (REVIEW R5-m2/m3, R7-1, R5-6, R8)."""
    rows = read_jsonl(path)
    layers = sorted(d for d in vec if d != "undefended" and not d.startswith("stack"))
    tables = {f"{a}|{b}": table2x2(vec[a], vec[b])
              for i, a in enumerate(layers) for b in layers[i + 1:]}
    parse: dict[str, dict] = {}
    for g in rows:
        if g.get("judged"):
            c = parse.setdefault(g["defense"], {"judged": 0, "with_unparsed_vote": 0})
            c["judged"] += 1
            c["with_unparsed_vote"] += int(bool(g.get("n_unparsed")) or not g.get("parse_ok", True))
    by_attack: dict[str, dict[str, np.ndarray]] = {}
    for att in sorted({g.get("attack") for g in rows}):
        for d, v in _gold_vectors(path, ids, att).items():
            by_attack.setdefault(d, {})[att] = v
    best = {d: best_of_strategies(a) for d, a in by_attack.items() if len(a) > 1}
    obj = {}
    for d in layers + ["undefended"]:
        pairs = [(g["in_loop_logprob"], int(bool(g["breach"]))) for g in rows
                 if g["defense"] == d and g.get("attack") == "adaptive"
                 and g.get("in_loop_logprob") is not None]
        if pairs:
            obj[d] = auroc([p[0] for p in pairs], [p[1] for p in pairs])
    return {
        "tables_2x2": tables,
        "parse": parse,
        "nonpairwise": nonpairwise(np.stack([vec[d] for d in layers])) if len(layers) >= 2 else {},
        "best_of_strategies": {d: {"asr": float(v.mean()), "attacks": sorted(by_attack[d])}
                               for d, v in best.items()},
        "objective_auroc": obj,
    }


def _cross_vectors(path: Path, ids: list[str]) -> dict[str, dict[str, np.ndarray]]:
    by: dict[str, dict[str, dict[str, int]]] = {}
    for g in read_jsonl(path):
        src = g["attack"].split(":", 1)[1]
        by.setdefault(src, {}).setdefault(g["defense"], {})[g["behaviour_id"]] = int(bool(g["breach"]))
    out = {}
    for src, row in by.items():
        full = {d: np.array([m[i] for i in ids]) for d, m in row.items() if all(i in m for i in ids)}
        if full:
            out[src] = full
    return out


def _frr(seed_dir: Path, model_id: str) -> dict:
    """{benign set: {configuration: FRR}}; "jbb" files end _benign, others _benign_<set>."""
    scorer = RefusalScorer(model_name=model_id)
    out: dict[str, dict] = {}
    for p in sorted(seed_dir.glob("stage0[45]_*_benign*.jsonl")):
        stem = p.stem.replace("stage04_", "").replace("stage05_", "")
        name, _, bset = stem.partition("_benign")
        bset = bset.lstrip("_") or "jbb"
        rows = read_jsonl(p)
        k = sum(scorer.is_refusal(r.get("response", ""), bool(r.get("blocked"))) for r in rows)
        out.setdefault(bset, {})[name] = {"frr": k / len(rows) if rows else float("nan"),
                                          "n": len(rows), "ci": wilson(k, len(rows))}
    return out


def _clean(o):
    """JSON-safe: nan -> None, numpy -> python, tuple -> list, int keys -> str."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if math.isnan(float(o)) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    return o


def _fmt(x, nd=2):
    return "-" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def _markdown(res: dict) -> str:
    L = [f"# v2 analysis: {res['run']} (seeds {res['seeds']}, n={res['n_behaviours']})", ""]
    L += ["## Marginal ASR", "", "| defense | mean ASR | pooled 95% CI | reliability r |",
          "|---|---|---|---|"]
    for d, m in res["marginals"].items():
        r = res["reliability"].get(d, {}).get("mean")
        L.append(f"| {d} | {_fmt(m['mean_asr'])} | [{_fmt(m['pooled_ci'][0])}, "
                 f"{_fmt(m['pooled_ci'][1])}] | {_fmt(r)} |")
    nul = res["null"]
    L += ["", f"**Null (undefended seed-pair phi):** {_fmt(nul.get('mean')) if nul.get('available') else 'needs >= 2 seeds'}", ""]
    L += ["## E1 behaviour-level phi (separate searches)", "",
          "| pair | mean phi | disattenuated | partial given undefended | per seed |",
          "|---|---|---|---|---|"]
    for k, v in res["e1"].items():
        per = ", ".join(_fmt(s["phi"]) for s in v["per_seed"].values())
        L.append(f"| {k} | {_fmt(v['mean_phi'])} | {_fmt(v['disattenuated'])} | "
                 f"{_fmt(res['partial'].get(k, {}).get('mean'))} | {per} |")
    for s, e2 in res["e2"].items():
        L += ["", f"## E2 prompt-level phi, seed {s} (same prompts)", "",
              "| pair | mean over sources |", "|---|---|"]
        L += [f"| {k} | {_fmt(v['mean_phi'])} |" for k, v in e2.items()]
    if res["e3"]:
        L += ["", "## E3 assembled stack", "",
              "| seed | stack ASR | 95% CI | intersection | strongest layer | McNemar p (vs strongest) |",
              "|---|---|---|---|---|---|"]
        for s, v in res["e3"].items():
            L.append(f"| {s} | {_fmt(v['stack_asr'])} | [{_fmt(v['stack_ci'][0])}, "
                     f"{_fmt(v['stack_ci'][1])}] | {_fmt(v['intersection_asr'])} | "
                     f"{v['strongest_layer']} {_fmt(v['strongest_asr'])} | "
                     f"{_fmt(v['stack_vs_strongest']['p'], 3)} |")
    dd = res.get("difficulty") or {}
    if dd:
        f = dd["fit"]
        L += ["", "## How much is behaviour difficulty? (random-intercept logistic model)", "",
              f"Behaviour SD {_fmt(f['sd_behaviour'])}, ICC {_fmt(f['icc'])} "
              f"({f['n_obs']} observations). Predicted phi assumes the pair fails "
              "independently given the behaviour.", "",
              "| pair | observed phi | predicted by difficulty alone | excess |", "|---|---|---|---|"]
        L += [f"| {k} | {_fmt(v['observed'])} | {_fmt(v['predicted_difficulty_only'])} | "
              f"{_fmt(v['excess'])} |" for k, v in dd["pairs"].items()]
    for s, x in res.get("extras", {}).items():
        L += ["", f"## Seed {s}: diversity, strategies, objective, grader parsing", ""]
        npw = x.get("nonpairwise") or {}
        if npw:
            L.append("Non-pairwise diversity over the layers (Kuncheva & Whitaker 2003): "
                     + ", ".join(f"{k} {_fmt(v, 3)}" for k, v in npw.items()))
        if x.get("best_of_strategies"):
            L += ["", "| defense | best-of-strategies ASR | attacks pooled |", "|---|---|---|"]
            L += [f"| {d} | {_fmt(v['asr'])} | {', '.join(v['attacks'])} |"
                  for d, v in x["best_of_strategies"].items()]
        if x.get("objective_auroc"):
            L += ["", "| defense | AUROC of in-loop logprob for grader breach |", "|---|---|"]
            L += [f"| {d} | {_fmt(v)} |" for d, v in x["objective_auroc"].items()]
        if x.get("parse"):
            L += ["", "| defense | judged | labels with an unparsed vote |", "|---|---|---|"]
            L += [f"| {d} | {v['judged']} | {v['with_unparsed_vote']} |"
                  for d, v in x["parse"].items()]
    for name, e3 in (res.get("e3_subsets") or {}).items():
        members = name[len("stack_"):].replace("+", ", ")
        L += ["", f"## E3 jointly attacked subset: {members}", "",
              "| seed | joint ASR | 95% CI | intersection of members | McNemar p (joint vs intersection) |",
              "|---|---|---|---|---|"]
        for s, v in e3.items():
            L.append(f"| {s} | {_fmt(v['stack_asr'])} | [{_fmt(v['stack_ci'][0])}, "
                     f"{_fmt(v['stack_ci'][1])}] | {_fmt(v['intersection_asr'])} | "
                     f"{_fmt(v['stack_vs_intersection']['p'], 3)} |")
    for s, sel in (res.get("selection") or {}).items():
        for variant, v in sel.items():
            ks = sorted({k for m in ("greedy", "row", "random") for k in v.get(m, {})}, key=int)
            L += ["", f"## Held-out member selection, seed {s}, {variant.replace('_', ' ')} "
                  f"({v['splits']} splits; test-half intersection residual)", "",
                  "| k | diversity-greedy | one per row | random subset |", "|---|---|---|---|"]
            L += [f"| {k} | {_fmt(v['greedy'].get(k))} | {_fmt(v['row'].get(k))} | "
                  f"{_fmt(v['random'].get(k))} |" for k in ks]
            L += ["", f"Single best layer alone: {_fmt(v['single_best'])}"]
    for s, sets in res["frr"].items():
        for bset, f in sets.items():
            L += ["", f"## Benign false-refusal rate, seed {s}, {bset} set", "",
                  "| configuration | FRR | 95% CI | n |", "|---|---|---|---|"]
            L += [f"| {d} | {_fmt(v['frr'])} | [{_fmt(v['ci'][0])}, {_fmt(v['ci'][1])}] | {v['n']} |"
                  for d, v in f.items()]
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--gold-name", default="gold.jsonl")
    ap.add_argument("--attack", default="adaptive",
                    help="which attack family's rows to analyse. The default is the genetic "
                         "arm every estimand in the paper was computed over; 'pair' is the "
                         "second family, and its results go to analysis_v2_pair.json so the "
                         "two cannot overwrite each other.")
    ap.add_argument("--resamples", type=int, default=2000)
    ap.add_argument("--splits", type=int, default=500, help="held-out selection splits")
    args = ap.parse_args()

    load_env()
    cfg = load_config(args.config)
    harmful, _ = load_eval(cfg)
    ids = [h["behaviour_id"] for h in harmful]
    run_dir = cfg.run_dir

    vec, stack, cross, frr, extras = {}, {}, {}, {}, {}
    for s in args.seeds:
        sd = run_dir / f"seed{s}"
        g = _gold_vectors(sd / args.gold_name, ids, args.attack)
        extras[s] = _extras(sd / args.gold_name, ids, g)
        for name in [k for k in g if k.startswith("stack")]:
            stack.setdefault(name, {})[s] = g.pop(name)
        vec[s] = g
        # The cross-evaluation is graded by stage 06 --src-dir cross, which writes gold.jsonl.
        # Stage 35 resolves abstentions for the main pass only, so with --gold-name
        # gold_resolved.jsonl this path does not exist and E2 came out silently empty. Fall
        # back, and say so, rather than dropping a whole estimand without a word.
        cxp = sd / "cross" / args.gold_name
        if not cxp.exists() and (sd / "cross" / "gold.jsonl").exists():
            print(f"[26][WARN] seed{s}: no cross/{args.gold_name}; using cross/gold.jsonl, "
                  f"whose grader abstentions are unresolved and count as non-breach")
            cxp = sd / "cross" / "gold.jsonl"
        cx = _cross_vectors(cxp, ids)
        if cx:
            cross[s] = cx
        elif (sd / "cross").is_dir():
            print(f"[26][WARN] seed{s}: cross/ exists but yielded no vectors; E2 will be empty")
        frr[s] = _frr(sd, cfg.get_path("target.model_id"))
    V = Vectors(behaviours=ids, vec=vec)
    layers = [d for d in V.defenses() if d != "undefended"]

    # E3 needs rows whose defense is the assembled stack. If --gold-name points at a file
    # written before the stack was graded -- which is what gold_resolved.jsonl is until
    # stage 35 is re-run -- E3 is empty for a reason that has nothing to do with the data.
    if not stack:
        present = sorted({k for s in args.seeds
                          for k in _gold_vectors(run_dir / f"seed{s}" / "gold.jsonl", ids)
                          if k.startswith("stack")})
        print(f"[26][WARN] no stack rows in {args.gold_name}, so E3 will be empty."
              + (f" gold.jsonl has {present}: re-run stage 35 to fold them in."
                 if present else " gold.jsonl has none either: the stack is not graded yet."))
    elif not [d for d in cfg.get_path("stack.order", []) if d in layers]:
        # This run attacked the stack but not its members, which is what a configuration-only
        # run looks like: v2_vicuna_nolg exists to add the six-layer stack, not to re-measure
        # members already measured under the same thresholds and seeds in the primary run.
        # Its intersection therefore has to be joined from that run, which
        # scripts/47_e3_table.py does; E3 here is correctly empty rather than an error.
        print(f"[26][WARN] {cfg.get_path('run_name')} has stack rows but no member arms "
              f"({sorted(layers)}), so its own E3 is empty. The intersection for this stack "
              f"comes from the run that measured the members; scripts/47_e3_table.py joins "
              f"them.")

    res = {
        "run": cfg.get_path("run_name"), "seeds": args.seeds, "n_behaviours": len(ids),
        "attack": args.attack,
        "marginals": marginals(V),
        "reliability": reliability(V) if len(args.seeds) >= 2 else {},
        "null": null_from_undefended(V),
        "e1": e1_pairs(V, resamples=args.resamples),
        "partial": partial_given_undefended(V) if "undefended" in V.defenses() else {},
        "e2": {s: e2_pairs(c) for s, c in cross.items()},
        "transfer": {s: transfer_matrix(c) for s, c in cross.items()},
        # E3 for the full stack and for every jointly attacked subset (05 --layers, plan 4.7):
        # each is compared against the intersection of ITS OWN members.
        "e3": e3_stack(V, stack.get("stack", {}),
                       [d for d in cfg.get_path("stack.order", []) if d in layers]),
        "e3_subsets": {name: e3_stack(V, per_seed,
                                      [d for d in name[len("stack_"):].split("+") if d in layers])
                       for name, per_seed in sorted(stack.items()) if name != "stack"},
        "frr": frr,
        "extras": extras,
        "difficulty": difficulty_decomposition(V) if len(V.defenses()) >= 3 else {},
        "selection": {s: {
            "all_layers": heldout_selection({d: V.vec[s][d] for d in layers}, ROWS,
                                            splits=args.splits, seed=s),
            "without_llamaguard": heldout_selection(
                {d: V.vec[s][d] for d in layers if d != "llamaguard"}, ROWS,
                splits=args.splits, seed=s),
        } for s in args.seeds} if len(layers) >= 2 else {},
    }
    res = _clean(res)
    stem = "analysis_v2" if args.attack == "adaptive" else f"analysis_v2_{args.attack}"
    write_json(run_dir / f"{stem}.json", res)
    (run_dir / f"{stem}.md").write_text(_markdown(res), encoding="utf-8")
    print(f"[26] wrote {run_dir / (stem + '.json')} and {stem}.md "
          f"(attack={args.attack})")


if __name__ == "__main__":
    main()
