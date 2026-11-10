"""Stage 24: same-input cross-evaluation (REVIEW R2-1, R5-1, R7-8a; plan item 1.5, Run C).

The behaviour-level estimand intersects breach vectors from SEPARATE searches, so two
defenses' entries for a behaviour come from different prompts. This stage measures the
same-input estimand that Eq. (6) describes: every defense is evaluated, as deployed, on the
final adaptive prompt that every OTHER defense's search produced (and on the undefended
search's prompts - the shared transfer pool R7 asks for).

Generation is shared wherever the deployed systems are identical. A defense that does not
change generation (perplexity, token-anomaly, probes, Llama Guard, undefended) answers an
admitted prompt with the target's plain greedy response, so that response is generated once
per prompt; refusal-prime and SmoothLLM each generate their own. Input filters and output
filters run once per defense over all prompts (one guard block each for Llama Guard).

Self-check: on the diagonal (a defense evaluated on its own search's prompts) the deployed
response must reproduce stage 04's. Mismatches are reported, never silently ignored.

Writes results/<run>[/seed<k>]/cross/stage04_<defense>.jsonl, rows keyed with
attack = "cross:<source>". Then:
  python scripts/06_judge_gold.py --config C [--seed-index k] --src-dir cross
  python scripts/24_cross_eval.py  --config C [--seed-index k] --export
writes cross/cross_breach.csv for scripts/input_level_phi.py.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict

from _bootstrap import load_env  # noqa: E402

from dcorr.gpulock import acquire_for_process
from dcorr.config import apply_seed_index, load_config
from dcorr.defenses import Undefended, build_defenses
from dcorr.defenses.base import REFUSAL_TEXT
from dcorr.io_utils import read_json, read_jsonl, set_all_seeds, write_jsonl
from dcorr.runtime import build_target


def _merge_calibration(cfg) -> None:
    calib = read_json(cfg.run_dir / "calibration.json")
    for name, thr in ((calib or {}).get("thresholds") or {}).items():
        if cfg.get_path(f"defenses.{name}.enabled"):
            cfg.set_path(f"defenses.{name}.threshold", thr)


def _sources(rd) -> dict[tuple[str, str], str]:
    """(source defense, behaviour_id) -> final adaptive prompt, from every stage-04 file."""
    out: dict[tuple[str, str], str] = {}
    for p in sorted(rd.glob("stage04_*.jsonl")):
        if p.name.endswith("_benign.jsonl"):
            continue
        src = p.stem[len("stage04_"):]
        for r in read_jsonl(p):
            if r.get("attack") == "adaptive":
                out[(src, r["behaviour_id"])] = r["prompt"]
    return out


def _gen_key(d) -> str:
    """Defenses sharing a key return identical responses to an admitted prompt."""
    if d.name == "smoothllm":
        return "smoothllm"
    if d.modifies_generation:
        return f"system:{d.system_for_generation()}"
    return "plain"


def evaluate(cfg, max_new: int) -> dict:
    rd = cfg.results_dir
    prompts = _sources(rd)
    if not prompts:
        raise RuntimeError(f"no stage04 adaptive rows under {rd}")
    uniq = list(dict.fromkeys(prompts.values()))

    target = build_target(cfg)
    defenses = build_defenses(cfg, target)
    defenses["undefended"] = Undefended({}, target, cfg)
    print(f"[24] {len(prompts)} (source, behaviour) prompts, {len(uniq)} unique, "
          f"{len(defenses)} defenses")

    blocked_in: dict[str, dict[str, bool]] = {}
    for name, d in defenses.items():
        if d.has_input_filter:
            fo = d.filter_input(uniq)
            blocked_in[name] = dict(zip(uniq, (bool(b) for b in fo.blocked)))
        else:
            blocked_in[name] = {p: False for p in uniq}

    groups: dict[str, list[str]] = defaultdict(list)
    for name, d in defenses.items():
        groups[_gen_key(d)].append(name)
    responses: dict[str, dict[str, str]] = {}
    for key, names in groups.items():
        need = [p for p in uniq if not all(blocked_in[n][p] for n in names)]
        d = defenses[names[0]]
        print(f"[24] generating '{key}' for {len(need)} prompts (shared by {names})", flush=True)
        if key == "plain":
            outs = target.generate(need, max_new_tokens=max_new, system="__default__")
        else:
            outs = d.transform_generate(need, max_new)
        responses[key] = dict(zip(need, outs))

    blocked_out: dict[str, dict[str, bool]] = {}
    for name, d in defenses.items():
        if not d.has_output_filter:
            continue
        cand = [p for p in uniq if not blocked_in[name][p]]
        resp = responses[_gen_key(d)]
        fo = d.filter_output(cand, [resp[p] for p in cand])
        blocked_out[name] = dict(zip(cand, (bool(b) for b in fo.blocked)))

    rows_by_def: dict[str, list[dict]] = defaultdict(list)
    for name, d in defenses.items():
        key = _gen_key(d)
        for (src, bid), p in sorted(prompts.items()):
            bi = blocked_in[name][p]
            bo = (not bi) and blocked_out.get(name, {}).get(p, False)
            raw = responses[key].get(p, REFUSAL_TEXT)
            rows_by_def[name].append({
                "model": target.model_id, "defense": name, "row": d.row,
                "behaviour_id": bid, "attack": f"cross:{src}", "source": src, "prompt": p,
                "response": REFUSAL_TEXT if (bi or bo) else raw, "raw_response": raw,
                "blocked": bi or bo,
                "blocked_stage": "input" if bi else ("output" if bo else None),
                "generation": key,
            })

    out_dir = rd / "cross"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in rows_by_def.items():
        write_jsonl(out_dir / f"stage04_{name}.jsonl", rows)

    # Self-check: the diagonal must reproduce the deployed responses of stage 04.
    check = {}
    for name in defenses:
        own = {r["behaviour_id"]: r for r in read_jsonl(rd / f"stage04_{name}.jsonl")
               if r.get("attack") == "adaptive"}
        diag = [r for r in rows_by_def[name] if r["source"] == name]
        same = sum(1 for r in diag if own.get(r["behaviour_id"], {}).get("response") == r["response"])
        check[name] = {"n": len(diag), "same_response": same}
    target.free()
    return check


def export(cfg) -> None:
    rd = cfg.results_dir / "cross"
    gold = read_jsonl(rd / "gold.jsonl")
    if not gold:
        raise RuntimeError(f"no {rd / 'gold.jsonl'}: run 06_judge_gold.py --src-dir cross first")
    out = rd / "cross_breach.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["source", "defense", "behavior", "breach"])
        for g in gold:
            src = g["attack"].split(":", 1)[1]
            w.writerow([f"adaptive:{src}", g["defense"], g["behaviour_id"], int(bool(g["breach"]))])
    print(f"[24] wrote {out} ({len(gold)} rows)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed-index", type=int, default=None,
                    help="attack replicate k: reads/writes results/<run>/seed<k>")
    ap.add_argument("--export", action="store_true",
                    help="after grading, write cross/cross_breach.csv for input_level_phi.py")
    args = ap.parse_args()

    load_env()

    cfg = load_config(args.config)
    apply_seed_index(cfg, args.seed_index)
    if args.export:
        export(cfg)          # CPU only: no GPU lock
        return
    acquire_for_process("stage 24 cross-eval")   # one GPU job at a time
    set_all_seeds(int(cfg.get("seed", 0)))
    _merge_calibration(cfg)
    check = evaluate(cfg, int(cfg.get_path("target.max_new_tokens", 256)))
    bad = {k: v for k, v in check.items() if v["n"] and v["same_response"] != v["n"]}
    for name, v in sorted(check.items()):
        print(f"[24] diagonal {name}: {v['same_response']}/{v['n']} responses reproduce stage 04")
    if bad:
        print(f"[24][WARN] diagonal mismatches: {bad}. Inspect before analysing the cross results.")
    print("[24] done.")


if __name__ == "__main__":
    main()
