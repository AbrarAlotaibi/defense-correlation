"""Stage 09: emit the fusion-analysis inputs from a run's stored artifacts.

scripts/fusion_analysis.py wants a behaviours x defenses breach matrix, a benign
prompts x defenses refusal matrix, and a defense -> dependency-row map. All three are
already implied by results/<run>/; this script pivots them, so no number is transcribed
by hand.

  breach.csv   one row per behaviour, one column per defense, 1 = breached under the
               chosen attack (default `adaptive`, the paper's fluent adversary), taken
               from gold.jsonl. `undefended` is dropped: it is the control, not a stack
               member.
  refusal.csv  one row per benign prompt, 1 = the deployed configuration refused it,
               taken from stage04_<defense>_benign.jsonl. The refusal decision is the
               same RefusalScorer call scripts/07_analyze.py uses for H3, so the column
               means reproduce analysis.json's h3_refusals exactly; --verify-frr asserts
               that rather than trusting it.
  rows.csv     defense,row  taken from the `row` field the run itself recorded.

Usage:
  python scripts/09_build_fusion_inputs.py --run results/hpc_vicuna_autodan \
      --config configs/hpc_vicuna_autodan.yaml --out fusion/ --verify-frr
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def read_jsonl(p: Path) -> list[dict]:
    with open(p, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_breach(rows: list[dict], attack: str, drop: set[str]):
    """Pivot to {(behaviour, defense): 0/1}, asserting a complete rectangle."""
    sel = [r for r in rows if r["attack"] == attack and r["defense"] not in drop]
    if not sel:
        raise SystemExit(f"no rows for attack={attack!r}")
    defenses = sorted({r["defense"] for r in sel})
    behaviours = sorted({r["behaviour_id"] for r in sel}, key=lambda s: (len(s), s))
    table: dict[tuple[str, str], int] = {}
    for r in sel:
        key = (r["behaviour_id"], r["defense"])
        if key in table:
            raise SystemExit(f"duplicate verdict for {key}")
        table[key] = int(bool(r["breach"]))
    missing = [(b, d) for b in behaviours for d in defenses if (b, d) not in table]
    if missing:
        raise SystemExit(f"{len(missing)} missing cells, e.g. {missing[:3]}")
    return defenses, behaviours, table


def build_refusal(run: Path, config: Path, defenses: list[str]):
    """Per-benign-prompt refusal, using the same scorer 07_analyze.py applies for H3."""
    from dcorr.config import load_config
    from dcorr.judge import RefusalScorer

    cfg = load_config(str(config))
    scorer = RefusalScorer(
        max_response_length=int(cfg.get_path("refusal.max_response_length", 600)),
        model_name=cfg.get_path("target.model_id"),
    )
    per: dict[str, list[dict]] = {}
    for d in defenses:
        p = run / f"stage04_{d}_benign.jsonl"
        if not p.is_file():
            print(f"[skip] refusal.csv: {p.name} not found")
            return None, None
        per[d] = read_jsonl(p)

    ids = [r.get("benign_id") for r in per[defenses[0]]]
    for d, rows in per.items():
        if [r.get("benign_id") for r in rows] != ids:
            raise SystemExit(f"benign prompt order differs for {d}; cannot align columns")

    matrix = [[int(bool(scorer.is_refusal(per[d][i].get("response", ""),
                                          per[d][i].get("blocked", False))))
               for d in defenses]
              for i in range(len(ids))]
    return ids, matrix


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--config", type=Path, help="run config; required for refusal.csv")
    ap.add_argument("--attack", default="adaptive")
    ap.add_argument("--drop", default="undefended,stack")
    ap.add_argument("--verify-frr", action="store_true",
                    help="assert the refusal column means equal analysis.json h3_refusals")
    a = ap.parse_args()

    drop = {s.strip() for s in a.drop.split(",") if s.strip()}
    rows = read_jsonl(a.run / "gold.jsonl")
    defenses, behaviours, table = build_breach(rows, a.attack, drop)
    a.out.mkdir(parents=True, exist_ok=True)

    with open(a.out / "breach.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(defenses)
        for b in behaviours:
            w.writerow([table[(b, d)] for d in defenses])

    rowmap: dict[str, str] = {}
    for r in rows:
        if r["defense"] in defenses:
            rowmap.setdefault(r["defense"], r["row"])
    with open(a.out / "rows.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["defense", "row"])
        for d in defenses:
            w.writerow([d, rowmap[d]])

    marg = {d: sum(table[(b, d)] for b in behaviours) / len(behaviours) for d in defenses}
    print(f"breach.csv : {len(behaviours)} behaviours x {len(defenses)} defenses "
          f"(attack={a.attack})")
    print(f"marginals  : { {d: round(v, 3) for d, v in marg.items()} }")

    if not a.config:
        print("no --config given, skipping refusal.csv")
        return 0

    ids, matrix = build_refusal(a.run, a.config, defenses)
    if matrix is None:
        return 0
    with open(a.out / "refusal.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(defenses)
        w.writerows(matrix)
    frr = {d: sum(r[i] for r in matrix) / len(matrix) for i, d in enumerate(defenses)}
    print(f"refusal.csv: {len(matrix)} benign prompts x {len(defenses)} defenses")
    print(f"FRR        : { {d: round(v, 3) for d, v in frr.items()} }")

    if a.verify_frr:
        gold = json.load(open(a.run / "analysis.json", encoding="utf-8"))
        gold = gold.get("h3_refusals", {}).get("per_defense_frr", {})
        bad = {d: (frr[d], gold[d]) for d in frr if d in gold and abs(frr[d] - gold[d]) > 1e-12}
        if bad:
            raise SystemExit(f"FRR disagrees with analysis.json: {bad}")
        print(f"verified   : all {len(gold)} FRR values match analysis.json exactly")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
