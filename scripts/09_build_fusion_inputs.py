"""Stage 09: emit the fusion-analysis inputs from a run's gold verdicts.

scripts/fusion_analysis.py wants a behaviours x defenses breach matrix and a
defense -> dependency-row map. Both are already implied by results/<run>/gold.jsonl;
this script just pivots them, so no number is transcribed by hand.

  breach.csv   one row per behaviour, one column per defense, 1 = breached under
               the chosen attack (default `adaptive`, the paper's fluent adversary).
               `undefended` is dropped: it is the control, not a stack member.
  rows.csv     defense,row  taken from the `row` field the run itself recorded.

The benign side (refusal.csv) is NOT derivable from gold.jsonl: gold judging covers
the harmful split only. It has to come from stage04_<defense>_benign.jsonl, which is
written by stage 04. See --benign-dir.

Usage:
  python scripts/09_build_fusion_inputs.py --run results/hpc_vicuna_autodan --out fusion/
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_jsonl(p: Path) -> list[dict]:
    with open(p, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_breach(rows: list[dict], attack: str, drop: set[str]):
    """Pivot to {behaviour_id: {defense: 0/1}}, asserting a complete rectangle."""
    sel = [r for r in rows if r["attack"] == attack and r["defense"] not in drop]
    if not sel:
        raise SystemExit(f"no rows for attack={attack!r}")
    defenses = sorted({r["defense"] for r in sel})
    behaviours = sorted({r["behaviour_id"] for r in sel},
                        key=lambda s: (len(s), s))   # jbb_2 before jbb_10
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--attack", default="adaptive")
    ap.add_argument("--drop", default="undefended,stack")
    ap.add_argument("--benign-dir", type=Path,
                    help="directory holding stage04_<defense>_benign.jsonl; "
                         "emits refusal.csv when present")
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

    rowmap = {}
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
    print(f"rows.csv   : { rowmap }")

    if a.benign_dir:
        per = {}
        for d in defenses:
            p = a.benign_dir / f"stage04_{d}_benign.jsonl"
            if not p.is_file():
                print(f"[skip] refusal.csv: {p.name} not found")
                return 0
            per[d] = read_jsonl(p)
        n = len(next(iter(per.values())))
        with open(a.out / "refusal.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(defenses)
            for i in range(n):
                w.writerow([int(bool(per[d][i].get("blocked") or per[d][i].get("refused")))
                            for d in defenses])
        print(f"refusal.csv: {n} benign prompts x {len(defenses)} defenses")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
