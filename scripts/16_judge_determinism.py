"""Stage 16: how reproducible is the gold judge?

An accidental replicate. The D2 run (hpc_vicuna_autodan_nolg) was seeded with a copy of the
primary run's stage04_undefended.jsonl, so stage 06 judged the SAME 300 undefended responses
a second time - same rubric, same model, same temperature 0, same binarisation threshold.
Comparing the two verdict sets measures run-to-run judge noise directly, which no planned
part of the study does: the bootstrap intervals resample behaviours and say nothing about
the grader.

This matters because every ASR in the paper is a judge output, so judge noise is a floor on
the precision of all of them, and it is not included in any reported interval.

Usage:
  python scripts/16_judge_determinism.py --a results/hpc_vicuna_autodan \
      --b results/hpc_vicuna_autodan_nolg --defense undefended --out fusion/manuscript
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(p: Path, defense: str) -> dict:
    out = {}
    with open(p / "gold.jsonl", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("defense") == defense:
                out[(r["behaviour_id"], r["attack"])] = r
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, type=Path)
    ap.add_argument("--b", required=True, type=Path)
    ap.add_argument("--defense", default="undefended")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    A, B = load(args.a, args.defense), load(args.b, args.defense)
    common = sorted(set(A) & set(B))
    if not common:
        raise SystemExit("no shared rows; are these the same responses?")

    res = {"defense": args.defense, "n_rows": len(common),
           "run_a": str(args.a), "run_b": str(args.b), "by_attack": {}}
    dis_all = [k for k in common if A[k]["breach"] != B[k]["breach"]]

    for atk in sorted({k[1] for k in common}):
        c = [k for k in common if k[1] == atk]
        a = sum(A[k]["breach"] for k in c)
        b = sum(B[k]["breach"] for k in c)
        d = [k for k in c if A[k]["breach"] != B[k]["breach"]]
        res["by_attack"][atk] = {
            "n": len(c), "asr_run_a": a / len(c), "asr_run_b": b / len(c),
            "asr_abs_diff": abs(a - b) / len(c),
            "verdict_disagreements": len(d),
            "disagreement_rate": len(d) / len(c),
        }

    parse_a = sum(1 for k in common if not A[k].get("parse_ok"))
    parse_b = sum(1 for k in common if not B[k].get("parse_ok"))
    from_parse = [k for k in dis_all
                  if not A[k].get("parse_ok") or not B[k].get("parse_ok")]
    res["overall"] = {
        "verdict_disagreements": len(dis_all),
        "disagreement_rate": len(dis_all) / len(common),
        "unparsed_run_a": parse_a, "unparsed_run_b": parse_b,
        "disagreements_involving_a_parse_failure": len(from_parse),
        "note": ("Same responses, same rubric, same model at temperature 0. Disagreement is "
                 "judge non-determinism, not a difference in the thing being measured. Only "
                 f"{len(from_parse)} of {len(dis_all)} are attributable to a parse failure."),
    }

    (args.out / "judge_determinism.json").write_text(json.dumps(res, indent=2),
                                                     encoding="utf-8")
    print(f"identical responses judged twice: {len(common)} rows")
    print(f"{'attack':16} {'ASR a':>7} {'ASR b':>7} {'|diff|':>7} {'disagree':>9}")
    for atk, v in res["by_attack"].items():
        print(f"{atk:16} {v['asr_run_a']:7.2f} {v['asr_run_b']:7.2f} "
              f"{v['asr_abs_diff']:7.2f} {v['verdict_disagreements']:6d} "
              f"({v['disagreement_rate']:.0%})")
    o = res["overall"]
    print(f"\noverall disagreement {o['verdict_disagreements']}/{len(common)} = "
          f"{o['disagreement_rate']:.1%}; only "
          f"{o['disagreements_involving_a_parse_failure']} involve a parse failure")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
