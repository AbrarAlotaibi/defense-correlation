"""Stage 10: emit the cross-evaluation breach matrix for scripts/input_level_phi.py.

That script distinguishes two estimands. phi_behaviour scores each defense on prompts
optimised against itself, which is what Table 6 reports. phi_input scores both members of a
pair on the *same* prompt, which is the quantity Eq. (9) is written over. Telling them apart
needs a long CSV of (source, defense, behavior, breach).

WHAT THIS RUN CAN AND CANNOT SUPPLY
-----------------------------------
Tier 0 only, and it is free. gold.jsonl holds three attacks per defense:

  adaptive      prompts optimised against that defense -> the DIAGONAL of the
                cross matrix, emitted as source="adaptive:<defense>". There is no
                off-diagonal: prompts optimised against d1 were never scored on d2.
  static_plain  the unmodified behaviour prompt
  static_gcg    one fixed transfer suffix, appended to the behaviour prompt

Both static attacks are shared-input by construction: dcorr/attacks/static.py builds them
from the behaviour and a single run-wide suffix and never consults the defense, so the prompt
is byte-identical across all seven. They are emitted as source="static:plain" and
source="static:transfer" and give a genuine input-level phi today.

Tiers 1 and 2 need new scoring passes on a GPU and are not derivable from stored artifacts;
input_level_phi.py will report their columns as "--".

Usage:
  python scripts/10_build_cross_breach.py --run results/hpc_vicuna_autodan \
      --out fusion/cross_breach.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

# gold.jsonl attack label -> source label understood by input_level_phi.py.
# "adaptive" is special-cased: it becomes "adaptive:<defense>", the diagonal.
SHARED_SOURCES = {"static_plain": "static:plain", "static_gcg": "static:transfer"}


def read_jsonl(p: Path) -> list[dict]:
    with open(p, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--drop", default="undefended,stack")
    a = ap.parse_args()

    drop = {s.strip() for s in a.drop.split(",") if s.strip()}
    rows = [r for r in read_jsonl(a.run / "gold.jsonl") if r["defense"] not in drop]
    defenses = sorted({r["defense"] for r in rows})

    out: list[dict] = []
    for r in rows:
        atk = r["attack"]
        if atk == "adaptive":
            source = f"adaptive:{r['defense']}"
        elif atk in SHARED_SOURCES:
            source = SHARED_SOURCES[atk]
        else:
            continue
        out.append({"source": source, "defense": r["defense"],
                    "behavior": r["behaviour_id"], "breach": int(bool(r["breach"]))})

    seen = {(r["source"], r["defense"], r["behavior"]) for r in out}
    if len(seen) != len(out):
        raise SystemExit(f"{len(out) - len(seen)} duplicate (source, defense, behaviour) rows")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["source", "defense", "behavior", "breach"])
        w.writeheader()
        w.writerows(out)

    by_source: dict[str, set[str]] = {}
    for r in out:
        by_source.setdefault(r["source"], set()).add(r["defense"])
    print(f"wrote {a.out}: {len(out)} rows, {len(defenses)} defenses")
    for s in sorted(by_source):
        n = len(by_source[s])
        kind = "diagonal only" if s.startswith("adaptive:") else f"SHARED across {n}"
        print(f"  {s:28} {n} defense(s)  [{kind}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
