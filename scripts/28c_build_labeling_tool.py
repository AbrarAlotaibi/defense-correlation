"""Build the single-file labeling tool for the human adjudication.

Reads sheet.csv and retest_sheet.csv from an adjudication directory and writes
labeling_tool.html beside them: one self-contained page, opened from disk in any browser,
with no network access and nothing installed. The page shows one request and one response
at a time, takes the label from the keyboard, saves after every keystroke in the browser's
own storage, and downloads the labels as CSV.

Blinding is kept by construction. The page embeds only the item id, the request and the
response. key.csv (the grader's verdicts and the sampling stratum) is never read here, and
the items are shown in a fixed shuffled order.

Usage:
  python scripts/28c_build_labeling_tool.py --dir results/v2_vicuna/adjudication
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

TEMPLATE = Path(__file__).parent / "templates" / "labeling_tool.html"


def read_csv(p: Path) -> list[dict]:
    with p.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def embed(items: list[dict]) -> str:
    # "</" must not appear inside a script element
    return json.dumps(items, ensure_ascii=False).replace("</", "<\\/")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--rng-seed", type=int, default=20261110)
    a = ap.parse_args()
    d = Path(a.dir)

    main_items = [{"id": r["item_id"], "request": r["behaviour"], "response": r["response"]}
                  for r in read_csv(d / "sheet.csv")]
    random.Random(a.rng_seed).shuffle(main_items)
    retest_path = d / "retest_sheet.csv"
    retest_items = ([{"id": r["retest_id"], "request": r["behaviour"], "response": r["response"]}
                     for r in read_csv(retest_path)] if retest_path.exists() else [])

    page = TEMPLATE.read_text(encoding="utf-8")
    for marker in ("/*__MAIN__*/[]", "/*__RETEST__*/[]"):
        if page.count(marker) != 1:
            raise SystemExit(f"template marker {marker} not found exactly once")
    page = page.replace("/*__MAIN__*/[]", embed(main_items)).replace("/*__RETEST__*/[]", embed(retest_items))
    out = d / "labeling_tool.html"
    out.write_text(page, encoding="utf-8")
    print(f"wrote {out}: {len(main_items)} main items, {len(retest_items)} retest items")


if __name__ == "__main__":
    main()
