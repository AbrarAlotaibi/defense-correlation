"""Single-annotator adjudication of the automated grader.

Companion to 28_adjudication.py for the case where only one annotator is available.
Two annotators give human-human agreement; one cannot, so this replaces it with a
test-retest check: the same annotator relabels a stratified subset after an interval,
in a new order and under new ids, without seeing the first labels.

  retest  Build retest_sheet.csv (retest_id, behaviour, response, annotator_1_retest_breach)
          and retest_key.csv (retest_id -> item_id) from sheet.csv. Run it now; open the
          retest sheet only after the main sheet is finished, in a later session.

  merge   Copy the labels downloaded from labeling_tool.html (labels_main.csv, and
          labels_retest.csv once it exists) into sheet.csv and retest_sheet.csv, and record
          when each set was labeled. Refuses to overwrite a label that disagrees.

  score   Read annotator_1_breach from sheet.csv and the grader's majority label from
          key.csv. Report, overall and per stratum, with counts and Wilson 95% intervals:
          grader-annotator raw agreement and Cohen's kappa, and the grader's precision and
          recall taking the annotator as the reference. If the retest sheet is filled, also
          report the annotator's own raw agreement and kappa. Writes adjudication_single.json.

The strata oversample the texts the grader found hard, so the overall row describes this
targeted sample, not the corpus. The per-stratum rows are the ones to quote.

Usage:
  python scripts/28b_adjudication_single.py retest --dir results/v2_vicuna/adjudication
  python scripts/28c_build_labeling_tool.py        --dir results/v2_vicuna/adjudication
  python scripts/28b_adjudication_single.py merge  --dir results/v2_vicuna/adjudication
  python scripts/28b_adjudication_single.py score  --dir results/v2_vicuna/adjudication
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path


def read_csv(p: Path) -> list[dict]:
    with p.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(p: Path, cols: list[str], rows: list[dict]) -> None:
    # utf-8-sig so Excel opens it with the right encoding
    with p.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def label(v: str) -> int | None:
    v = (v or "").strip()
    if v in ("1", "1.0"):
        return 1
    if v in ("0", "0.0"):
        return 0
    if v == "":
        return None
    raise ValueError(f"label must be 1 or 0, found {v!r}")


def wilson(k: int, n: int, z: float = 1.959964) -> list[float] | None:
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [max(0.0, c - h), min(1.0, c + h)]


def cohen_kappa(a: list[int], b: list[int]) -> float | None:
    n = len(a)
    if n == 0:
        return None
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    if pe == 1.0:
        return None  # one class only: kappa is undefined, not 1
    return (po - pe) / (1 - pe)


def rate(k: int, n: int) -> dict:
    return {"k": k, "n": n, "rate": (k / n if n else None), "wilson95": wilson(k, n)}


def block(human: list[int], grader: list[int]) -> dict:
    n = len(human)
    agree = sum(h == g for h, g in zip(human, grader))
    tp = sum(h == 1 and g == 1 for h, g in zip(human, grader))
    return {
        "n": n,
        "annotator_breach": rate(sum(human), n),
        "grader_breach": rate(sum(grader), n),
        "raw_agreement": rate(agree, n),
        "kappa_grader_annotator": cohen_kappa(human, grader),
        # annotator as the reference; "reference-confirmed", not ground truth
        "grader_precision": rate(tp, sum(grader)),
        "grader_recall": rate(tp, sum(human)),
    }


def cmd_retest(d: Path, n: int, rng_seed: int) -> None:
    sheet = {r["item_id"]: r for r in read_csv(d / "sheet.csv")}
    key = read_csv(d / "key.csv")
    out_sheet, out_key = d / "retest_sheet.csv", d / "retest_key.csv"
    if out_sheet.exists():
        raise SystemExit(f"{out_sheet} already exists; not overwriting a sheet that may hold labels")
    rng = random.Random(rng_seed)
    by: dict[str, list[str]] = {}
    for k in key:
        by.setdefault(k["stratum"], []).append(k["item_id"])
    total = sum(len(v) for v in by.values())
    picked: list[str] = []
    for stratum, ids in sorted(by.items()):
        take = min(len(ids), max(20, round(n * len(ids) / total)))
        picked += rng.sample(sorted(ids), take)
    rng.shuffle(picked)
    rows, krows = [], []
    for i, item_id in enumerate(picked):
        rid = f"retest_{i:04d}"
        s = sheet[item_id]
        rows.append({"retest_id": rid, "behaviour": s["behaviour"], "response": s["response"],
                     "annotator_1_retest_breach": ""})
        krows.append({"retest_id": rid, "item_id": item_id})
    write_csv(out_sheet, ["retest_id", "behaviour", "response", "annotator_1_retest_breach"], rows)
    write_csv(out_key, ["retest_id", "item_id"], krows)
    print(f"wrote {out_sheet} ({len(rows)} rows) and {out_key}")


def _write_plain(p: Path, rows: list[dict]) -> None:
    # sheet.csv is also read by 28_adjudication.py, which expects no byte-order mark
    with p.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _merge_one(sheet_path: Path, labels_path: Path, id_col: str, label_col: str, note_col: str | None) -> dict:
    sheet = read_csv(sheet_path)
    got = {r[id_col]: r for r in read_csv(labels_path)}
    unknown = set(got) - {r[id_col] for r in sheet}
    if unknown:
        raise SystemExit(f"{labels_path.name} has {len(unknown)} ids that are not in {sheet_path.name}")
    n, times = 0, []
    for r in sheet:
        g = got.get(r[id_col])
        if not g:
            continue
        new = label(g.get("label", ""))
        if new is None:
            continue
        old = label(r.get(label_col, ""))
        if old is not None and old != new:
            raise SystemExit(f"{r[id_col]}: {sheet_path.name} already holds {old}, the download says {new}")
        r[label_col] = str(new)
        if note_col and g.get("note"):
            r[note_col] = g["note"]
        if g.get("labeled_at"):
            times.append(g["labeled_at"])
        n += 1
    _write_plain(sheet_path, sheet)
    return {"labels": n, "rows": len(sheet), "first": min(times) if times else None,
            "last": max(times) if times else None}


def cmd_merge(d: Path) -> None:
    out: dict = {}
    main_labels = d / "labels_main.csv"
    if not main_labels.exists():
        raise SystemExit(f"{main_labels} not found: download it from labeling_tool.html into this folder")
    out["main"] = _merge_one(d / "sheet.csv", main_labels, "item_id", "annotator_1_breach", "annotator_1_notes")
    retest_labels = d / "labels_retest.csv"
    if retest_labels.exists():
        out["retest"] = _merge_one(d / "retest_sheet.csv", retest_labels, "retest_id",
                                   "annotator_1_retest_breach", None)
    (d / "labeling_times.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    for k, v in out.items():
        print(f"{k}: {v['labels']} of {v['rows']} rows labeled, {v['first']} to {v['last']}")


def cmd_score(d: Path) -> None:
    sheet = read_csv(d / "sheet.csv")
    key = {k["item_id"]: k for k in read_csv(d / "key.csv")}
    first: dict[str, int] = {}
    for r in sheet:
        v = label(r.get("annotator_1_breach", ""))
        if v is not None:
            first[r["item_id"]] = v
    missing = len(sheet) - len(first)
    if missing:
        print(f"WARNING: {missing} of {len(sheet)} rows are unlabeled; scoring the {len(first)} labeled rows")
    if not first:
        raise SystemExit("no labels in annotator_1_breach yet")

    out: dict = {"n_sheet": len(sheet), "n_labeled": len(first), "strata": {}}
    strata = sorted({key[i]["stratum"] for i in first})
    for name in strata + ["all (targeted sample, not the corpus)"]:
        ids = [i for i in first if name.startswith("all") or key[i]["stratum"] == name]
        out["strata"][name] = block([first[i] for i in ids], [int(key[i]["grader_breach"]) for i in ids])

    rs, rk = d / "retest_sheet.csv", d / "retest_key.csv"
    if rs.exists() and rk.exists():
        rmap = {r["retest_id"]: r["item_id"] for r in read_csv(rk)}
        a, b = [], []
        for r in read_csv(rs):
            v = label(r.get("annotator_1_retest_breach", ""))
            item = rmap[r["retest_id"]]
            if v is not None and item in first:
                a.append(first[item])
                b.append(v)
        if a:
            out["retest"] = {"n": len(a), "raw_agreement": rate(sum(x == y for x, y in zip(a, b)), len(a)),
                             "kappa_test_retest": cohen_kappa(a, b)}
        else:
            print("retest sheet present but not yet labeled")

    (d / "adjudication_single.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    def f(x):
        return "  n/a" if x is None else f"{x:5.3f}"

    def fr(r):
        if r["rate"] is None:
            return "n/a"
        lo, hi = r["wilson95"]
        return f"{r['rate']:.3f} [{lo:.3f}, {hi:.3f}] ({r['k']}/{r['n']})"

    for name, b in out["strata"].items():
        print(f"\n{name}  n={b['n']}")
        print(f"  agreement  {fr(b['raw_agreement'])}   kappa {f(b['kappa_grader_annotator'])}")
        print(f"  grader precision {fr(b['grader_precision'])}")
        print(f"  grader recall    {fr(b['grader_recall'])}")
    if "retest" in out:
        t = out["retest"]
        print(f"\ntest-retest  n={t['n']}  agreement {fr(t['raw_agreement'])}  kappa {f(t['kappa_test_retest'])}")
    print(f"\nwrote {d / 'adjudication_single.json'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("retest")
    r.add_argument("--dir", required=True)
    r.add_argument("--n", type=int, default=120)
    r.add_argument("--rng-seed", type=int, default=20261110)
    m = sub.add_parser("merge")
    m.add_argument("--dir", required=True)
    s = sub.add_parser("score")
    s.add_argument("--dir", required=True)
    a = ap.parse_args()
    if a.cmd == "retest":
        cmd_retest(Path(a.dir), a.n, a.rng_seed)
    elif a.cmd == "merge":
        cmd_merge(Path(a.dir))
    else:
        cmd_score(Path(a.dir))


if __name__ == "__main__":
    main()
