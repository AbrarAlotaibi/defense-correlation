"""Stage 14 (D3): build an external benign corpus for out-of-sample threshold calibration.

WHY AN EXTERNAL CORPUS, AND WHY THIS ONE.
Every threshold in the study is set on the same 100 JailbreakBench benign prompts that are
later used to report false-refusal rates, so each filter's operating point is chosen
in-sample. At n = 100 a 1% target rate is a single prompt, which is essentially
unidentified. That is REVIEW item M4.

The corpus has to be one that neither the evaluation set nor the probe has seen:

  * data/eval_benign.jsonl (100, jbb_benign) is the in-sample set - the problem itself.
  * XSTest is already in this repo as the benign half of data/probe_train.jsonl (250 rows).
    Calibrating the probe on it would be leakage, since those are its training examples.
  * AlpacaEval (805 general instructions) is seen by neither, and is 8x the evaluation set,
    so a 1% quantile rests on 8 prompts rather than 1.

Deduplicated against BOTH the evaluation benign set and the probe training pool by the same
normalised token Jaccard the probe audit uses, so nothing that survives is contaminated.

Usage:
  python scripts/14_build_external_benign.py --alpaca /tmp/alpaca_eval.json \
      --out data/external_benign_alpacaeval.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

_WORD = re.compile(r"[a-z0-9]+")


def toks(s: str) -> set[str]:
    return set(_WORD.findall(s.lower()))


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def read_jsonl(p: Path) -> list[dict]:
    with open(p, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpaca", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--eval-benign", default=Path("data/eval_benign.jsonl"), type=Path)
    ap.add_argument("--probe-train", default=Path("data/probe_train.jsonl"), type=Path)
    ap.add_argument("--jaccard", type=float, default=0.6)
    ap.add_argument("--max-chars", type=int, default=2000)
    a = ap.parse_args()

    raw = json.loads(a.alpaca.read_text(encoding="utf-8"))
    prompts = [r["instruction"].strip() for r in raw if r.get("instruction", "").strip()]
    # dedup within the corpus itself first
    seen, uniq = set(), []
    for p in prompts:
        k = " ".join(sorted(toks(p)))
        if k not in seen:
            seen.add(k)
            uniq.append(p)
    print(f"AlpacaEval: {len(prompts)} instructions, {len(uniq)} after internal dedup")

    contam = [toks(r["prompt"]) for r in read_jsonl(a.eval_benign)]
    contam += [toks(r["text"]) for r in read_jsonl(a.probe_train)]
    print(f"contamination reference: {len(contam)} texts "
          f"(eval benign + probe training pool, both labels)")

    kept, dropped = [], []
    for p in uniq:
        if len(p) > a.max_chars:
            continue
        t = toks(p)
        worst = max((jaccard(t, c) for c in contam), default=0.0)
        (kept if worst < a.jaccard else dropped).append((p, worst))

    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        for i, (p, w) in enumerate(kept):
            f.write(json.dumps({"benign_id": f"alpaca_{i}", "prompt": p,
                                "category": "general_instruction",
                                "source": "alpaca_eval",
                                "max_jaccard_to_repo_texts": round(w, 3)}) + "\n")

    report = {
        "corpus": "alpaca_eval",
        "n_raw": len(prompts), "n_unique": len(uniq),
        "n_kept": len(kept), "n_dropped_contaminated": len(dropped),
        "jaccard_threshold": a.jaccard,
        "max_jaccard_kept": round(max((w for _, w in kept), default=0.0), 3),
        "deduped_against": ["data/eval_benign.jsonl", "data/probe_train.jsonl"],
        "why": ("XSTest is the benign half of probe_train.jsonl, so calibrating the probe on "
                "it would be leakage; jbb_benign is the in-sample set M4 objects to."),
    }
    Path("data/external_benign_report.json").write_text(json.dumps(report, indent=2),
                                                        encoding="utf-8")
    print(f"kept {len(kept)}, dropped {len(dropped)} as contaminated "
          f"(max Jaccard kept {report['max_jaccard_kept']})")
    print(f"wrote {a.out} and data/external_benign_report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
