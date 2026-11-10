"""Build the enlarged benign evaluation set (plan Run D; REVIEW R5-5).

At n=100 benign prompts a 1% filter block rate is one prompt, and every false-refusal rate
in v1 carries an interval about 0.1 wide. This adds a held-out sample of OR-Bench-Hard-1K
(Cui et al., 2024; CC-BY-4.0): benign prompts that look harmful, built to measure
over-refusal - the regime where filters calibrated on easy benign text fail.

Contamination control, as in scripts/14_build_external_benign.py: a prompt is dropped when
its token Jaccard with any repo text reaches 0.6 - the evaluation behaviours (harmful and
benign), the probe training pool (whose benign half is XSTest, so XSTest itself cannot be
used here) and the AlpacaEval calibration corpus that set the v2 thresholds.

Writes data/eval_benign_orbench.jsonl and data/eval_benign_orbench_report.json.

Usage: python scripts/27_build_benign_orbench.py --n 400
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
from pathlib import Path

_WORD = re.compile(r"[a-z0-9]+")
REPO = Path(__file__).resolve().parent.parent
DATASET, FILE = "bench-llm/or-bench", "or-bench-hard-1k.csv"


def toks(s: str) -> set[str]:
    return set(_WORD.findall(s.lower()))


def jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def read_jsonl(p: Path) -> list[dict]:
    with open(p, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400, help="prompts to keep after dedup")
    ap.add_argument("--seed", type=int, default=20261011)
    ap.add_argument("--jaccard", type=float, default=0.6)
    ap.add_argument("--csv", type=Path, default=None, help="local copy (default: download)")
    a = ap.parse_args()

    if a.csv is None:
        from huggingface_hub import hf_hub_download
        a.csv = Path(hf_hub_download(DATASET, FILE, repo_type="dataset"))
    with open(a.csv, encoding="utf-8", newline="") as f:
        raw = [(r["prompt"].strip(), r.get("category", "")) for r in csv.DictReader(f)
               if r.get("prompt", "").strip()]

    seen, uniq = set(), []
    for p, c in raw:
        k = " ".join(sorted(toks(p)))
        if k not in seen:
            seen.add(k)
            uniq.append((p, c))

    data = REPO / "data"
    ref = [toks(r["prompt"]) for r in read_jsonl(data / "eval_benign.jsonl")]
    ref += [toks(r["prompt"]) for r in read_jsonl(data / "eval_harmful.jsonl")]
    ref += [toks(r["text"]) for r in read_jsonl(data / "probe_train.jsonl")]
    ref += [toks(r["prompt"]) for r in read_jsonl(data / "external_benign_alpacaeval.jsonl")]

    kept, dropped = [], 0
    for p, c in uniq:
        t = toks(p)
        worst = max((jaccard(t, r) for r in ref), default=0.0)
        if worst < a.jaccard:
            kept.append((p, c, worst))
        else:
            dropped += 1
    if len(kept) < a.n:
        raise SystemExit(f"only {len(kept)} clean prompts, fewer than --n {a.n}")
    sample = random.Random(a.seed).sample(kept, a.n)

    out = data / "eval_benign_orbench.jsonl"
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        for i, (p, c, w) in enumerate(sample):
            f.write(json.dumps({"benign_id": f"orb_{i}", "prompt": p, "category": c,
                                "source": "or-bench-hard-1k",
                                "max_jaccard_to_repo_texts": round(w, 3)}) + "\n")
    report = {
        "dataset": DATASET, "file": FILE, "license": "CC-BY-4.0",
        "n_raw": len(raw), "n_unique": len(uniq), "n_clean": len(kept),
        "n_dropped_contaminated": dropped, "n_sampled": a.n, "sample_seed": a.seed,
        "jaccard_threshold": a.jaccard,
        "deduped_against": ["data/eval_benign.jsonl", "data/eval_harmful.jsonl",
                            "data/probe_train.jsonl (benign half = XSTest)",
                            "data/external_benign_alpacaeval.jsonl (v2 calibration)"],
        "categories": sorted({c for _, c, _ in sample}),
    }
    (data / "eval_benign_orbench_report.json").write_text(json.dumps(report, indent=2) + "\n",
                                                          encoding="utf-8")
    print(f"[27] {len(raw)} raw, {len(uniq)} unique, {len(kept)} clean "
          f"({dropped} dropped as contaminated); sampled {a.n} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
