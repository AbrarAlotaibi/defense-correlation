"""Stage 13: analyse the two runs submitted as B3 and D2.

B3 - attributable false refusal.
    The per-defense FRRs of H3 are raw: a defense at 0.08 has not been separated from a
    model that already refuses some of those prompts unprompted. With the undefended
    benign sweep we can report the floor and the attributable component, where
    attributable means refused by the defended configuration but NOT by the bare model.
    Refusal is the same RefusalScorer call 07_analyze.py uses, so the floor is defined
    identically to every FRR in the paper.

D2 - intersection versus direct, with resolution.
    The primary check compares a predicted 0.010 against a measured 0.000, both on the
    floor. Removing Llama Guard leaves six defenses whose intersection predicts 0.190, so
    agreement or disagreement is now informative. This reports the predicted residual, the
    measured direct residual, and the per-behaviour agreement count.

Usage:
  python scripts/13_b3_d2_analysis.py --primary results/hpc_vicuna_autodan \
      --nolg results/hpc_vicuna_autodan_nolg --config configs/hpc_vicuna_autodan.yaml \
      --prediction fusion/manuscript/D2_intersection_prediction.json \
      --out fusion/manuscript
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dcorr.config import load_config          # noqa: E402
from dcorr.io_utils import read_jsonl          # noqa: E402
from dcorr.judge import RefusalScorer          # noqa: E402


def b3(primary: Path, config: Path, out: Path) -> dict | None:
    p = primary / "stage04_undefended_benign.jsonl"
    if not p.is_file():
        print(f"[B3] {p.name} not present yet - skipping")
        return None
    cfg = load_config(str(config))
    scorer = RefusalScorer(
        max_response_length=int(cfg.get_path("refusal.max_response_length", 600)),
        model_name=cfg.get_path("target.model_id"),
    )

    def refused(rows):
        return {r["benign_id"]: bool(scorer.is_refusal(r.get("response", ""),
                                                       r.get("blocked", False)))
                for r in rows}

    base = refused(read_jsonl(p))
    floor = sum(base.values()) / len(base)
    res = {"n_benign": len(base), "undefended_floor": round(floor, 3), "per_defense": {}}

    for f in sorted(primary.glob("stage04_*_benign.jsonl")):
        name = f.name[len("stage04_"):-len("_benign.jsonl")]
        if name == "undefended":
            continue
        d = refused(read_jsonl(f))
        common = [k for k in d if k in base]
        raw = sum(d[k] for k in common) / len(common)
        # attributable: refused by the deployed configuration on a prompt the bare model
        # did NOT refuse. Denominator is the non-floor prompts, which is the population the
        # defense can actually add refusals to.
        elig = [k for k in common if not base[k]]
        attributable = sum(d[k] for k in elig) / len(elig) if elig else float("nan")
        # how much of the raw rate is just the floor showing through
        overlap = sum(d[k] and base[k] for k in common) / len(common)
        res["per_defense"][name] = {
            "raw_frr": round(raw, 3),
            "attributable_frr": round(attributable, 3),
            "shared_with_floor": round(overlap, 3),
            "n_eligible": len(elig),
        }

    stack = primary / "stage05_stack_benign.jsonl"
    if stack.is_file():
        d = refused(read_jsonl(stack))
        common = [k for k in d if k in base]
        elig = [k for k in common if not base[k]]
        res["stack"] = {
            "raw_frr": round(sum(d[k] for k in common) / len(common), 3),
            "attributable_frr": round(sum(d[k] for k in elig) / len(elig), 3) if elig else None,
        }

    (out / "B3_attributable_refusal.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"[B3] undefended floor: {res['undefended_floor']:.3f} on n={res['n_benign']}")
    print(f"{'defense':16} {'raw':>7} {'attributable':>13} {'shared w/ floor':>16}")
    for k, v in sorted(res["per_defense"].items(), key=lambda t: -t[1]["raw_frr"]):
        print(f"{k:16} {v['raw_frr']:7.3f} {v['attributable_frr']:13.3f} {v['shared_with_floor']:16.3f}")
    if "stack" in res:
        print(f"{'stack':16} {res['stack']['raw_frr']:7.3f} {res['stack']['attributable_frr']:13.3f}")
    return res


def d2(nolg: Path, prediction: Path, out: Path) -> dict | None:
    g = nolg / "gold.jsonl"
    if not g.is_file():
        print(f"[D2] {g} not present yet - run stage 06 (judge) on a login node first")
        return None
    pred = json.loads(prediction.read_text(encoding="utf-8"))
    rows = [r for r in read_jsonl(g)
            if r.get("defense") == "stack" and r.get("attack") == "adaptive"]
    if not rows:
        print("[D2] gold.jsonl holds no adaptive stack rows")
        return None
    meas = {r["behaviour_id"]: int(bool(r["breach"])) for r in rows}
    common = sorted(set(meas) & set(pred), key=lambda s: (len(s), s))
    agree = sum(1 for b in common if meas[b] == pred[b])
    pr = sum(pred[b] for b in common) / len(common)
    mr = sum(meas[b] for b in common) / len(common)
    res = {
        "n": len(common),
        "predicted_intersection_residual": round(pr, 3),
        "measured_direct_residual": round(mr, 3),
        "difference": round(mr - pr, 3),
        "per_behaviour_agreement": f"{agree}/{len(common)}",
        "agreement_rate": round(agree / len(common), 3),
        "predicted_breach_only": sum(1 for b in common if pred[b] and not meas[b]),
        "measured_breach_only": sum(1 for b in common if meas[b] and not pred[b]),
        "_note": ("Intersection is over the six non-Llama-Guard defenses from the primary "
                  "per-defense vectors; direct is the adaptive attack on the assembled "
                  "six-layer stack."),
    }
    (out / "D2_intersection_vs_direct.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"[D2] predicted {pr:.3f} vs measured {mr:.3f} (difference {mr - pr:+.3f})")
    print(f"[D2] per-behaviour agreement {agree}/{len(common)} = {agree / len(common):.3f}")
    print(f"[D2] predicted-only breaches {res['predicted_breach_only']}, "
          f"measured-only {res['measured_breach_only']}")
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--primary", required=True, type=Path)
    ap.add_argument("--nolg", required=True, type=Path)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--prediction", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    print("=== B3: attributable false refusal ===")
    b3(a.primary, a.config, a.out)
    print("\n=== D2: intersection versus direct ===")
    d2(a.nolg, a.prediction, a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
