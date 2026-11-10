"""Check the Design section's method claims against the configs that actually ran.

The numbers audit covers results. This covers the method: generations, population, thresholds,
set sizes, guard-round structure. A mismatch here is worse than a wrong result number, because
a reviewer who opens the repository is checking exactly these.
"""
import json
import re
import sys
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from dcorr.config import load_config  # noqa: E402

TEX = Path(os.environ.get(
    "DCORR_MANUSCRIPT", REPO.parent / "defense-correlation-paper" / "paper_A_mlj" / "main.tex",
)).read_text(encoding="utf-8")
TEX = re.sub(r"(?m)^\s*%.*$", "", TEX)
cfg = load_config(str(REPO / "configs/v2_vicuna.yaml"))

# (what the paper says, the literal it should contain, the config path or computed value)
CHECKS = [
    ("generations", "30 generations", cfg.get_path("attack.adaptive.generations"), 30),
    ("population", "population of\n24", cfg.get_path("attack.adaptive.population"), 24),
    ("elitism", "elitism at one quarter", cfg.get_path("attack.adaptive.elite_frac"), 0.25),
    ("mutation rate", "mutation rate of 0.15",
     cfg.get_path("attack.adaptive.mutation_rate"), 0.15),
    ("gcg suffix length", "20-token suffix", cfg.get_path("attack.adaptive.suffix_n_tokens"), 20),
    ("gcg steps", "150 steps", cfg.get_path("attack.adaptive.steps"), 150),
    ("affirmative prefix", "first 12 tokens",
     cfg.get_path("attack.adaptive.target_prefix_tokens"), 12),
    ("likelihood window", "window 16", cfg.get_path("defenses.ppl_filter.window_tokens"), 16),
    ("calibration FPR", "1\\%", cfg.get_path("calibration.target_fpr"), 0.01),
    ("guard rounds", "checked in rounds",
     cfg.get_path("attack.adaptive.expensive_rounds"), 3),
    ("judge reps", "five\ntimes", cfg.get_path("judge.reps"), 5),
    ("framing pool", "pool of 30 fluent framing", 30, 30),
    ("perturbation q", "$q=4$", cfg.get_path("defenses.smoothllm.n_copies")
     or cfg.get_path("defenses.smoothllm.q"), 4),
]

print(f"{'claim':<22}{'config':>10}{'expected':>10}  in text?")
bad = 0
for name, literal, got, want in CHECKS:
    agree = (got == want) if got is not None else None
    present = literal in TEX
    flag = ""
    if agree is False:
        flag = "  <-- CONFIG DISAGREES"
        bad += 1
    elif agree is None:
        flag = "  <-- config value not found"
    if not present:
        flag += "  <-- literal not in text"
        bad += 1
    print(f"{name:<22}{str(got):>10}{str(want):>10}  {'yes' if present else 'NO '}{flag}")

# set sizes
h = sum(1 for line in (REPO / "data/eval_harmful.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip())
b = sum(1 for line in (REPO / "data/eval_benign.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip())
o = sum(1 for line in (REPO / "data/eval_benign_orbench.jsonl").read_text(
    encoding="utf-8").splitlines() if line.strip())
e = sum(1 for line in (REPO / "data/external_benign_alpacaeval.jsonl").read_text(
    encoding="utf-8").splitlines() if line.strip())
print()
for name, n, literal in (("harmful behaviours", h, "100 harmful prompts"),
                         ("matched benign", b, "100"),
                         ("over-refusal sample", o, "400 prompts drawn from OR-Bench"),
                         ("external corpus", e, "805")):
    present = literal in TEX
    print(f"{name:<22}{n:>10}  text says {literal!r}: {'yes' if present else 'NO'}")
    if not present:
        bad += 1

# defense roster
enabled = cfg.enabled_defenses()
print(f"\nenabled defenses ({len(enabled)}): {enabled}")
if "seven members" not in TEX or "Seven safeguards" not in TEX:
    print("  <-- the text does not say seven in both places")
    bad += 1

print(f"\n{bad} problem(s)" if bad else "\nevery design claim matches the config")
