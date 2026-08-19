# Classifier-fusion analysis

Recasts the paper's failure-correlation result in the vocabulary of classifier fusion
(Kuncheva and Whitaker, 2003), adds a k-of-n combination-rule curve, and compares the
paper's row-based composition rule against a diversity-greedy selection.

Produced by [`scripts/fusion_analysis.py`](../scripts/fusion_analysis.py) from inputs built
by [`scripts/09_build_fusion_inputs.py`](../scripts/09_build_fusion_inputs.py).

```bash
python scripts/09_build_fusion_inputs.py --run results/hpc_vicuna_autodan --out fusion/
cd fusion
python ../scripts/fusion_analysis.py --breach breach.csv --refusal refusal.csv \
    --rows rows.csv --out fusion_results.tex
python ../scripts/fusion_analysis.py --breach breach.csv --refusal refusal.csv \
    --rows rows.csv --exclude llamaguard --out fusion_results_no_llamaguard.tex
```

Source run: `results/hpc_vicuna_autodan` — Vicuna-7B-v1.5, fluent (AutoDAN-style) adversary,
n = 100 JailbreakBench behaviours, StrongREJECT gold judging, `attack == adaptive`.
`undefended` is dropped: it is the control, not a stack member.

## Status of each number in this directory

| Quantity | Status |
| --- | --- |
| Q, disagreement, double-fault, φ, and its bootstrap CI | **measured** |
| Marginal and joint ASR, k-of-n residual ASR, selection ASR | **measured** |
| Every FRR | **not available** — blanked to `--` |

`breach.csv` is a pivot of `gold.jsonl`, so it carries no new modelling. It reproduces the
published Table 6 exactly: double-fault equals the stored joint breach rate and `DF_indep`
equals the stored independence prediction to floating-point epsilon (max deviation
5.6 × 10⁻¹⁷ across all 21 pairs), and φ agrees at the four decimals `table6.csv` stores.

### Why the FRR column is empty

`fusion_analysis.py` requires `--refusal`, but every ASR and diversity quantity it computes
depends only on `--breach`. The per-prompt benign vectors
(`stage04_<defense>_benign.jsonl`) were written on the cluster during stage 04 and were
never copied back; `results/hpc_vicuna_autodan/` holds only the aggregates. They cannot be
reconstructed from `gold.jsonl`, which covers the harmful split only.

`refusal_PLACEHOLDER_zeros.csv` exists solely to satisfy the required argument. It is
all zeros and carries no information, so every FRR derived from it was blanked by
[`scripts/strip_placeholder_frr.py`](../scripts/strip_placeholder_frr.py), which also stamps
a warning banner on each `.tex`. **Do not un-blank those cells.**

The false-refusal rates that *were* measured are in
`results/hpc_vicuna_autodan/analysis.json` under `h3_refusals`, and are unaffected by any of
this:

| | FRR |
| --- | --- |
| perplexity / probe L16 | 0.08 |
| probe L8 / token anomaly | 0.09 |
| Llama Guard / SmoothLLM | 0.26 |
| refusal prime | 0.39 |
| **assembled stack (measured)** | **0.81** vs 0.766 predicted under independence |

To complete the table: recover the six `stage04_*_benign.jsonl` files from the cluster, pass
`--benign-dir` to `09_build_fusion_inputs.py` to emit a real `refusal.csv`, re-run
`fusion_analysis.py`, and skip the stripping pass.

## Results

Two variants are stored. The **15-pair variant is the one to report**: it is exactly the set
Table 6 reports, and it excludes Llama Guard, whose 0.01 marginal makes its six pairs
degenerate in this vocabulary.

| | all 7 defenses | excluding Llama Guard |
| --- | --- | --- |
| pairs | 21 | **15** |
| Q > 0 | 21 / 21 | **15 / 15** |
| double-fault above independence | 21 / 21 | **15 / 15** |
| Q range | 0.610 – 1.000 | **0.610 – 0.977** |
| double-fault range | 0.010 – 0.580 | 0.270 – 0.580 |
| Spearman Q vs φ | −0.26 | **0.99** |
| Spearman disagreement vs φ | −0.97 | −0.97 |
| Spearman double-fault vs φ | 0.95 | 0.89 |

Three things worth stating in the manuscript:

1. **Every pair is positively associated on every measure.** Q > 0 and double-fault above
   the independence prediction for all 15 pairs, which is the same verdict Table 6 reaches
   through φ, reached independently in the fusion vocabulary.
2. **Llama Guard breaks the Q-statistic, not the finding.** With a marginal of 0.01 the
   `n01 · n10` term collapses and Q saturates at exactly 1.000 for all six of its pairs.
   That degeneracy alone drives the Q vs φ rank agreement from 0.99 down to −0.26. It is an
   artefact of Q's definition at extreme marginals, and is the reason to report the 15-pair
   variant. Disagreement and double-fault are unaffected.
3. **The row heuristic recovers the diversity-greedy selection.** Excluding Llama Guard,
   composition rule 2 and a greedy search that explicitly minimises double-fault choose the
   same members in the same order — token anomaly, SmoothLLM, probe L8, refusal prime —
   with identical ASR at every stack size (0.35 → 0.27 → 0.23 → 0.20). The row heuristic
   costs nothing against a search that optimises the diversity criterion directly.

With Llama Guard included the selection comparison is floor-limited exactly as the script's
`--exclude` help predicts: both rules pin to 0.010 at every size, because one member already
sits at the floor and no combination rule can improve on it.

## Caveat this analysis inherits

Breach vectors come from re-optimising the attack against each defense **separately**. Every
combination rule evaluated here therefore rests on the same intersection assumption used for
the pairwise analysis, which was validated against a direct attack on the assembled stack at
**k = 1 only** (intersection predicts 0.010, direct attack yields 0.000, agreement 99/100).
Every k > 1 row is an estimate and must be labelled as such until a direct attack is run
against that configuration.

## Files

| File | Contents |
| --- | --- |
| `breach.csv` | 100 behaviours × 7 defenses, 1 = breached (measured) |
| `rows.csv` | defense → dependency row, as recorded by the run |
| `refusal_PLACEHOLDER_zeros.csv` | placeholder only — not data |
| `fusion_diversity.csv`, `fusion_combination_rules.csv` | all 7 defenses |
| `fusion_diversity_no_llamaguard.csv`, `fusion_combination_rules_no_llamaguard.csv` | 15-pair variant |
| `fusion_results.tex`, `fusion_results_no_llamaguard.tex` | LaTeX tables + `\newcommand` macros, FRR blanked |
| `run_all7.log`, `run_no_llamaguard.log` | full console output of both runs |
