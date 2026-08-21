# Task D — external threshold calibration

The primary configuration re-run with every filter threshold set on **805 AlpacaEval
instructions** (disjoint from the evaluation set and from the probe's training pool) instead
of in-sample on the 100 evaluation benign prompts. Job 2316332, `COMPLETED` in 7:34:49, all 7
defenses plus the assembled stack, 100/100 behaviours, no fatal errors, judged post hoc.

```bash
python scripts/22_task_d_external.py --primary results/hpc_vicuna_autodan \
    --external results/hpc_vicuna_autodan_extthr \
    --config configs/hpc_vicuna_autodan.yaml --out fusion/taskD
```

Both threshold sets are reported side by side; neither replaces the other.

## Operating points

| Defense | thr in-sample | thr external | FRR in → ext | ASR in → ext |
| --- | --- | --- | --- | --- |
| perplexity | 4.849 | 6.516 | 0.08 → 0.08 | 0.66 → 0.67 |
| token-anomaly | 0.273 | 0.523 | 0.09 → 0.08 | **0.35 → 0.64** |
| **probe₁₆** | 28.924 | **−5.210** | **0.08 → 0.26** | 0.68 → 0.70 |
| probe₈ | 110.768 | 100.593 | 0.09 → 0.15 | 0.60 → 0.66 |
| Llama Guard / refusal-prime / SmoothLLM | no threshold | — | unchanged / 0.26→0.29 | 0.01 / 0.60→0.65 / 0.54→0.59 |

**The predicted probe₁₆ blow-up is confirmed exactly.** Calibrating out of sample moves its
false-refusal rate from 0.08 to **0.26**, the figure the calibration pass forecast.

**token-anomaly is the bigger surprise.** Its external threshold is far more permissive, so it
blocks much less and its residual ASR nearly doubles, 0.35 → 0.64. The in-sample threshold was
not merely optimistic about false refusals; it was doing most of that filter's work.

## The correlation result survives recalibration

**All 15 pairs remain positively correlated and significant** (q ≤ 5.0 × 10⁻⁴ throughout).
Several associations are *stronger* under external thresholds — perplexity × probe₈ reaches
φ = 0.843 and perplexity × token-anomaly 0.759, against 0.621 and 0.350 in-sample.

The composition conclusion — Δ > 0 everywhere, layered defenses do not compound — does not
depend on how the thresholds were set.

## The H1 mechanism claim does not transfer, but not because it was refuted

**probe₁₆ × probe₈ has φ = 0.590 [0.413, 0.751], q = 5.0 × 10⁻⁴** under external thresholds —
still strongly positive, down from 0.753 in-sample.

Its **CMH stratification becomes undefined**, and this needs stating carefully because the
software returns a number that looks like a refutation:

```
probe16 x probe8 : CMH OR 0.000, q 0.959
```

That 0.000 is **not evidence of no association**. Every one of the five difficulty strata has
an empty concordant cell, so the CMH numerator Σ(a·d/n) is identically zero:

| stratum (other defenses breached) | n | a | b | c | d |
| --- | --- | --- | --- | --- | --- |
| 0 | 22 | **0** | 2 | 1 | 19 |
| 1 | 7 | **0** | 1 | 2 | 4 |
| 2 | 7 | 4 | 3 | 0 | **0** |
| 3 | 22 | 18 | 3 | 1 | **0** |
| 4 | 42 | 37 | 2 | 3 | **0** |

In the easy strata neither probe is breached; in the hard strata both are. There is no
stratum containing both concordant outcomes, so there is nothing for the statistic to measure.
**It is exactly the untestability the manuscript already documents for the semantic row** —
extreme marginals destroying a diagnostic — now landing on the probe pair. Of the 15 pairs,
this is the *only* one affected.

**And the pair that does survive is cross-row.** Under external thresholds the single pair
clearing BH after stratification is **perplexity × probe₈** (CMH OR 65.2, q = 2.2 × 10⁻⁶) —
token surface × internal representations, a *cross-row* pair. Under in-sample thresholds the
sole survivor was the same-row probe pair.

So the same-row/cross-row story does not merely weaken under recalibration; the surviving pair
switches sides.

## What the manuscript should conclude

1. **The composition result is robust.** It holds under both threshold regimes, and the
   associations are if anything stronger out of sample.
2. **The mechanism-specific reading of H1 cannot be adjudicated on this data.** In-sample it
   rests on one post-hoc pair; out of sample the test for that pair is undefined and a
   cross-row pair takes its place. The honest move is to retire the mechanism claim as
   unsupported rather than report either regime's answer as the finding.
3. **probe₁₆'s operating point does not transfer**, confirmed at 0.26 against 0.08, and
   **token-anomaly's in-sample threshold was carrying its measured effectiveness**.

## Incidental cross-check

Llama Guard, refusal-prime and SmoothLLM have no thresholds, so their configuration is
identical across the two runs, yet their ASRs shift by up to +0.05 (refusal-prime 0.60 → 0.65,
SmoothLLM 0.54 → 0.59). That is within the grader-noise envelope Task B measured
independently (refusal-prime moved 0.60 → 0.66 under majority vote), so the two tasks
corroborate each other on the size of the noise floor.

## Files

| File | Contents |
| --- | --- |
| `D3_external_thresholds.csv` | both threshold sets, FRR and residual ASR side by side |
| `D3_table10_external.csv` | Table 10 recomputed under external thresholds |
| `D3_cmh_external.csv` | difficulty-stratified CMH under external thresholds |
| `D3_verdict.json` | the probe-pair verdict, informative-strata count, degenerate pairs |
