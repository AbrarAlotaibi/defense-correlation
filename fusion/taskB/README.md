# Task B — grader variance

Every response sent to StrongREJECT was judged **5 times** at the same temperature with the
same rubric and prompt digest: 1,887 judged responses × 5 = **9,435 judgments**, plus 813
blocked rows that never reach the grader. Raw judgments in
`results/hpc_vicuna_autodan/gold_multi.jsonl`.

```bash
python scripts/19_rejudge_multi.py --config configs/hpc_vicuna_autodan.yaml --reps 5
python scripts/20_task_b_grader_variance.py --run results/hpc_vicuna_autodan --out fusion/taskB
```

Provenance (seed, prompt SHA-256, judge model, family sizes) is in `meta.json`. BH over the
**15** measurable pairs, using permutation p-values so significance is distribution-free and
comparable with Task A.

## B5 — agreement (`B5_agreement.txt`)

| | |
| --- | --- |
| responses judged 5× | 1,887 |
| **unanimous** | **90.46%** |
| any split | 9.54% |
| bare 3–2 majority | **4.24%** |

About one response in ten gets an inconsistent verdict, and one in twenty-four rests on a 3–2
majority.

## The headline: the conclusion survives, the numbers move

**All 15 pairs remain positive and significant under majority-vote labels** (q ≤ 0.0015, every
pair). The correlation result is not a grader artefact.

But the individual values move, and they move **systematically upward**: 13 of 15 φ estimates
are higher under majority vote than under the single judgment the manuscript reports.

| Pair | published | majority | shift |
| --- | --- | --- | --- |
| refusal-prime × probe₁₆ | 0.446 | 0.605 | **+0.159** |
| perplexity × probe₁₆ | 0.594 | 0.736 | **+0.142** |
| refusal-prime × SmoothLLM | 0.557 | 0.682 | +0.125 |
| refusal-prime × probe₈ | 0.458 | 0.574 | +0.116 |
| probe₁₆ × probe₈ | 0.753 | 0.759 | +0.006 |
| perplexity × SmoothLLM | 0.566 | 0.515 | −0.051 |

**Two published values fall outside the grader-only interval entirely** — refusal-prime ×
probe₁₆ (published 0.446, interval [0.473, 0.670]) and refusal-prime × probe₈ (0.458,
[0.479, 0.670]). For those two the reported run drew a judgment set that no resample of the
five reproduces. The other 13 are inside.

**Marginals move too, and one ordering inverts.** probe₈ goes 0.60 → **0.67** while probe₁₆
goes 0.68 → **0.64**, so under majority labels probe₈ is the *weaker* member of the probe row,
reversing the published ordering. refusal-prime moves 0.60 → 0.66. Anything that ranked
defenses by marginal ASR — including the greedy and row-based selection order — should be
re-derived on these labels before being restated.

The undefended adaptive ASR is **0.63** under majority vote, between the 0.58 the manuscript
reports and the 0.64 seen on the second pass. Static: 0.08 plain, 0.27 transfer.

## B4 — the grader-only interval (`B4_grader_intervals.csv`)

Resampling one judgment per response, 1,000 times, with the behaviour set held fixed.

**Mean grader interval width is 0.185 against 0.312 for the behaviour bootstrap — 59%.**
Grader noise is not a rounding concern; it is a comparable source of uncertainty to sampling
behaviours, and the manuscript currently reports only the latter.

The two intervals answer different questions and are given side by side rather than combined.
The behaviour bootstrap covers *a different sample of behaviours*; the grader interval covers
*a different draw from the same grader*. Combining them would assume an independence that
nothing here establishes.

## The caveat should stay, not go

The brief anticipated that Task B would let the manuscript **drop** the warning against
ordering Δ values within 0.05 of each other. **The measurement says the opposite: keep it.**

Of the 74 pairs-of-pairs whose Δ differs by less than 0.05, **67 (91%) have overlapping
grader intervals** — they cannot be ordered. Across all 105 pairs-of-pairs, 66% overlap. The Δ
grader intervals are roughly 0.03–0.05 wide, which is the same scale as the differences the
caveat warns about.

So the caveat is now empirically grounded rather than precautionary, and it should be restated
with this number attached rather than removed.

## Follow-up: what changes when the downstream analyses are re-derived

Task B's marginal shifts are not cosmetic. Re-running the fusion analyses on majority-vote
labels (`fusion/breach_majority.csv`, produced from `gold_multi.jsonl`; the benign side is
untouched because refusals are scored by `RefusalScorer`, not the gold judge) changes two
conclusions the manuscript currently states.

### 1. The row heuristic no longer matches the greedy search

Under single-judgment labels the two procedures selected identical members in identical order
at every size the row rule admits. **Under majority labels they diverge from size 3.**

| Size | row-based (rule 2) | diversity-greedy |
| --- | --- | --- |
| 1 | token-anomaly — 0.34 / 0.09 | token-anomaly — 0.34 / 0.09 |
| 2 | + SmoothLLM — 0.28 / 0.27 | + SmoothLLM — 0.28 / 0.27 |
| 3 | + **probe₁₆** — 0.26 / 0.27 | + **perplexity** — 0.26 / 0.27 |
| 4 | + refusal-prime — 0.25 / **0.44** | + probe₈ — 0.25 / **0.28** |

The cause is the marginal inversion: rule 2 takes the strongest member of each row by marginal
ASR, and under majority labels probe₁₆ (0.64) overtakes probe₈ (0.67), so the row rule now
picks the other probe. At size 4 both reach ASR 0.25, but greedy does it at **0.28 false
refusal against the row rule's 0.44**.

So the claim that the taxonomy costs nothing against a search with full access to the joint
data holds under one labelling and not the other. The honest statement is that the two agreed
on the labels originally reported and diverge on the better-estimated ones, with the row rule
paying ~16 points of false refusal for the same attack success at size 4.

### 2. Difficulty stratification: one surviving pair becomes three

| | single judgment | majority of 5 |
| --- | --- | --- |
| pairs surviving BH (q < 0.05) | **1** | **3** |
| probe₁₆ × probe₈ | OR 158.7, q < 0.001 | OR **11.76**, q = 0.0006 |
| refusal-prime × SmoothLLM | not significant | OR **10.81**, q = 0.0007 |
| perplexity × probe₁₆ | not significant | OR **10.10**, q = 0.0111 |

This cuts both ways and the manuscript should say so rather than pick the convenient half.

**For H1:** the same-row probe pair still survives stratification, so the mechanism-specific
reading is not an artefact of one judgment draw. But its common odds ratio falls from 158.7 to
11.76 — still decisive, no longer an order of magnitude above every alternative. The published
158.7 was inflated by a favourable labelling.

**Against the clean H1/H2 split:** two *cross-row* pairs now also survive. The manuscript's
interpretation — that cross-row correlation is predominantly a shared difficulty gradient
rather than shared mechanism — is weakened. Under the better-estimated labels, 2 of the 13
cross-row pairs retain a mechanism-specific association after difficulty is held fixed.

Neither the H1 verdict nor the composition conclusion changes. What changes is the sharpness
of the same-row/cross-row contrast, which is currently drawn more starkly than the labels
support.

## Files

| File | Contents |
| --- | --- |
| `B3_table10_majority.csv` | Table 10 on majority labels: p1, p2, joint, Δ, φ, φ/φ_max, bootstrap CI, q |
| `B4_grader_intervals.csv` | φ and Δ with grader-only intervals |
| `B5_agreement.txt` | unanimity and 3–2 split rates |
| `meta.json` | seed, prompt digest, majority-vote marginals, undefended ASR |
| `../breach_majority.csv` | per-behaviour breach under majority vote |
| `../fusion_*_majority.csv` | diversity, combination rules and CMH re-derived on those labels |

`φ/φ_max` is included because raw φ understates how tight these pairs are: the marginals cap
the attainable value. probe₁₆ × probe₈ reaches **0.811 of its maximum**, and
perplexity × probe₁₆ **0.805** — both far closer to their ceiling than the raw φ suggests.
