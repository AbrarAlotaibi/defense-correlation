# Response to the analysis brief (closing the variance gap)

Tasks A, B and D are complete. Task C was not run, by decision. Every number below is
regenerable from artifacts in this repository.

**Conventions.** Primary configuration throughout (Vicuna-7B-v1.5, fluent AutoDAN-style
adversary, n = 100 JailbreakBench behaviours, 7 defenses, StrongREJECT at the standard
binarisation threshold). **McNemar is the exact two-sided binomial on the discordant cells**
(`scipy.stats.binomtest`) — not the chi-square, not the continuity-corrected variant.
Multiplicity family sizes are stated per analysis: **Holm over 21** for the full pairwise
matrix, **BH over 15** for the measurable-pair analyses. Odds ratios use Haldane–Anscombe
+0.5 applied uniformly. φ to three decimals; p below 0.001 in scientific notation. Seed,
grader-prompt SHA-256 and model are in each output directory's `meta.json`.

---

## Task A — analyses on existing data · `fusion/taskA/`

| Requested | File |
| --- | --- |
| pair, n01, n10, p_exact, p_holm | `A1_mcnemar.csv` |
| defense, n_defense_only, n_stack_only, p_exact | `A1_vs_stack.csv` |
| Q, df, p | `A2_cochran.txt` |
| pair, phi, p_perm | `A3_permutation.csv` |
| re-judge concordance | `A4_rejudge.csv` |

**The requested one-line answer: 11 of 21 pairs differ significantly after Holm** (14
uncorrected), and **yes — Llama Guard dominates every other member pairwise**: there is no
behaviour on which Llama Guard is breached while another defense holds.

Cochran's Q = 193.407, df = 6, p = 4.8 × 10⁻³⁹. All 15 permutation tests significant, agreeing
with the existing bootstrap on significance for **15/15** pairs.

Two results worth using:

- **Llama Guard vs the seven-layer stack: n01 = 1, n10 = 0, p = 1.0** — statistically
  indistinguishable. This is the proper form of the "one strong layer dominates" claim; the
  manuscript's existing McNemar is against the probe, the *weakest* available contrast.
- **probe₁₆ × probe₈ does not separate after Holm** (0.039 → 0.309) despite carrying the
  highest failure correlation. Indistinguishable marginals and correlated failures are
  separate claims.

**A4 is not computable and the file says so** rather than returning an empty result. The
stored re-judge covers 300 responses that are all `defense = undefended`, so it contains no
per-defense breach vector and no pair φ can be recomputed. Task B is what answers it.

---

## Task B — grader variance · `fusion/taskB/`

9,435 judgments: every one of the 1,887 graded responses scored 5 times under an unchanged
rubric, model and temperature. (813 further rows are blocked and never reach the grader.)

| Requested | File |
| --- | --- |
| Table 10 on majority labels | `B3_table10_majority.csv` (adds `phi_over_phi_max`) |
| φ and Δ with grader-only intervals | `B4_grader_intervals.csv` |
| agreement rates | `B5_agreement.txt` |

**Agreement: 90.46% unanimous, 9.54% split, 4.24% resting on a bare 3–2 majority.**

**Requested extras.** Undefended ASR under majority vote: **0.63** adaptive (0.08 plain, 0.27
transfer). The seven marginal residual ASRs under majority vote: perplexity **0.68**,
token-anomaly **0.34**, Llama Guard **0.01**, refusal-prime **0.66**, SmoothLLM **0.56**,
probe₁₆ **0.64**, probe₈ **0.67**.

**All 15 pairs stay positive and significant** (q ≤ 1.5 × 10⁻³), so the correlation result is
not a grader artefact. But 13 of 15 φ estimates rise under majority labels, by up to +0.159,
and two published values fall outside the grader interval entirely.

**The grader-only interval averages 0.185 against 0.312 for the behaviour bootstrap — 59%.**
They are reported side by side, not combined: one covers a different sample of behaviours, the
other a different draw from the grader, and combining them would assume an independence
nothing here establishes.

> **One expectation in the brief is contradicted.** It anticipated that Task B would let the
> manuscript **drop** the warning against ordering Δ values within 0.05. The measurement says
> keep it: of the 74 pairs-of-pairs whose Δ differs by less than 0.05, **67 (91%) have
> overlapping grader intervals**. The caveat is now empirically grounded rather than
> precautionary.

**Downstream effects** (`fusion/breach_majority.csv`, `fusion/fusion_*_majority.csv`):
probe₈ overtakes probe₁₆ on marginal ASR, which breaks the row-vs-greedy agreement from size 3
onward; and difficulty stratification goes from **1 surviving pair to 3**, with the probe
pair's common odds ratio falling from 158.7 to 11.76.

---

## Task C — attack-seed variance · not run

Skipped by decision. The single-seed limitation remains stated accurately in Limitations.

Partial incidental evidence exists: Llama Guard, refusal-prime and SmoothLLM carry no
thresholds and are configured identically across the primary and external-threshold runs, yet
their ASRs shift by up to +0.05 — inside the grader-noise envelope Task B measured
independently. That is consistent with judge noise rather than a separate seed effect, and it
is not a substitute for C.

---

## Task D — threshold transfer · `fusion/taskD/`

Full re-run of the primary configuration (7:34:49 on one H100, exit 0) with every threshold set
on 805 external AlpacaEval instructions, disjoint from both the evaluation set and the probe's
training pool.

| Requested | File |
| --- | --- |
| both threshold sets, block rates, residual ASR | `D3_external_thresholds.csv` |
| Table 10 under external thresholds | `D3_table10_external.csv` |
| CMH under external thresholds | `D3_cmh_external.csv` |
| the probe-pair verdict | `D3_verdict.json` |

**The requested answer — and it is neither "yes" nor "no".** probe₁₆ × probe₈ keeps a strong
raw association (φ = 0.590 [0.413, 0.751], q = 5 × 10⁻⁴), but its **stratified test becomes
undefined**. The CMH routine returns an odds ratio of 0.000, which is *not* evidence of no
association: every one of the five difficulty strata has an empty concordant cell — in the easy
strata neither probe is breached, in the hard strata both are — so the numerator is identically
zero. Of the 15 pairs it strikes only this one. `D3_verdict.json` records `UNDEFINED` rather
than `False` for exactly this reason.

**And the surviving pair switches sides.** Under external thresholds the single pair clearing
BH after stratification is **perplexity × probe₈ — a cross-row pair** (CMH OR 65.2,
q = 2.2 × 10⁻⁶). In-sample the sole survivor was the same-row probe pair.

Operating points:

| Defense | thr in-sample | thr external | FRR in → ext | ASR in → ext |
| --- | --- | --- | --- | --- |
| perplexity | 4.849 | 6.516 | 0.08 → 0.08 | 0.66 → 0.67 |
| token-anomaly | 0.273 | 0.523 | 0.09 → 0.08 | **0.35 → 0.64** |
| **probe₁₆** | 28.924 | **−5.210** | **0.08 → 0.26** | 0.68 → 0.70 |
| probe₈ | 110.768 | 100.593 | 0.09 → 0.15 | 0.60 → 0.66 |

The predicted probe₁₆ blow-up is confirmed exactly at 0.26. **token-anomaly is the larger
surprise**: its in-sample threshold was carrying most of its measured effectiveness, not
merely flattering its refusal rate — relaxed out of sample its residual ASR nearly doubles.

**All 15 pairs remain positive and significant** under external thresholds (q ≤ 5 × 10⁻⁴),
several more strongly than in-sample.

---

## Two confirmations requested by the writer

**The 66 parse failures are the SINGLE-JUDGMENT count, and scoping them that way is
necessary, not merely careful.** Directly counted from `results/hpc_vicuna_autodan/gold.jsonl`:
1,887 responses judged, 66 unparsed = 3.50% of judged, 2.44% of all 2,700 rows. The
majority-vote equivalent, computed over the five repetitions, is **72** — they differ, so the
figure must be labelled by regime.

1,887 is the direct count; 1,885 does not correspond to any count in the run.

Worth one clause in the text: across the five repetitions the unparsed count ran **68, 70, 73,
73, 79** (mean 72.6). The reported 66 is *below every repetition*, so 2.4% sits at the
favourable end of the distribution rather than in the middle of it.

**Table 10 should stay on single-judgment values.** The conservatism argument holds, but the
decisive reason is consistency: Table 10's φ values are not standalone. The difficulty-
stratified analysis, all four figures, the McNemar comparisons, the intersection check and the
fusion recast all derive from the same single-judgment breach vectors. Swapping Table 10 alone
would desynchronise it from Table 12 and every figure; swapping everything would additionally
move the confound result from one surviving pair to three, reopening the H1/H2 narrative.

One scoping fix, though: the published table is the smaller of the two for **thirteen of the
fifteen** pairs, not all of them. Two fall under majority labels — perplexity × SmoothLLM
(0.566 → 0.515) and token-anomaly × probe₈ (0.385 → 0.369) — and probe₁₆'s marginal ASR drops
0.68 → 0.64, so the published value overstates that one. A blanket "the published table is the
conservative estimate" overstates the claim; "for thirteen of the fifteen pairs" is accurate.

---

## What this changes in the manuscript — already applied

`paper/robustness_insert.tex` is **new** and carries all of the above. The other inserts were
edited; **do not re-apply these changes.**

1. **The composition result is reinforced.** It survives permutation inference, majority-of-five
   grader labels, and external thresholds. Minimum φ across the entire grid is **0.300**.
2. **The mechanism-specific reading of H1 is retired** as unadjudicable — it depends on both
   the grader draw and the calibration corpus.
3. **The row-vs-greedy agreement is marked labelling-dependent** rather than a finding.
4. **The one-layer-dominates claim now rests on the right comparison** (Llama Guard vs stack,
   p = 1.0).
5. **The judge-reproducibility paragraph** quotes the five-repetition measurement.

**Figures.** `fig3a` and `fig4` are unchanged. **`fig3b` was regenerated** — it previously
drew only the behaviour bootstrap, which is now known to cover ~60% of the uncertainty, so the
grader interval is drawn behind it. **`fig5` is new**: φ under all three regimes.

Re-run `python scripts/17_audit_inserts.py` after any edit — 56 assertions tying the prose to
the CSVs, currently passing.
