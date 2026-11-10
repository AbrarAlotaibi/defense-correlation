# Classifier-fusion analysis

Recasts the paper's failure-correlation result in the vocabulary of classifier fusion
(Kuncheva and Whitaker, 2003), adds a k-of-n combination-rule operating curve, and compares
the paper's row-based composition rule against a diversity-greedy selection.

Produced by [`scripts/fusion_analysis.py`](../scripts/fusion_analysis.py) from inputs built
by [`scripts/09_build_fusion_inputs.py`](../scripts/09_build_fusion_inputs.py).

```bash
python scripts/09_build_fusion_inputs.py --run results/hpc_vicuna_autodan \
    --config configs/hpc_vicuna_autodan.yaml --out fusion/ --verify-frr
cd fusion
python ../scripts/fusion_analysis.py --breach breach.csv --refusal refusal.csv \
    --rows rows.csv --out fusion_results.tex
python ../scripts/fusion_analysis.py --breach breach.csv --refusal refusal.csv \
    --rows rows.csv --exclude llamaguard --out fusion_results_no_llamaguard.tex
```

Source run: `results/hpc_vicuna_autodan` — Vicuna-7B-v1.5, fluent (AutoDAN-style) adversary,
n = 100 JailbreakBench behaviours, 100 matched benign prompts, StrongREJECT gold judging,
`attack == adaptive`. `undefended` is dropped: it is the control, not a stack member.

**Every number here is measured.** Both inputs are pivots of stored artifacts, so nothing is
transcribed by hand, and both are checked against results already in the repository:

- `breach.csv` comes from `gold.jsonl` and reproduces the published Table 6 exactly —
  double-fault equals the stored joint breach rate and `DF_indep` the stored independence
  prediction to floating-point epsilon (max deviation 5.6 × 10⁻¹⁷ across all 21 pairs), with
  φ agreeing at the four decimals `table6.csv` stores.
- `refusal.csv` comes from `stage04_<defense>_benign.jsonl` via the same `RefusalScorer` call
  `07_analyze.py` uses for H3. `--verify-frr` asserts the column means equal
  `analysis.json`'s `h3_refusals`; all seven match exactly.

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

### 1. Every pair is positively associated on every measure

Q > 0 and double-fault above the independence prediction for all 15 pairs. That is the same
verdict Table 6 reaches through φ, reached independently in the fusion vocabulary.

### 2. Llama Guard breaks the Q-statistic, not the finding

With a marginal of 0.01 the `n01 · n10` term collapses and Q saturates at exactly 1.000 for
all six of its pairs. That degeneracy alone drives Spearman Q vs φ from 0.99 down to −0.26.
It is an artefact of Q's definition at extreme marginals, and is the reason to report the
15-pair variant. Disagreement and double-fault are unaffected.

### 3. The row heuristic recovers the diversity-greedy selection

Excluding Llama Guard, composition rule 2 and a greedy search that explicitly minimises
double-fault choose the same members in the same order, at identical ASR *and* identical FRR
at every stack size:

| Size | Members | ASR | FRR |
| --- | --- | --- | --- |
| 1 | token anomaly | 0.35 | 0.09 |
| 2 | + SmoothLLM | 0.27 | 0.27 |
| 3 | + probe L8 | 0.23 | 0.28 |
| 4 | + refusal prime | 0.20 | 0.45 |

The row heuristic costs nothing against a search that optimises the diversity criterion
directly. With Llama Guard included the comparison is floor-limited exactly as the script's
`--exclude` help predicts: both rules pin to ASR 0.010 at every size.

### 4. The operating curve is steep, and k = 1 is the sensible corner

| Rule | residual ASR | benign FRR |
| --- | --- | --- |
| **k = 1 of 7 (veto, as reported)** | **0.010** | **0.520** |
| k = 2 of 7 | 0.190 | 0.300 |
| k = 3 of 7 | 0.430 | 0.110 |
| k = 4 of 7 | 0.590 | 0.080 |
| k = 7 of 7 | 0.790 | 0.080 |
| strongest single layer (Llama Guard) | 0.010 | 0.260 |

Relaxing from k = 1 to k = 2 buys 22 points of benign FRR for 18 points of ASR — a poor
trade at this operating point. Note the last row: Llama Guard alone matches the whole stack's
0.010 ASR at half its false-refusal cost, which is the intersection-vs-direct point of the
Results section restated as an operating decision.

### 5. Benign refusals are correlated too, and the stack refuses more than any rule predicts

Three quantities on the same 7 defenses and the same 100 benign prompts:

| | FRR |
| --- | --- |
| empirical union — any member refuses (k = 1) | **0.520** |
| independence prediction 1 − ∏(1 − fᵢ) | 0.766 |
| measured deployed stack | **0.810** |

Pairwise φ on the benign refusal vectors is positive for **21 / 21** pairs, mean 0.573,
range 0.272 to 1.000, which is why the empirical union, 0.520, falls well below the 0.766
independence prediction.

**That correlation is almost entirely the base model's own refusal floor, not shared defense
behaviour.** B3 measured the undefended model on the same benign set: it refuses **8 of 100**
prompts with no defense attached, and every one of the seven defended configurations refuses a
**superset** of exactly those 8. Perplexity and probe L16 refuse on precisely the 8 floor
prompts and nothing else, which is why their φ is exactly 1.000 — identical vectors, not a
shared blind spot. Removing the 8 floor prompts and recomputing on the remaining 92 collapses
the effect: 11 of the 21 pairs become degenerate because a member adds nothing beyond the
floor, and across the 10 that remain measurable the mean φ falls from 0.573 to **0.056**, with
only **4 of 10** still positive (range −0.075 to 0.460).

So the honest statement is the reverse of the obvious reading: benign refusals look strongly
correlated, but conditioning on the base model's floor removes nearly all of it. This mirrors
what stratification does to the cross-row breach correlations — a shared common cause, not a
shared mechanism.

The measured stack nevertheless refuses *more* than either, 0.810. A union over
independently-measured decisions cannot explain that: in the assembled stack the members run
in sequence and change the response the next member sees, so refusals compound rather than
merely intersect. This is worth a line in the paper — the independence assumption fails on
the benign side as well as the harmful side, and it fails in **both** directions depending on
whether you are predicting the union or the deployed pipeline.

### 6. Difficulty-stratified CMH, all pairs

The script also runs a Cochran-Mantel-Haenszel test per pair, stratifying behaviours by a
difficulty score built from the *other* defenses only, so the stratifier is independent of the
pair under test. This reproduces `results/hpc_vicuna_autodan/confound_check.json` **exactly**
(max deviation 0.0 on the common odds ratio and 3.1e-15 on p across all 15 pairs) and adds
the Benjamini-Hochberg q column, which the manuscript's claim rested on but no artifact
stored.

| Pair | CMH OR | p | q (BH) |
| --- | --- | --- | --- |
| **probe L16 x probe L8** | **158.72** | <0.001 | **<0.001** |
| refusal prime x SmoothLLM | 4.34 | 0.023 | 0.173 |
| perplexity x refusal prime | 3.95 | 0.063 | 0.314 |
| perplexity x probe L8 | 3.07 | 0.209 | 0.562 |
| probe L16 x refusal prime | 0.72 | 0.943 | 0.943 |
| probe L8 x SmoothLLM | 0.91 | 0.835 | 0.943 |

**Two pairs clear the nominal level; only one survives correction** — the same-row probe
pair, whose within-stratum odds ratio of 159 is an order of magnitude above its crude value.
That is exactly what §"Is the correlation just behaviour difficulty?" already claims.

Two cautions. The crude OR here is **uncorrected**, while `confound_check.json` and the paper
apply a Haldane-Anscombe +0.5 correction: 17.400 against 16.059 for perplexity x probe L16.
Do not mix the two conventions in one table. And the `survives` field in
`confound_check.json` is raw-p based, so it flags two pairs; the paper's text correctly
applies multiplicity correction and reports one.

Including Llama Guard makes its six pairs degenerate here too — the odds ratio is infinite
for all six and undefined for one — a third reason to report the 15-pair variant.

### 7. Input-level vs behaviour-level phi (the estimand question)

Eq. (9) is written over *inputs*; the measurement is indexed by *behaviour*, because the
adaptive attack is re-optimised against each defense separately. `scripts/input_level_phi.py`
separates the two, from a matrix built by `scripts/10_build_cross_breach.py`:

```bash
python scripts/10_build_cross_breach.py --run results/hpc_vicuna_autodan --out fusion/cross_breach.csv
cd fusion && python ../scripts/input_level_phi.py --cross cross_breach.csv     --published ../results/hpc_vicuna_autodan/table6.csv --out r3_results.tex
```

Only **tier 0** is derivable from stored artifacts, and it is free. `dcorr/attacks/static.py`
builds both static baselines from the behaviour and one run-wide suffix and never consults the
defense, so those prompt sets are byte-identical across all seven defenses and give a genuine
shared-input phi. The off-diagonal of the cross matrix does not exist — prompts optimised
against d1 were never scored on d2 — so tiers 1 and 2 report as `--` rather than being guessed.

| | plain set | transfer set |
| --- | --- | --- |
| measurable pairs | 15 | 10 |
| same sign as phi_behaviour | **15 / 15** | **10 / 10** |
| positive | **15 / 15** | **10 / 10** |
| mean shift vs phi_behaviour | +0.144 | +0.101 |
| intervals spanning zero | 5 | **0** |

**The sign never flips**, so the composition conclusion holds under both estimands and the
objection is a clarification rather than a threat. The behaviour-level figure the paper
reports is the *conservative* one: holding the prompt fixed removes the adversary's freedom to
build a different input per layer, which is the only thing that could decorrelate them.

Power is the limit. Static marginals run 0.03 to 0.28 against 0.35 to 0.68 for the fluent
adversary, so several plain-set pairs rest on six to eight breach events; the 1.00 for
perplexity x token-anomaly means only that seven behaviours breached both filters. The
transfer set is better powered and all ten of its intervals exclude zero.

**Two bugs fixed in the supplied script** (`scripts/input_level_phi.py`):

- `main()` never called `selfcheck()`, although the docstring says the check is "reported
  first and loudly". It is now wired to `--published`, aborts with exit 2 on mismatch, and
  `--tolerate-selfcheck` downgrades that to a warning. Both paths were tested against a
  deliberately corrupted table. On the real data it passes: all 21 published phi values
  reproduced to 0.00e+00.
- `analyse()` computed bootstrap CIs for the shared-input columns and then discarded them.
  They are now kept as `phi_<source>_lo/_hi`, which matters precisely because these are the
  low-marginal columns.

## Caveat this analysis inherits

Breach vectors come from re-optimising the attack against each defense **separately**. Every
combination rule evaluated here therefore rests on the same intersection assumption used for
the pairwise analysis, which was validated against a direct attack on the assembled stack at
**k = 1 only** (intersection predicts 0.010, direct attack yields 0.000, agreement 99/100).
Every k > 1 row is an estimate and must be labelled as such until a direct attack is run
against that configuration. Point 5 above shows the assumption is measurably imperfect on the
benign side, which is a reason to state this caveat plainly rather than to bury it.

## Files

| File | Contents |
| --- | --- |
| `breach.csv` | 100 behaviours × 7 defenses, 1 = breached |
| `refusal.csv` | 100 benign prompts × 7 defenses, 1 = refused |
| `rows.csv` | defense → dependency row, as recorded by the run |
| `fusion_diversity.csv`, `fusion_combination_rules.csv` | all 7 defenses |
| `fusion_diversity_no_llamaguard.csv`, `fusion_combination_rules_no_llamaguard.csv` | 15-pair variant |
| `fusion_results.tex`, `fusion_results_no_llamaguard.tex` | LaTeX tables + `\newcommand` macros |
| `fusion_cmh_all_pairs.csv`, `fusion_cmh_all_pairs_no_llamaguard.csv` | CMH per pair with BH q |
| `cross_breach.csv` | source x defense x behaviour breach matrix (tier 0) |
| `r3_estimand_comparison.csv`, `r3_results.tex` | input- vs behaviour-level phi |
| `run_all7.log`, `run_no_llamaguard.log`, `run_r3.log` | full console output of the runs |
