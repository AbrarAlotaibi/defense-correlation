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
range 0.272 to 1.000 (perplexity × probe L16 refuse on identical prompts, φ = 1.000). So the
refusal side carries the same positive correlation the breach side does — which is why the
empirical union, 0.520, falls well below the 0.766 independence prediction.

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
| `run_all7.log`, `run_no_llamaguard.log` | full console output of both runs |
