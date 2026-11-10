# Task A — analyses on existing data

No new runs, no API calls. Regenerate with:

```bash
python scripts/18_task_a_variance.py --primary results/hpc_vicuna_autodan \
    --nolg results/hpc_vicuna_autodan_nolg --config configs/hpc_vicuna_autodan.yaml \
    --out fusion/taskA
```

Provenance for every file is in `meta.json`: seed 20260727, grader prompt SHA-256, judge model
and temperature, target model, and the marginal ASRs the vectors reproduce.

**McNemar variant used: exact two-sided binomial on the discordant cells**
(`scipy.stats.binomtest`). Not the chi-square approximation and not the continuity-corrected
variant — several discordant cells are small enough that the approximation would misbehave.

**Multiplicity family sizes.** Holm over **21** for the full pairwise matrix (A1/A2). A3 covers
the **15** measurable pairs, excluding Llama Guard, matching the family the manuscript uses.

## The one-line answer

**11 of 21 pairs differ significantly after Holm** (14 of 21 uncorrected), and **yes, Llama
Guard dominates every other member pairwise** — there is no behaviour on which Llama Guard is
breached while another defense holds.

## A1 — pairwise McNemar (`A1_mcnemar.csv`)

The five Llama Guard pairs are the strongest results in the matrix (p from 1.4e-20 to 2.2e-16),
all in the same direction. Token-anomaly is next, significantly better than every remaining
defense. The seven pairs that do not separate after Holm are the mid-strength defenses among
themselves — perplexity, refusal-prime, SmoothLLM, probe₁₆, probe₈ — which is the expected
picture if those layers are close in strength.

Worth noting for the manuscript: **probe₁₆ × probe₈ does not separate after Holm**
(p = 0.039 raw, 0.309 corrected). Their *marginals* are statistically indistinguishable even
though their *failure correlation* is the highest in the study (φ = 0.75). Those are different
claims and the paper should not let one imply the other.

## A1 vs the assembled stacks (`A1_vs_stack.csv`)

Every defense except Llama Guard differs from both stacks at p ≤ 5.8e-11, always in the same
direction: `n_stack_only = 0` in all thirteen comparisons, so no behaviour is breached by a
stack but held by an individual layer.

**Llama Guard versus the seven-layer stack: n01 = 1, n10 = 0, p = 1.0.** They are statistically
indistinguishable. This is the quantitative form of the "one strong layer dominates the stack"
finding — previously resting on a single McNemar against the probe, which was the *weakest*
layer to compare against. The comparison that matters is this one.

## A2 — Cochran's Q (`A2_cochran.txt`)

Q = 193.407, df = 6, p = 4.8e-39. The seven defenses are not interchangeable; the omnibus
licenses the pairwise follow-up.

## A3 — permutation test for φ (`A3_permutation.csv`)

10,000 permutations of one member per pair, which holds both marginals fixed by construction.
**All 15 pairs are significant**, twelve at the resolution floor of the test
(p = 1/(10,000+1) = 9.999e-05). The permutation p-values agree with the existing bootstrap
intervals on significance for **15/15** pairs, so the correlation result does not depend on the
bootstrap's distributional assumptions.

## A4 — re-judge concordance (`A4_rejudge.csv`)

**Not computable as specified, and the file says so rather than reporting an empty result.**
The stored re-judge covers 300 responses that are all `defense = undefended` (100 behaviours ×
3 attacks). It contains no per-defense breach vector, so no pair φ can be recomputed under
re-judged labels.

What the re-judge does support is the undefended ASR: 0.58 → 0.64 adaptive, 0.27 → 0.26
static_gcg, 0.06 → 0.10 static_plain, with 19 of 300 verdicts flipped. **Recomputing pair φ
under re-judging is exactly what Task B does**, and is the reason Task B is the highest-value
item in the brief.

## Files

| File | Contents |
| --- | --- |
| `A1_mcnemar.csv` | 21 pairs: n01, n10, exact p, Holm p |
| `A1_vs_stack.csv` | each defense vs the seven- and six-layer stacks |
| `A2_cochran.txt` | Q, df, p, and the Holm follow-up counts |
| `A3_permutation.csv` | 15 pairs: φ and two-sided permutation p |
| `A4_rejudge.csv` | why this is not computable from stored data |
| `meta.json` | seed, prompt digest, models, marginals |
