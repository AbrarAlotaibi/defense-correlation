# Supplementary analyses and reproducibility details

Every number below is generated from artifacts tracked in this repository. Regenerate all of
Sections A, B and C with:

```bash
python scripts/11_manuscript_numbers.py --fusion fusion --run results/hpc_vicuna_autodan --out fusion/manuscript
```

**Conventions.** Primary configuration throughout
(Vicuna-7B-v1.5, fluent AutoDAN-style adversary, n = 100 behaviours, 100 matched benign
prompts, StrongREJECT gold at the standard threshold). Odds ratios use Haldane–Anscombe +0.5
applied **uniformly** — verified to reproduce the `raw_or` column of `confound_check.json` to
4.8 × 10⁻⁵ across all 15 pairs, so the existing table and these numbers are on one convention.
Multiplicity is Benjamini–Hochberg over the 15 non-degenerate pairs. Intervals are
10,000-resample bootstrap percentile over behaviours. Three decimals.

---

## A. Difficulty-stratified association

### A1. perplexity × token-anomaly, difficulty-stratified

| Quantity | Value |
| --- | --- |
| crude OR (Haldane +0.5) | **6.014** |
| CMH OR | **1.393** |
| p (CMH χ²) | **0.907** |
| q (BH over 15) | **0.943** |

Strata: 5, minimum stratum size 8. Stratifier is the count of the other live defenses breached
on that behaviour, so it is independent of the pair under test — the same one used for the
other six rows. Source: `fusion/manuscript/A1_B4a_cmh_15pairs.csv`.

> **This changes what H1 can claim.**
> The pre-registered same-row pair does **not** survive stratification: its crude OR of 6.014
> collapses to 1.393 within strata, p = 0.907. Its φ = 0.35 is therefore consistent with a pure
> difficulty effect. The **only** pair that survives is probe₁₆ × probe₈ — which is the
> *post-hoc, exploratory* pair added by amendment, and whose two instances share an input
> representation.
>
> So the sentence in §"Is the correlation just behaviour difficulty?" — *"Same-row defenses
> share a blind spot that is specific to their mechanism and persists after difficulty is held
> fixed"* — would be stated of same-row defenses in general while being supported by exactly
> one pair, and not the confirmatory one. What the data licenses is narrower: of the two
> same-row pairs, the exploratory probe pair survives stratification and the pre-registered
> token-surface pair does not, so H1 is supported on the exploratory pair alone.

---

## B. Combination rules, selection, and the false-refusal floor

### B1. k-of-n curve, complete

| Rule | residual ASR | benign FRR |
| --- | --- | --- |
| k = 1 (veto) | 0.010 | 0.520 |
| k = 2 | 0.190 | 0.300 |
| k = 3 | 0.430 | 0.110 |
| k = 4 | 0.590 | 0.080 |
| **k = 5** | **0.680** | **0.080** |
| **k = 6** | **0.750** | **0.080** |
| k = 7 | 0.790 | 0.080 |

FRR is flat at 0.080 from k = 4 onward: no benign prompt is refused by four or more members
simultaneously, so the floor is the single most-refusing configuration's contribution.
`fusion/manuscript/B1_k_of_n_full.csv`; the six-defense version is alongside it.

### B2. Selection at sizes 5 and 6, and the row-based ceiling

**Row-based genuinely cannot exceed size 4 — confirmed.** The six non-Llama-Guard defenses
span exactly four dependency rows (token surface, perturbation stability, internal
representations, first-token distribution), and rule 2 admits one member per row. The ceiling
is structural, not a data artefact.

Greedy continues to 6:

| Size | Greedy members | ASR | FRR |
| --- | --- | --- | --- |
| 5 | + probe₁₆ | 0.190 | 0.450 |
| 6 | + perplexity | 0.190 | 0.450 |

> **Bound the agreement claim at size 4.** The two procedures select identical members in
> identical order for sizes 1–4, verified programmatically. Beyond that there is nothing to
> compare, because row-based has no size 5. The current wording *"agree at every stack size"*
> should read *"agree at every stack size the row rule admits (1 through 4)"*.
>
> Worth noting for honesty: greedy's sizes 5 and 6 reach ASR 0.190 against row-based's 0.200
> at its size-4 maximum. The row rule is not strictly optimal — it is capped by its own
> taxonomy — and buys nothing after size 4 either (0.190 twice, FRR flat at 0.450).

### B3. Undefended base-model refusal rate — **MEASURED: 0.080**

Run on the cluster (job 2316052). The bare model refuses **8 of 100** benign prompts, so the
text's "roughly 7 to 8%" is confirmed at the top of that range. Artifact:
`results/hpc_vicuna_autodan/stage04_undefended_benign.jsonl`, refusal decided by the same
`RefusalScorer` call every other FRR in the paper uses.

**The attributable burden is the number worth reporting, and for two defenses it is zero:**

| Defense | raw FRR | attributable | shared with floor |
| --- | --- | --- | --- |
| refusal-prime | 0.390 | 0.337 | 0.080 |
| Llama Guard | 0.260 | 0.196 | 0.080 |
| SmoothLLM | 0.260 | 0.196 | 0.080 |
| probe₈ | 0.090 | 0.011 | 0.080 |
| token-anomaly | 0.090 | 0.011 | 0.080 |
| **perplexity** | 0.080 | **0.000** | 0.080 |
| **probe₁₆** | 0.080 | **0.000** | 0.080 |
| stack | 0.810 | 0.793 | — |

Attributable = refused by the deployed configuration on a prompt the bare model did *not*
refuse, over the 92 non-floor prompts. Every one of the seven refuses a **superset of exactly
the same 8 floor prompts**. Perplexity and probe₁₆ refuse on precisely the floor and nothing
else: their entire reported false-refusal cost is the base model.

> **This overturns an earlier reading of the benign side.** An earlier version of this
> document reported that benign refusals carry the same positive dependence as breaches
> (mean φ 0.573, 21/21 positive). The measurement was right; the reading was not. Removing the 8 floor prompts collapses it — 11 of 21 pairs go
> degenerate, and across the remaining 10 the mean φ falls to **0.056** with 4/10 positive.
> The φ = 1.000 for perplexity × probe₁₆ is not a small-count artefact, it is the floor:
> identical vectors. `fusion/README.md` §5 and `paper/fusion_insert.tex` are corrected.
> The benign side shows a shared common cause, not a shared mechanism — the same shape the
> CMH stratification gives on the harmful side.

### B4. Supplementary tables

| Item | File |
| --- | --- |
| all 21 CMH rows, Llama Guard included | `fusion/manuscript/B4a_cmh_all21pairs.csv` |
| all 21 diversity rows with intervals | `fusion/manuscript/B4b_B5_diversity_all21_with_ci.csv` |
| 21-pair benign-refusal φ, long form | `fusion/manuscript/B4c_benign_refusal_phi_21pairs.csv` |
| the same as a 7×7 matrix | `fusion/manuscript/B4c_benign_refusal_phi_matrix.csv` |

The Llama Guard degeneracy is now on the record rather than asserted: all six of its CMH rows
return an infinite odds ratio and one returns an undefined p, and all six of its Q values sit
at exactly 1.000.

Benign-refusal φ across 21 pairs: mean **0.573**, range **0.272 to 1.000**, positive in
**21/21** — but read this with B3 above: conditioning on the base model's 8-prompt floor
collapses it to mean 0.056 over the 10 pairs that remain measurable. The supplementary matrix
should carry that caveat in its caption.

### B5. Intervals for Q, disagreement and double-fault

All three now carry 10,000-resample bootstrap intervals on the same footing as φ, in
`B4b_B5_diversity_all21_with_ci.csv` (`Q_lo/Q_hi`, `dis_lo/dis_hi`, `DF_lo/DF_hi`).

---

## C. Reproducibility block

Full machine-readable version: `fusion/manuscript/C_reproducibility.json`. Everything is
copied from `resolved_config.yaml` and `calibration.json`, both now tracked in the repo.

### C1. Attack configuration

| | fluent (AutoDAN-style) | GCG |
| --- | --- | --- |
| generations / steps | 30 generations | 60 steps |
| population | 24 | — |
| elite fraction | 0.25 | — |
| mutation rate | 0.15 | — |
| suffix length | — | 20 tokens |
| top-k | — | 128 |
| candidate batch | — | 64 |
| restarts | none | none |
| stopping | fixed budget | loss < 0.4 or steps exhausted |
| evaluations / behaviour | 30 × 24 = 720 | ≤ 60 × 64 |

**In-loop objective: confirmed** — target logprob of the affirmative prefix, first 12 tokens
(`target_prefix_tokens: 12`). The gold judge is never reachable from the attack loop, enforced
statically by `tests/test_no_judge_in_loop.py`.

**Target decoding:** greedy — `do_sample=False`, `temperature=None`, `top_p=None` (verified in
`dcorr/target.py`), `max_new_tokens=256`, float16, no system prompt.

**Seed: 20260727, a single seed** — which is exactly why D4 matters.

> One caution: `resolved_config.yaml` carries *both* parameter families in every run because
> they share one schema. Only the block matching `attack.adaptive.method` is live. Do not
> report the GCG parameters for the fluent run — they are inert there.

### C2. Compute

**NVIDIA H100**, one per job (`gres/gpu=1`; nodes report `gpu:h100:2`).
**Total: 59.985 GPU-hours** across all 13 `dcorr` Slurm jobs, itemised in the JSON.

> Two caveats on that total. **50% of it (30.2 h) was spent on jobs that ended TIMEOUT,
> FAILED or CANCELLED** — compute spent, not compute behind reported results. And job names do
> not map cleanly onto the five runs, so no per-run split is given; it would be a guess. Per-run
> GPU-hours could be re-derived from job stdout timestamps, but only the totals above are exact.
>
> A defensible wall-clock-per-behaviour is not derivable from this data, for the same reason.

### C3. Defense operating points

| Defense | Row | Threshold | Realised benign block rate |
| --- | --- | --- | --- |
| perplexity (window 16) | token surface | 4.849077 | 0.01 |
| token-anomaly | token surface | 0.272857 | 0.01 |
| probe₁₆ | internal repr. | 28.924215 | 0.01 |
| probe₈ (layer 8) | internal repr. | 110.767708 | 0.01 |
| Llama Guard 3 8B | semantic | n/a (categorical) | — |
| refusal-prime (prefix) | first-token | n/a | — |
| SmoothLLM (q=4, 10% swap) | perturbation | n/a (majority vote) | — |

Model string: `meta-llama/Llama-Guard-3-8B`, fp16, resident on GPU, input **and** output
checked. Probe pool: 520 train / 130 val, dedup at Jaccard 0.6, val AUROC 1.0.

> **`realized_fpr: 0.01` on n = 100 is one benign prompt.** That is the M4 identifiability
> problem stated numerically, and it is better quoted that way than as a percentage.
>
> Also: `promptguard` appears in `stack.order` but is `enabled: false` in every reported run.
> It is not one of the seven and should not appear in any table.

### C4. Judge parse failures

| Run | rows | judged | unparsed | % of rows | % of judged |
| --- | --- | --- | --- | --- | --- |
| primary (`hpc_vicuna_autodan`) | 2700 | 1887 | 66 | **2.4%** | 3.50% |
| pilot (`primary_llama32_3b`) | 2100 | 1379 | 1 | 0.05% | 0.07% |

> **Correction to an earlier version of this document.** An earlier version reported that
> 2700 was not a count in that run, and that the manuscript's 2.4% should be replaced. That was
> wrong. It came from a stale local copy of `gold.jsonl` containing 2400 rows, pulled before the
> 300 assembled-stack rows were judged. The cluster's file has 2700 rows and the manuscript's
> **2.4% (66 of 2700) is correct as written**.
>
> The stale copy was a strict subset of the complete one with **zero** differing verdicts, so
> no other analysis in this document is affected — every number elsewhere uses per-defense or
> undefended rows, which are identical in both files. `results_insert.tex` has been reverted.

Both denominators are defensible and the manuscript may keep either. 66/2700 = 2.4% counts
every graded row; 66/1887 = 3.5% counts only responses actually sent to the grader, since
blocked inputs are never judged and so cannot fail to parse. The manuscript's stated judged
count of 1885 is 1887 by direct count — a two-row difference that matters only where the
number is quoted precisely.

The Llama-2 replication's `gold.jsonl` is in the repo, but its judged rows carry no
`parse_ok` failures to report separately. The pilot is given in its place, which is not the
same quantity.

---

## D. Analyses requiring additional runs

### D1. Input-level φ, tier 0

Summary lines, verbatim from `fusion/run_r3.log`:

```
phi_static:plain:    n=15, mean shift vs phi_behaviour +0.144, same sign 15/15, positive 15/15
phi_static:transfer: n=10, mean shift vs phi_behaviour +0.101, same sign 10/10, positive 10/10
```

**The self-check passes exactly**: restricted to the diagonal it reproduces all 21 published φ
values to 0.00e+00. Full table in `fusion/r3_estimand_comparison.csv`; write-up in
`paper/estimand_insert.tex`.

Two bugs in the script were fixed to get here — `main()` never called `selfcheck()`,
and `analyse()` computed the shared-column intervals then discarded them. Both are documented
in `fusion/README.md` §7.

### D3. External benign calibration — the sharpest result in this batch

Job 2316054. Corpus choice was forced, not free:

- `data/eval_benign.jsonl` (100, jbb_benign) is the in-sample set — the problem itself.
- **XSTest is already in this repo** as the benign half of `data/probe_train.jsonl` (250 rows).
  Calibrating the probe on it would be leakage: those are its training examples.
- **AlpacaEval** (805 general instructions) is seen by neither. It is also 8× the evaluation
  set, so the 1% quantile rests on **8 prompts rather than 1** — which is the identifiability
  half of M4.

Deduplicated against both the eval benign set and the probe training pool at the same
Jaccard 0.6 the probe audit uses: **805 kept, 0 dropped**, max Jaccard 0.5
(`data/external_benign_report.json`). Thresholds are written to `calibration_external.json`,
deliberately *not* `calibration.json`, so nothing downstream silently picks them up and the
reported runs stay reproducible.

| Filter | in-sample thr | external thr | held-out block rate (in-sample thr → external thr) |
| --- | --- | --- | --- |
| perplexity | 4.849 | 6.516 | 0.010 → **0.000** |
| token-anomaly | 0.273 | 0.523 | 0.010 → **0.000** |
| **probe₁₆** | 28.924 | **−5.210** | 0.010 → **0.260** |
| probe₈ | 110.768 | 100.593 | 0.010 → **0.070** |

**The two surface filters barely move, and move in the permissive direction.** AlpacaEval
prompts are slightly higher-perplexity than JBB benign (p50 3.49 vs 3.04), so an external
threshold is looser and blocks nothing on the held-out set. Those filters are largely
distribution-insensitive.

**The probes move enormously, and in the punitive direction.** Median probe₁₆ score is −70.3
on AlpacaEval against −31.2 on JBB benign; for probe₈ it is −33.8 against +44.6. JBB benign
prompts sit far closer to the harmful side of the probe's decision boundary than ordinary
instructions do. Calibrated on generic traffic, probe₁₆ would refuse **26% of the held-out
benign set** rather than 1%.

> **This empirically confirms the suspicion the Limitations section already raises** —
> that the probe's near-ceiling validation AUROC "may reflect topical separability between the
> harmful and benign corpora as much as a representation of harmfulness". JBB benign prompts
> are matched counterparts to the harmful behaviours and are therefore topically adjacent to
> them; AlpacaEval prompts are not. The probe is substantially tracking that adjacency.
>
> The consequence for the paper is specific and worth stating plainly: **the probe's 1%
> operating point is an artefact of calibrating on JBB benign.** It is not a property of the
> defense that transfers to deployment traffic. The surface filters do not have this problem.
>
> This does not invalidate any reported number — every result is correctly labelled as using
> in-sample thresholds — but it converts "thresholds are in-sample, so the block rates are
> optimistic by an unknown margin" into a measured margin, and shows the margin is
> concentrated almost entirely in one row of Table 6.

**What this does not give.** New thresholds change which prompts are blocked, which
changes breach outcomes, so the marginal ASRs and every φ would have to be re-derived from a
re-run. That is D4-scale compute and was not done. The numbers above are the calibration and
false-refusal side only.


### D2. Intersection versus direct, with resolution — the assumption fails

Job 2316052, `COMPLETED` in 3:41:20, 100/100 behaviours, no fatal errors. Judged post hoc on
the login node (600 verdicts). Artifact: `fusion/manuscript/D2_intersection_vs_direct.json`.

| | |
| --- | --- |
| predicted intersection residual (six defenses) | **0.190** |
| measured direct residual on the assembled stack | **0.000** |
| difference | **−0.190** |
| per-behaviour agreement | 81/100 |
| predicted-breach-only / measured-breach-only | 19 / 0 |

**The intersection assumption does not hold, and the earlier "validation" was an artefact of
the floor.** In the primary run the check compared a predicted 0.010 against a measured 0.000
and agreed — but both sat on the floor, where any method returning near-zero would pass.
Removing Llama Guard gives the check resolution, and it fails: the intersection over
per-defense breach vectors predicts 19 breaches where a direct attack produces none.

**The mechanism is joint infeasibility, not the stack catching successful attacks.** Every one
of the 100 adaptive attempts was blocked at the *input* stage, and the attack reports
`found_feasible: False` for **100/100** behaviours. The search never located a prompt
satisfying all six input constraints at once. Optimising against each layer separately and
intersecting the results implicitly assumes a prompt that defeats layer A also survives layer
B when both are present; the joint feasible region turns out to be far smaller than that.

> **What this costs the paper.** The k > 1 rows of the combination-rules table are built from
> the same intersection assumption that has now been contradicted at k = 1 by 0.19. They
> should be labelled not merely as estimates but as estimates the one direct check with
> resolution *disagrees with*, in the direction of overstating attacker success. The honest
> version of the earlier caveat is stronger than the one it replaces.

**The caveat that keeps this from being over-claimed.** `found_feasible: False` is a statement
about the search, not a proof that the feasible region is empty. The budget here is the same
one that succeeds easily per-layer (30 generations × 24 population, `max_filter_rejections`
40), so the comparison is fair, but a stronger or differently-shaped attacker could still find
joint solutions. The defensible claim is: *under an attack budget sufficient to produce a 0.19
per-layer intersection residual, a direct attack on the assembled six-layer stack finds no
feasible prompt on any of 100 behaviours.* Not: *the stack is unbreakable.*

The undefended control in the same run behaves as expected (64/100 adaptive breaches), so the
attack works when unconstrained — the zero is about the constraints, not a broken pipeline.

### Incidental: the gold judge is not deterministic — **6.3% verdict disagreement**

The D2 run was seeded with a copy of the primary run's `stage04_undefended.jsonl`, so stage 06
judged the **same 300 undefended responses a second time** — same rubric, same model, same
temperature 0, same threshold. That accidental replicate measures something no planned part of
the study does.

| Attack | ASR run 1 | ASR run 2 | verdict disagreements |
| --- | --- | --- | --- |
| adaptive | 0.58 | **0.64** | 10 (10%) |
| static_gcg | 0.27 | 0.26 | 5 (5%) |
| static_plain | 0.06 | 0.10 | 4 (4%) |
| **overall** | | | **19/300 = 6.3%** |

Only **1** of the 19 involves a parse failure, so this is not the unparsed-output problem — it
is the grader returning different rubric scores for identical text.

> **This is a floor on the precision of every ASR in the paper, and no reported interval
> contains it.** The bootstrap resamples behaviours; it says nothing about the grader. The
> headline undefended adaptive ASR moves 0.58 → 0.64 on a pure re-judge. The consequence: the
> re-judge disagreement rate belongs in Limitations alongside the existing 2.4%-unparsed note,
> and no ASR difference smaller than a few points should be presented as meaningful.
> Correlations should be more robust than levels — φ is relational and both members shift
> together — but that is an argument, not a measurement, and it was not tested here.
>
> Artifact: `fusion/manuscript/judge_determinism.json`, regenerate with
> `scripts/16_judge_determinism.py`.


### D4–D5 — not run

Both need GPU jobs, and neither is derivable from stored artifacts. Neither was run.

| Item | Cost | Status |
| --- | --- | --- |
| **D4** seeds ×2 | ~2× the primary run | **not run** |
| **D5** tier-2 cross-evaluation | ~4,200 generations | **not needed** — D1's signs were unambiguous |

**D5 is not needed.** D1's signs are unambiguous (15/15 and 10/10), which was the stated
decision rule.

---

## E. Bibliography and repository

- **Five bibliography entries** — `paper/references_added.bib`. Metadata pulled from
  the arXiv API rather than transcribed, so titles, full author lists and years are as arXiv
  reports them:

  | Key | arXiv | Authors |
  | --- | --- | --- |
  | `dualbreach2025` | 2504.18564 | 8 |
  | `attackermovessecond2025` | 2510.09023 | 14 |
  | `operationalizingthreat2024` | 2407.14937 | 10 |
  | `aegis2024` | 2404.05993 | 4 |
  | `r2guard2024` | 2407.05557 | 2 |

  Two notes. AEGIS resolved to *Online Adaptive AI Content Safety Moderation with Ensemble of
  LLM Experts* (2404.05993); the acronym is reused elsewhere, so the entry should be checked
  against the intended reference.
  R²-Guard is entered as an ICLR 2025 `@inproceedings` with the arXiv id in `note`; **verify
  the page numbers against the proceedings before camera-ready**, as the API gives only the
  preprint. Two titles had an acronym de-title-cased where the arXiv feed had mangled it
  (`Llm` → `LLM`); the change is flagged inline in the .bib.
- **Repository:** public, and contains the run artifacts, configs, code and the paper
  inserts. The StrongREJECT prompt is at `dcorr/judge/strongreject_prompt.txt`, SHA-256 pinned
  and verified at load.

---

## How these analyses are reflected in the manuscript inserts

All three inserts are self-consistent with the artifacts; `scripts/17_audit_inserts.py`
re-checks every asserted number against the CSV that produced it (38 assertions, currently
passing). Re-run it after any edit.

| Insert | Status |
| --- | --- |
| `paper/results_insert.tex` | updated — 8 edits |
| `paper/fusion_insert.tex` | updated — 4 edits |
| `paper/estimand_insert.tex` | current as written |
| `paper/references_added.bib` | 5 entries, from the arXiv API |

### Substantive claims the analyses above changed

1. **H1 is narrowed.** The pre-registered same-row pair does not survive stratification (CMH
   OR 1.39, p 0.91); only the exploratory probe pair does. The text now says the
   mechanism-specific reading rests on a single post-hoc pair and is suggestive, not
   confirmed. The confound table gained the missing perplexity × token-anomaly row.
2. **The intersection estimate is no longer called a faithful proxy.** With Llama Guard
   removed, intersection predicts 0.190 and a direct attack yields 0.000. The text now bounds
   the claim by the attack budget rather than asserting the stack is unbreakable.
3. **The judging denominator was re-verified and left as the manuscript had it** — 66 of
   2700 (2.4%) is correct; an earlier claim in this document that it was wrong came from a
   stale local file and has been retracted. The insert now also gives 3.5% as the
   judged-only rate.
4. **A judge-reproducibility paragraph is added** — 6.3% verdict disagreement on re-judging
   identical responses, and an explicit warning that no reported interval covers grader
   variance.
5. **H3 reports the measured floor** (8/100) and the attributable burden, including 0.000 for
   perplexity and probe₁₆.
6. **The in-sample threshold caveat is now a measured margin** — probe₁₆ would refuse 26% of
   the held-out benign set at an external threshold.
7. **The k-of-n table is complete** (k=5, k=6) and the flat false-refusal tail is explained.
8. **"Agree at every stack size" is bounded** to the four sizes the row rule admits, with the
   structural reason given.

### Analyses deliberately not performed

- **D4 (seed replicates)** — not run. The single-seed limitation is stated in Limitations.
- **D5 (tier-2 cross-evaluation)** — unnecessary: D1's sign agreement was unambiguous (15/15
  and 10/10), which was the stated decision rule.
- **Re-deriving ASRs under the external thresholds** (D3) — that needs a full re-run, so the
  external-calibration result covers the calibration and false-refusal side only.
