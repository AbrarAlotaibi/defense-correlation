# Response to the manuscript data request

Every number below is generated from artifacts tracked in this repository. Regenerate all of
Sections A, B and C with:

```bash
python scripts/11_manuscript_numbers.py --fusion fusion --run results/hpc_vicuna_autodan --out fusion/manuscript
```

**Conventions used, matching the request exactly.** Primary configuration throughout
(Vicuna-7B-v1.5, fluent AutoDAN-style adversary, n = 100 behaviours, 100 matched benign
prompts, StrongREJECT gold at the standard threshold). Odds ratios use Haldane–Anscombe +0.5
applied **uniformly** — verified to reproduce the `raw_or` column of `confound_check.json` to
4.8 × 10⁻⁵ across all 15 pairs, so the existing table and these numbers are on one convention.
Multiplicity is Benjamini–Hochberg over the 15 non-degenerate pairs. Intervals are
10,000-resample bootstrap percentile over behaviours. Three decimals.

---

## A. Blocking — delivered

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

> **This changes what H1 can claim, and the manuscript should say so.**
> The pre-registered same-row pair does **not** survive stratification: its crude OR of 6.014
> collapses to 1.393 within strata, p = 0.907. Its φ = 0.35 is therefore consistent with a pure
> difficulty effect. The **only** pair that survives is probe₁₆ × probe₈ — which is the
> *post-hoc, exploratory* pair added by amendment, and whose two instances share an input
> representation.
>
> So the sentence in §"Is the correlation just behaviour difficulty?" — *"Same-row defenses
> share a blind spot that is specific to their mechanism and persists after difficulty is held
> fixed"* — is currently stated of same-row defenses in general but is supported by exactly one
> pair, and not the confirmatory one. Recommended: state that of the two same-row pairs, the
> exploratory probe pair survives stratification and the pre-registered token-surface pair does
> not, and that H1 is therefore supported on the exploratory pair alone. That is a weaker claim
> than the current text, and it is the one the data licenses.

---

## B. Table completion — delivered, except B3

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

### B3. Undefended base-model refusal rate — **NOT AVAILABLE, needs one cheap pass**

There is no `stage04_undefended_benign.jsonl` in any run, on the cluster or locally. The
undefended control was evaluated on the harmful split only. The "roughly 7 to 8%" in the text
is not traceable to a stored artifact and I could not confirm it.

This is the cheapest outstanding item: 100 benign prompts through the undefended model, greedy
decode, no attack search — a few minutes on one H100. See "Ready to run" below.

### B4. Supplementary tables — delivered

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
**21/21**.

### B5. Intervals for Q, disagreement and double-fault — delivered

All three now carry 10,000-resample bootstrap intervals on the same footing as φ, in
`B4b_B5_diversity_all21_with_ci.csv` (`Q_lo/Q_hi`, `dis_lo/dis_hi`, `DF_lo/DF_hi`).

---

## C. Reproducibility block — delivered

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

> Two honest caveats. **50% of that total (30.2 h) was spent on jobs that ended TIMEOUT,
> FAILED or CANCELLED** — wasted compute, not reported results. And job names do not map
> cleanly onto the five runs, so I am not offering a per-run split; it would be a guess. If
> you want per-run GPU-hours I can re-derive them from job stdout timestamps, but the totals
> above are exact and the split is not.
>
> I also cannot give a defensible wall-clock-per-behaviour from this data for the same reason.

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
> problem stated numerically, and it should be quoted that way rather than as a percentage.
>
> Also: `promptguard` appears in `stack.order` but is `enabled: false` in every reported run.
> It is not one of the seven and should not appear in any table.

### C4. Judge parse failures — **the reported denominator is wrong**

| Run | rows | judged | unparsed | % of judged |
| --- | --- | --- | --- | --- |
| primary (`hpc_vicuna_autodan`) | 2400 | 1885 | 66 | **3.50%** |
| pilot (`primary_llama32_3b`) | 2100 | 1379 | 1 | 0.07% |

The numerator 66 is right. **2700 is not a count that exists in the primary run** — it has
2400 rows, of which 1885 were actually sent to the judge (blocked inputs are never judged, so
they cannot fail to parse). The defensible figure is **66 of 1885 = 3.50% of judged
responses**; 66/2400 = 2.75% if you prefer the all-rows denominator. Either way, 2.4% of 2700
should not stand.

The Llama-2 replication's `gold.jsonl` is now in the repo, but its judged rows carry no
`parse_ok` failures to report separately — I have given the pilot instead, and flag that as
not the same thing as the replication figure you asked for.

---

## D. Needs compute

### D1. Input-level φ, tier 0 — **DONE**

Already delivered in the previous round. Summary lines, verbatim from `fusion/run_r3.log`:

```
phi_static:plain:    n=15, mean shift vs phi_behaviour +0.144, same sign 15/15, positive 15/15
phi_static:transfer: n=10, mean shift vs phi_behaviour +0.101, same sign 10/10, positive 10/10
```

**The self-check passes exactly**: restricted to the diagonal it reproduces all 21 published φ
values to 0.00e+00. Full table in `fusion/r3_estimand_comparison.csv`; write-up in
`paper/estimand_insert.tex`.

Two bugs in the supplied script were fixed to get here — `main()` never called `selfcheck()`,
and `analyse()` computed the shared-column intervals then discarded them. Both are documented
in `fusion/README.md` §7.

### D2–D5 — not run

These need GPU jobs. Nothing about them is derivable from stored artifacts, and I have not
started any of them.

| Item | Cost | Status |
| --- | --- | --- |
| **B3** undefended benign refusal | ~100 generations, minutes | ready, see below |
| **D2** direct attack on the six-layer stack | one run | ready, see below |
| **D3** external benign calibration | no attack search, one scoring pass | needs a corpus choice |
| **D4** seeds ×2 | ~2× the primary run | expensive |
| **D5** tier-2 cross-evaluation | ~4,200 generations | only if D1 were ambiguous — it is not |

**D5 is not needed.** D1's signs are unambiguous (15/15 and 10/10), which was the stated
decision rule.

**Ready to run**, once you say go — these are prepared but not launched:

```bash
# B3: undefended model on the benign set. Add `undefended` to the benign sweep.
CONFIG=configs/hpc_vicuna_autodan.yaml FROM=4 bash scripts/run_pipeline.sh

# D2: direct attack on the stack with Llama Guard disabled, so the intersection
# prediction is non-trivial and the check has resolution.
```

D2 needs a config variant with `llamaguard.enabled: false` and the stack order trimmed. I can
write it, but it changes what gets run on your allocation, so I would rather you confirm the
GPU spend first.

---

## E. Not data

- **Five bibliography entries: not done.** I did not fetch them, and I will not guess author
  lists. Say the word and I will pull the BibTeX from arXiv.
- **Repository:** public and resolving — `isPrivate: false`, confirmed via `gh`. It contains
  the run artifacts, configs, code, and both paper inserts. The StrongREJECT prompt is at
  `dcorr/judge/strongreject_prompt.txt` and is SHA-256 pinned and verified at load.
- **Elsevier package** (Highlights, CRediT, AI declaration): not started; not a data item.

---

## Summary of what changes in the manuscript

Three corrections, in descending order of consequence:

1. **H1's interpretation must be narrowed** (A1). The pre-registered same-row pair does not
   survive stratification; only the exploratory probe pair does.
2. **The judge parse-failure denominator is wrong** (C4). 2700 is not a count in that run.
3. **"Agree at every stack size" must be bounded to sizes 1–4** (B2), because row-based has no
   size 5.

And one gap: **B3 has no artifact behind it**, so the "roughly 7 to 8%" base-model refusal
figure is currently unsupported.
