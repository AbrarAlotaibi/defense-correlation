# Measuring failure correlation between stacked LLM defenses

Code and data artifact for **"Defending Large Language Models: Access Tiers, Inference Cost,
and Why Layered Defenses Compound Only Under Independence."**

Layered defenses are assumed to compound: if filter A admits 30% of attacks and filter B
admits 30%, a stack of both should admit ~9%. That multiplication is valid **only if the two
fail on different inputs**. This repository measures whether they do.

For each defense `d` and jailbreak behaviour `i` we record a binary breach `b_d(i)` under a
post-hoc gold judge, then for every pair report the marginals, the joint breach rate, the
excess over the independence prediction `Δ`, and the failure correlation `φ` with bootstrap
confidence intervals.

---

## Headline results

Primary target Vicuna-7B-v1.5, fluent (AutoDAN-style) adversary, n = 100 JailbreakBench
behaviours, StrongREJECT gold judging.

| Finding | Result |
| --- | --- |
| **H1** same-row defenses fail together | **Supported.** probe₁₆ × probe₈ φ = 0.75 [0.62, 0.88]; perplexity × token-anomaly φ = 0.35 [0.18, 0.50] |
| **H2** cross-row defenses are independent | **Rejected.** All 13 non-degenerate cross-row pairs positive, φ = 0.30–0.62, all significant after BH correction |
| **H3** refusal budget binds | **Supported.** Stack false-refusal 0.810 vs independence prediction 0.766 |
| Attack class determines what is measurable | Perplexity filter residual ASR **0.00 under a suffix attack → 0.66 under a fluent attack**. Its apparent invulnerability is a property of the attack, not the defense |
| Intersection vs direct stack attack | Intersection predicts 0.010, direct attack yields 0.000, agreement 99/100 |

The independence assumption fails: for perplexity × probe₁₆ the multiplicative model predicts
a joint residual of 0.449 while the measured value is 0.580. `Δ > 0` for every pair measured.

A difficulty-stratified re-analysis (`paper/results_insert.tex`, §"Is the correlation just
behaviour difficulty?") separates mechanism-specific correlation from a shared difficulty
gradient; only the same-row probe pair survives that test, and the paper says so.

---

## Where each reported number comes from

Every figure in the paper traces to an artifact in `results/`. Nothing is re-derived by hand.

| Paper element | Artifact |
| --- | --- |
| Table 6 (15 pairs, φ, CI, Δ, q) | `results/hpc_vicuna_autodan/table6.csv` (and `.tex`) |
| Marginal ASRs, H1/H2/H3 verdicts, McNemar, intersection-vs-direct | `results/hpc_vicuna_autodan/analysis.json`, summarised in `REPORT.md` |
| Filter thresholds and realised block rates | `results/<run>/calibration.json` |
| Undefended positive control (gate) | `results/<run>/positive_control.json` — **proxy** heuristic, not gold; do not quote as ASR |
| Per-response gold verdicts | `results/<run>/gold.jsonl` |
| Attack-class contrast (suffix vs fluent) | `results/hpc_vicuna/` vs `results/hpc_vicuna_autodan/` |
| Llama-2 replication (n = 50) | `results/hpc_llama2_autodan/` |
| 3B pilot referenced for cross-target consistency | `results/primary_llama32_3b/` |
| Figures | `paper/figures/*.pdf` with matching `*.csv` |

**Runs in `results/`**

| Run | Target | Adversary | n | Role |
| --- | --- | --- | --- | --- |
| `hpc_vicuna_autodan` | Vicuna-7B-v1.5 | fluent GA | 100 | **primary** — all headline results |
| `hpc_vicuna` | Vicuna-7B-v1.5 | GCG suffix | 100 | attack-class contrast |
| `hpc_llama2_autodan` | Llama-2-7b-chat | fluent GA | 50 | replication (4 defenses) |
| `hpc_llama2` | Llama-2-7b-chat | GCG suffix | 100 | attack-class contrast |
| `primary_llama32_3b` | Llama-3.2-3B-Instruct | template only | 100 | early pilot |

---

## Defenses

One or two instances per dependency row of Table 6, all applicable to the same A1 jailbreak
threat so that a joint breach is well defined.

| Row | Instances |
| --- | --- |
| Token surface | windowed perplexity filter; surface token-anomaly filter |
| Semantic classification | Llama Guard 3 8B (input + output) |
| First-token distribution | refusal-priming prefix (SafeDecoding stand-in) |
| Perturbation stability | SmoothLLM, q = 4, 10% character perturbation |
| Internal representations | linear probes on hidden states at layers 16 and 8 |

Input-provenance defenses are excluded: they target prompt injection (A0), so a joint breach
with the above is not commensurable.

---

## Layout

```
dcorr/            library
  target.py         one batched forward pass -> target logprob + hidden state + windowed NLL
  attacks/          gcg.py (gradient suffix), autodan.py (fluent GA), random_search.py, static.py
  defenses/         one module per row instance + stack.py
  judge/            StrongREJECT (post hoc, SHA-256-pinned official prompt) + refusal scorer
  analysis/         phi, bootstrap, McNemar, BH; Table 6 emitter
scripts/          00..08 pipeline stages, run_pipeline.sh, slurm_pipeline.sbatch, make_figures.py
configs/          base.yaml + one config per run
data/             evaluation splits and the probe-training pool (regenerable by stage 00)
results/          per-run artifacts, one directory per run
paper/            results_insert.tex (drop-in Results section) and figures/
tests/            CPU-only tests, including a guard that the judge never enters the attack loop
```

## Reproducing

See **[REPRODUCING.md](REPRODUCING.md)** for the full HPC/Slurm procedure. In short:

```bash
conda env create -f environment.yml && conda activate defense-correlation
export HF_TOKEN=... REQUESTY_API_KEY=...
python scripts/fetch_weights.py --models lmsys/vicuna-7b-v1.5 meta-llama/Llama-Guard-3-8B
CONFIG=configs/hpc_vicuna_autodan.yaml bash scripts/run_pipeline.sh
```

Every stage appends to a JSONL keyed by `(model, defense, behaviour, attack)` and skips keys
already present, so an interrupted run resumes. Stages 06 (gold judging) and 07 (analysis)
need no GPU.

Figures are regenerated from the artifacts with:

```bash
python scripts/make_figures.py --fluent results/hpc_vicuna_autodan --suffix results/hpc_vicuna
```

## Validity guards

- The gold judge is **never** reachable from the attack loop; the only in-loop signal is the
  target logprob of an affirmative prefix. A test enforces this statically
  (`tests/test_no_judge_in_loop.py`).
- The StrongREJECT rubric is the official prompt, pinned by SHA-256 and verified at load.
- Probe-training data is deduplicated against every evaluation behaviour by normalised token
  Jaccard; the audit is in `data/probe_dedup_report.json`.
- The analysis flags a run invalid if the in-loop signal saturates while gold ASR stays flat.

## Pre-registration

`PREREGISTRATION.md` fixes the hypotheses, row assignments, decision rules and stopping rules
before the runs. It carries one dated amendment: a second internal-representations instance
was added after observing that the pre-registered same-row pair was unmeasurable under the
suffix attack. That pair is reported as **exploratory**, and the amendment states why.

Known limitations — in-sample threshold calibration, single attack seed, an AutoDAN-*style*
rather than byte-exact attack implementation, and an adaptive column that reports one
optimiser rather than a best-of-strategies adversary — are stated in the Limitations section
of `paper/results_insert.tex`.

## Citation and licence

See `CITATION.cff`. Code is MIT (`LICENSE`). Evaluation prompts come from JailbreakBench and
are redistributed here only as the derived splits stage 00 produces; model weights are not
included.
