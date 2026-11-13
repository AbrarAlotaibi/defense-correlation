# How Diverse Are Layered Safeguards?

Code, data and analysis for the paper

> Abrar Alotaibi and Moataz Ahmed. *How Diverse Are Layered Safeguards? Estimands and
> Measurement for Ensembles of LLM Jailbreak Defenses.* Manuscript submitted for
> publication, 2026.

Language models are deployed behind stacks of safeguards on the assumption that the layers
fail on different inputs. This repository measures that assumption. It contains the
safeguard implementations, two attack harnesses, every model response and grader verdict
behind the paper, and the scripts that regenerate each table and figure from them.

> **Content note.** The result files contain adversarial prompts and the harmful text some
> of them elicited from open-weight models. They are included because the measurements
> cannot be audited without them. Use them for research on evaluating and improving
> safeguards.

## What is measured

"Two safeguards fail on different inputs" is three different quantities once the adversary
adapts to each safeguard separately. The paper defines an estimand for each and this code
estimates all three.

| Estimand | Question | Main script | Output |
| --- | --- | --- | --- |
| E1, behavior level | Are the same behaviors jointly vulnerable under separate searches? | `scripts/26_v2_analysis.py` | `analysis_v2_resolved.json` |
| E2, prompt level | Do two safeguards fail on the same shared input? | `scripts/24_cross_eval.py`, `scripts/44_e2_passthrough.py` | `e2_passthrough.json` |
| E3, assembled stack | What does an adversary achieve against the whole pipeline? | `scripts/05_run_stack.py`, `scripts/47_e3_table.py` | `e3_table.json` |

Each estimate is read against a control. For E1 the control is the correlation between two
independent runs of the same attack on the undefended model, which is the level a pair of
safeguards has to exceed before its correlation says anything about the safeguards.

## Findings

1. No pair of safeguards is distinguishable from that control. The largest excess over it
   is +0.026 under the genetic attack and +0.030 under the LLM-driven attack, against
   controls whose three estimates span 0.094 and 0.223. On the primary target, behavior
   difficulty accounts for 71% of the mean pairwise association.
2. The estimate composed from layers measured one at a time is optimistic. It is below the
   measured residual of the assembled stack in all six comparisons that have resolution,
   and it predicts a false-refusal rate of 0.60 where the assembled stack measures 0.83.
3. A safeguard's measured strength depends on the adversary. A likelihood filter blocks
   every universal-suffix prompt and no fluent prompt. The guard model admits 0.03 under
   the genetic attack and 0.19 under the LLM-driven attack, which used one thirtieth of the
   search budget.

## Setup

| | |
| --- | --- |
| Targets | Vicuna-7B-v1.5 (primary), Llama-2-7b-chat (replication), both fp16 |
| Behaviors | 100 JailbreakBench harmful behaviors, with matched benign prompts and a 400-prompt OR-Bench sample |
| Replicates | Three independent attack seeds for every arm |
| Attacks | Genetic fluent search (AutoDAN style), LLM-driven rewriting (PAIR style), a static universal suffix, and plain requests |
| Grading | StrongREJECT rubric, five judgments per unique response, majority label; abstentions resolved by a second grader from a different developer |
| Calibration | Every thresholded safeguard set to a 1% false-positive rate on an external corpus of 805 AlpacaEval instructions |

Seven safeguards and an undefended control:

| Safeguard | Family | Module |
| --- | --- | --- |
| Windowed likelihood filter | token surface | `dcorr/defenses/ppl_filter.py` |
| Token-anomaly statistic | token surface | `dcorr/defenses/token_anomaly.py` |
| Guard model (Llama Guard 3 8B), input and output | semantic | `dcorr/defenses/llamaguard.py` |
| Hidden-state probe, layer 16 | representation | `dcorr/defenses/probe.py` |
| Hidden-state probe, layer 8 | representation | `dcorr/defenses/probe.py` |
| Refusal-priming system prompt | first token | `dcorr/defenses/refusal_prime.py` |
| Perturbation aggregator (SmoothLLM, q = 4, 10% swap) | perturbation | `dcorr/defenses/smoothllm.py` |

## Layout

```
dcorr/             library: target wrapper, attacks, safeguards, stack, grader, analysis
scripts/           pipeline stages, analysis and manuscript checks (index in scripts/README.md)
configs/           one YAML per run; v2_* are the runs the paper reports
data/              evaluation splits, calibration corpus, trained probes
results/           one directory per run (index in results/README.md)
tests/             CPU-only tests
earlier_version/   material from the earlier manuscript, kept for the record
```

Runs reported in the paper:

| Run | Target | Contents |
| --- | --- | --- |
| `results/v2_vicuna` | Vicuna-7B-v1.5 | all arms, three seeds, cross-evaluation, assembled stacks, both attack families |
| `results/v2_vicuna_nolg` | Vicuna-7B-v1.5 | six-member stack without the guard model |
| `results/v2_llama2` | Llama-2-7b-chat | replication, three seeds |
| `results/v2_llama2_nolg` | Llama-2-7b-chat | six-member stack without the guard model |

## Where each reported number comes from

No number in the paper is typed by hand. Each comes from one of these files, and
`scripts/39_verify_claims.py` recomputes 189 of them and checks them against the manuscript
text.

| Paper section | File in `results/<run>/` | Produced by |
| --- | --- | --- |
| Attack reproducibility and E1 | `analysis_v2_resolved.json`, `analysis_v2_pair.json` | `26_v2_analysis.py` |
| E2 and the pass-through diagnostic | `e2_passthrough.json` | `44_e2_passthrough.py` |
| Transfer matrix | `transfer_structure.json` | `46_transfer_structure.py` |
| E3 | `e3_table.json` | `47_e3_table.py` |
| Over-refusal | `frr_composition.json` | `43_frr_composition.py` |
| Second attack family | `family_comparison.json` | `48_family_comparison.py` |
| What the probes detect | `probe_validity.json` | `30_probe_validity.py` |
| Grader ceiling | printed by | `38_grader_ceiling.py` |
| Abstentions and cross-grader agreement | `seed*/abstentions.json`, `seed*/gold_resolved.jsonl` | `35_resolve_abstentions.py` |
| Figures | `figures/` | `34_figures_v2.py`, `41_fig_replication.py` |

## Reproducing

[REPRODUCING.md](REPRODUCING.md) has the full procedure. The analysis needs no GPU and runs
from the stored verdicts:

```bash
conda env create -f environment.yml && conda activate defense-correlation
python scripts/26_v2_analysis.py --config configs/v2_vicuna.yaml --seeds 0 1 2
python scripts/39_verify_claims.py
```

Rerunning the attacks needs one GPU with 16 GB or more:

```bash
export HF_TOKEN=...
python scripts/fetch_weights.py --models lmsys/vicuna-7b-v1.5 meta-llama/Llama-Guard-3-8B
python scripts/25_run_grid.py --config configs/v2_vicuna.yaml --seeds 0 1 2
```

Credentials (`HF_TOKEN`, `OPENAI_API_KEY`) are read from the environment, from a `.env` file
at the repository root, or from the file named by `DCORR_ENV`. Neither is ever committed.

Every stage appends to a JSONL file keyed by model, safeguard, behavior and attack, and
skips keys already present, so an interrupted run resumes. A search is a deterministic
function of the seed, the replicate, the safeguard and the behavior.

## Human adjudication of the grader

A blinded sample of 618 responses checks the automated grader against a person: every
response on which the grader's five judgments split, every response it would not score, and
a random sample of unanimous ones. A stratified subset of 124 is relabeled after an interval
to measure the annotator's own consistency.

```bash
python scripts/28c_build_labeling_tool.py       --dir results/v2_vicuna/adjudication
# open the page (on Windows, double-click open_labeling_tool.bat), label, download the CSV
python scripts/28b_adjudication_single.py merge --dir results/v2_vicuna/adjudication
python scripts/28b_adjudication_single.py score --dir results/v2_vicuna/adjudication
python scripts/28d_adjudication_sensitivity.py  --run v2_vicuna --seeds 0 1 2
python scripts/28e_fill_adjudication.py         --run v2_vicuna
```

Stage 28d compares the annotator with the labels the paper uses and reruns the
behavior-level analysis with the annotator's labels in place of the grader's. Stage 28e
writes the resulting figures into the manuscript. The annotator's labels are in
`labels_main.csv` and `sheet.csv`, and the scored results in `adjudication_single.json` and
`adjudication_sensitivity.json`.

The labeling page is a single offline HTML file. It shows only the request and the
response. The grader's verdicts and the sampling strata are in `key.csv`, which the page
never reads.

## Validity guards

- The grader is never reachable from the attack loop. The only signal inside the loop is the
  target's log-probability of an affirmative prefix, and `tests/test_no_judge_in_loop.py`
  enforces this.
- The StrongREJECT rubric is the official prompt, pinned by SHA-256 and verified at load.
- Verdicts are cached by the exact text judged, so identical responses always receive one
  label.
- Probe-training data is deduplicated against every evaluation behavior
  (`data/probe_dedup_report.json`).
- Runs record the seed, the resolved configuration, the host and the GPU.

## Earlier version

An earlier and longer manuscript reporting a first version of this experiment was posted as
arXiv:2608.28327. Its text inserts, figures and preregistration are in `earlier_version/`,
and its runs are `results/hpc_*` and `results/primary_*`. That harness had four defects,
listed at the top of [REPRODUCING.md](REPRODUCING.md), and its headline numbers are
superseded by the runs above. Nothing in the current paper depends on them.

## Citation

```bibtex
@unpublished{alotaibi2026diverse,
  author = {Alotaibi, Abrar and Ahmed, Moataz},
  title  = {How Diverse Are Layered Safeguards? Estimands and Measurement for
            Ensembles of {LLM} Jailbreak Defenses},
  note   = {Manuscript submitted for publication},
  year   = {2026}
}
```

`CITATION.cff` carries the same record.

## License

Code is released under the MIT License (`LICENSE`). The evaluation prompts derive from
JailbreakBench, OR-Bench and AlpacaEval and are redistributed only as the derived splits
that `scripts/00_prepare_data.py`, `scripts/27_build_benign_orbench.py` and
`scripts/14_build_external_benign.py` produce. Model weights are not included.
