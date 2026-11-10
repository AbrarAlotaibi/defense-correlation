# Reproducing the experiments

## The current harness: read this first

The runs of the earlier preprint (`results/hpc_*`, `results/primary_*`) were produced by a
harness with four defects. The current harness, which produced every `results/v2_*` run the
paper reports, fixes them:

| defect | earlier behaviour | fix |
| --- | --- | --- |
| F2 seeding | every defense used one RNG stream, reset per chunk, so inert defenses replayed one search | one seed per (replicate, defense, behaviour), `dcorr/seeding.py`; `--seed-index k` |
| F1 stack feasibility | stack filters checked on SmoothLLM's perturbed copies, under refusal-prime's system prompt; GCG skipped them with Llama Guard present | `dcorr/attacks/feasibility.py`: raw prompt, default system prompt, as deployed |
| F4 Llama Guard | never consulted by the AutoDAN or GCG search | AutoDAN lockstep rounds with guard blocks (`attacks/autodan.py`); GCG x Llama Guard remains non-adaptive (stated limitation) |
| grader | identical responses graded independently (7% self-disagreement) | verdict cache by exact text, `judge.reps` with majority (`dcorr/judge/cache.py`) |

The reported v2 runs executed on a KFUPM H100 (80 GB) under SLURM, via
`scripts/slurm_v2_core.sbatch` and the other `slurm_v2_*` jobs. The configuration is
nonetheless sized for a 16 GB card and was not relaxed for the larger one, so it reproduces on
either: fp16 Vicuna-7B with fp16 Llama Guard 3 8B under `reload` residency (loaded per guard
block, target evicted, layers beyond `max_gpu_memory` on CPU). Keep `target.batch_size` as it
is wherever you run this, because batch-dependent arithmetic accounts for part of the
run-to-run variation the paper reports.

The commands below are what the SLURM jobs invoke, and run unchanged on one card:

```bash
python scripts/25_run_grid.py --config configs/v2_vicuna.yaml --seeds 0 1 2      # 03/04/12/05/24
python scripts/25_run_grid.py --config configs/v2_vicuna_nolg.yaml --seeds 0 1 2 --stages 05
python scripts/06_judge_gold.py --config configs/v2_vicuna.yaml --seed-index 0   # per seed
python scripts/06_judge_gold.py --config configs/v2_vicuna.yaml --seed-index 0 --src-dir cross
python scripts/24_cross_eval.py --config configs/v2_vicuna.yaml --seed-index 0 --export
python scripts/26_v2_analysis.py --config configs/v2_vicuna.yaml --seeds 0 1 2
```

Thresholds in v2 are the external (AlpacaEval) calibration, copied into
`results/v2_vicuna*/calibration.json`. Tests:

```bash
python -m pytest tests -q                                  # CPU, ~1 min
DCORR_E2E=1 python -m pytest tests/e2e -q -s               # GPU, real models, ~20 min
```

The end-to-end suite runs the real stage scripts on Llama-3.2-3B with real Llama Guard and
also runs the v1 code from the tag, asserting that v1 shows each defect and v2 does not.

---

## v1 (as submitted)

The reported runs were produced on a single H100 (80 GB) on a Slurm cluster. Any GPU with
≥40 GB will run the primary configuration unchanged; smaller cards need the batch-size notes
in §7.

## 1. Environment

```bash
conda env create -f environment.yml && conda activate defense-correlation
# or: pip install -r requirements.txt into a CUDA-enabled Python 3.11
python -m pytest tests/ -q          # CPU only, ~14 tests, no GPU needed
```

`dcorr/compat.py` supports transformers 4.45+ and 5.x, so no pin is forced. If a checkpoint
ships a SentencePiece tokenizer (Vicuna does), `sentencepiece` must be installed.

## 2. Credentials

Two secrets, read from the environment or a `.env` file at the repo root, never printed:

- `HF_TOKEN` — Hugging Face, approved for the gated Meta models (also exported as
  `HUGGING_FACE_HUB_TOKEN`).
- `REQUESTY_API_KEY` — the StrongREJECT gold judge, routed through an OpenAI-compatible
  endpoint (`https://router.requesty.ai/v1`). Only stage 06 needs it. Any OpenAI-compatible
  endpoint works; set `judge.base_url` and `judge.api_key_env` in the config.

## 3. Weights

Fetch on a node with internet; compute nodes then read the cache offline.

```bash
python scripts/fetch_weights.py --models \
  lmsys/vicuna-7b-v1.5 meta-llama/Llama-2-7b-chat-hf meta-llama/Llama-Guard-3-8B
```

Point `HF_HOME` at a filesystem both node types can see. Note that some checkpoints ship
only PyTorch `.bin` shards and no safetensors, which the fetcher allows.

## 4. Running

```bash
CONFIG=configs/hpc_vicuna_autodan.yaml bash scripts/run_pipeline.sh
```

Stages, in the order the orchestrator runs them:

| # | script | what it does | needs |
| --- | --- | --- | --- |
| 00 | `00_prepare_data.py` | JBB harmful/benign split + probe-training pool, deduplicated | HF cache |
| 08 | `08_smoke_test.py` | 3 behaviours × 5 iterations, wiring check | GPU |
| 02 | `02_train_probe.py` | linear probe on mid-layer hidden states | GPU |
| 01 | `01_calibrate.py` | filter thresholds to a 1% benign block rate | GPU |
| 03 | `03_positive_control.py` | undefended control; **gate** on the grid | GPU |
| 04 | `04_run_attacks.py` | the grid: every defense, adaptive + static, harmful + benign | GPU |
| 05 | `05_run_stack.py` | direct attack on the assembled stack | GPU |
| 06 | `06_judge_gold.py` | StrongREJECT gold judging, post hoc | internet |
| 07 | `07_analyze.py` | Table 6, H1–H3, bootstrap, McNemar | CPU |

Stage 02 runs before 01 because the probe threshold is calibrated from the trained probe.

**Resumability.** Each stage appends to a JSONL keyed by
`(model, defense, behaviour, attack)` and skips keys already present. Stage 04 additionally
persists every `attack.behaviour_chunk` behaviours (default 10), so an interrupted job loses
at most one chunk rather than a whole defense.

## 5. Air-gapped compute nodes

Most clusters give internet to login nodes only. Two stages need it: the weight fetch (§3)
and gold judging (stage 06). Split the run:

```bash
sbatch scripts/slurm_pipeline.sbatch                                    # stages 00-05, offline
CONFIG=configs/hpc_vicuna_autodan.yaml API_ONLY=1 bash scripts/run_pipeline.sh   # 06-07, login node
```

Edit the sbatch header for your partition, account and environment activation. If your
compute nodes have internet, drop `GPU_ONLY=1` from the sbatch and run 00–07 in one job.

## 6. Configurations

| Config | Target | Adversary | Notes |
| --- | --- | --- | --- |
| `hpc_vicuna_autodan.yaml` | Vicuna-7B-v1.5 | fluent GA | **primary**, all headline results |
| `hpc_vicuna.yaml` | Vicuna-7B-v1.5 | GCG suffix | attack-class contrast |
| `hpc_llama2_autodan.yaml` | Llama-2-7b-chat | fluent GA | replication, `data.eval_limit: 50`, 4 defenses |
| `hpc_llama2.yaml` | Llama-2-7b-chat | GCG suffix | attack-class contrast |
| `primary_llama2.yaml`, `primary_llama32_3b.yaml` | — | — | early single-GPU configs, kept for provenance |

Key knobs: `attack.adaptive.method` (`autodan` | `gcg` | `random_search` | `template_only`),
`attack.adaptive.use_template`, `data.eval_limit`, and `positive_control.abort_on_fail`.

## 8. Checks on the manuscript and the results

These do not run the experiments; they check that what the paper says matches what the code and
the artefacts say. Each was written because it caught something the others could not see.

| script | what it checks | what it found |
|---|---|---|
| `32_texcheck.py` | balanced environments, brace depth, duplicate and undefined labels | structural errors before an Overleaf round trip |
| `33_checkcites.py` | every `\cite` key resolves in the .bib | missing keys |
| `37_tablecheck.py` | each tabular's declared column count against the fields in every row | a table declaring six columns against five-field rows, which LaTeX renders silently as a stray empty column |
| `38_grader_ceiling.py` | phi between two independent gradings of byte-identical responses | the ceiling of the measurement, 0.926, which is what phi = 1 looks like under this grader |
| `39_verify_claims.py` | every quantity quoted in the manuscript, in both directions | a stale two-replicate range still in the Discussion after the Results copy had been updated |
| `40_verify_design.py` | every method parameter in the Design section against the config that ran | nothing yet; all 17 parameters and 4 set sizes agree |

The reverse direction of `39` is the one worth keeping. A forward check confirms the right
numbers are present; only the reverse can see a stale number sitting beside them.

Usage, from the repository root with the manuscript alongside:

```
python scripts/32_texcheck.py   <paper>/main.tex
python scripts/33_checkcites.py <paper>/main.tex <paper>/cas-refs.bib
python scripts/37_tablecheck.py <paper>/main.tex
python scripts/39_verify_claims.py
python scripts/40_verify_design.py
```

## 7. Practical notes

These cost real debugging time; they are recorded so you do not repeat them.

- **Attack class decides what a surface filter can measure.** A GCG suffix has a windowed
  NLL around 13.6 against a 1%-FPR threshold near 4.8, so a perplexity filter blocks
  essentially all of it and its residual ASR is 0 — an *undefined* correlation, not an
  independent one. The fluent attack sits below the threshold and makes the row measurable.
- **Memory.** The scoring path slices logits to the scored positions and chunks the
  `logsumexp` over time; the probe layer is captured with a forward hook rather than
  `output_hidden_states=True`. Both matter on long prompts and large vocabularies. If a run
  OOMs, lower `target.batch_size` (8 is safe for a 7B on 40 GB with a co-resident guard).
- **Do not run two GPU jobs on one node.** Contention roughly halved throughput and
  triggered the OOM that lost a full defense before chunked persistence was added.
- **Llama Guard 3 8B** needs ~16 GB in fp16. On a ≥40 GB card keep it resident
  (`residency: resident_gpu`); on smaller cards set `load_in_4bit: true` or use
  `residency: remote` against an OpenAI-compatible endpoint.
- **The gate is a proxy**, not gold: it uses a non-refusal + affirmative-continuation
  heuristic to catch a broken setup cheaply. Never quote `positive_control.json` as an ASR;
  gold ASRs come from stage 07.
