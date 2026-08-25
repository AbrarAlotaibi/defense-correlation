# Reproducing the experiments

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
