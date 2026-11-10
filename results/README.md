# Results

One directory per run. A run directory holds the resolved configuration, the calibrated
thresholds, one JSONL file of responses per safeguard and stage, the grader's verdicts, and
the analysis outputs computed from them.

## Runs reported in the paper

| Run | Config | Target | Contents |
| --- | --- | --- | --- |
| `v2_vicuna` | `configs/v2_vicuna.yaml`, `configs/v2_vicuna_pair.yaml` | Vicuna-7B-v1.5 | All arms, three seeds, cross-evaluation, assembled stacks, both attack families, the adjudication sample |
| `v2_vicuna_nolg` | `configs/v2_vicuna_nolg.yaml` | Vicuna-7B-v1.5 | Six-member stack without the guard model |
| `v2_llama2` | `configs/v2_llama2.yaml` | Llama-2-7b-chat | Replication, three seeds |
| `v2_llama2_nolg` | `configs/v2_llama2_nolg.yaml` | Llama-2-7b-chat | Six-member stack without the guard model |

Inside a run, `seed0`, `seed1` and `seed2` are the three replicates. In each, `gold.jsonl`
holds the first grader's verdicts and `gold_resolved.jsonl` the labels the paper uses, after
abstentions were resolved by the second grader.

`v2_vicuna_smoke` and `v2_vicuna_smoke_nolg` are five-behavior shape checks of the same
configuration.

## Earlier version

`hpc_vicuna`, `hpc_vicuna_autodan`, `hpc_vicuna_autodan_extthr`, `hpc_vicuna_autodan_nolg`,
`hpc_llama2`, `hpc_llama2_autodan`, `hpc_llama2_smoke`, `primary_llama2`,
`primary_llama2_smoke` and `primary_llama32_3b` belong to the earlier manuscript described in
`earlier_version/README.md`. Their numbers are superseded.
