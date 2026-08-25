# Results: hpc_llama2_autodan (meta-llama/Llama-2-7b-chat-hf)

## Marginal ASR (adaptive, gold)

- `ppl_filter`: 0.44
- `probe_b`: 0.44
- `undefended`: 0.40
- `probe`: 0.28
- `token_anomaly`: 0.02

## Headline hypotheses

- **H1** `probexprobe_b`: phi=0.25 CI=[-0.03, 0.52] -> **NOT supported**
- **H2** `ppl_filterxprobe`: phi=0.17 CI=[-0.12, 0.44] -> **UNDERPOWERED**
- **H2** `probextoken_anomaly`: phi=-0.09 CI=[-0.17, 0.0] -> **SUPPORTED**

## Adaptive vs static gap

- `ppl_filter`: adaptive 0.44 - static 0.00 = **0.44**
- `probe_b`: adaptive 0.44 - static 0.02 = **0.42**
- `probe`: adaptive 0.28 - static 0.02 = **0.26**
- `token_anomaly`: adaptive 0.02 - static 0.02 = **0.00**