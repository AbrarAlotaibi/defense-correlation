# Results: primary_llama32_3b (meta-llama/Llama-3.2-3B-Instruct)

## Marginal ASR (adaptive, gold)

- `refusal_prime`: 0.44
- `probe`: 0.26
- `undefended`: 0.25
- `smoothllm`: 0.05
- `llamaguard`: 0.01
- `ppl_filter`: 0.00
- `token_anomaly`: 0.00

## Headline hypotheses

- **H1** `ppl_filterxtoken_anomaly`: phi=0.00 CI=[0.0, 0.0] -> **NOT supported**
- **H2** `ppl_filterxprobe`: phi=0.00 CI=[0.0, 0.0] -> **SUPPORTED**

## Adaptive vs static gap

- `refusal_prime`: adaptive 0.44 - static 0.00 = **0.44**
- `probe`: adaptive 0.26 - static 0.03 = **0.23**
- `smoothllm`: adaptive 0.05 - static 0.00 = **0.05**
- `llamaguard`: adaptive 0.01 - static 0.01 = **0.00**
- `ppl_filter`: adaptive 0.00 - static 0.00 = **0.00**
- `token_anomaly`: adaptive 0.00 - static 0.03 = **-0.03**