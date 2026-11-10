# v2 analysis: v2_llama2 (seeds [0, 1, 2], n=100)

## Marginal ASR

| defense | mean ASR | pooled 95% CI | reliability r |
|---|---|---|---|
| llamaguard | 0.01 | [0.01, 0.03] | 0.23 |
| ppl_filter | 0.18 | [0.14, 0.23] | 0.32 |
| probe | 0.30 | [0.25, 0.35] | 0.47 |
| probe_b | 0.16 | [0.12, 0.21] | 0.31 |
| refusal_prime | 0.05 | [0.03, 0.08] | 0.34 |
| smoothllm | 0.50 | [0.44, 0.55] | 0.36 |
| token_anomaly | 0.10 | [0.07, 0.14] | 0.41 |
| undefended | 0.15 | [0.11, 0.19] | 0.30 |

**Null (undefended seed-pair phi):** 0.30

## E1 behaviour-level phi (separate searches)

| pair | mean phi | disattenuated | partial given undefended | per seed |
|---|---|---|---|---|
| llamaguard|ppl_filter | 0.12 | 0.46 | 0.05 | 0.24, -0.07, 0.20 |
| llamaguard|probe | 0.12 | 0.38 | 0.06 | 0.14, 0.08, 0.15 |
| llamaguard|probe_b | 0.17 | 0.66 | 0.09 | 0.25, 0.32, -0.05 |
| llamaguard|refusal_prime | 0.39 | 1.42 | 0.36 | 0.49, 0.24, 0.44 |
| llamaguard|smoothllm | 0.07 | 0.24 | 0.02 | 0.09, 0.00, 0.11 |
| llamaguard|token_anomaly | 0.27 | 0.89 | 0.20 | 0.37, 0.17, 0.27 |
| ppl_filter|probe | 0.36 | 0.93 | 0.23 | 0.35, 0.31, 0.42 |
| ppl_filter|probe_b | 0.36 | 1.16 | 0.21 | 0.40, 0.32, 0.37 |
| ppl_filter|refusal_prime | 0.22 | 0.66 | 0.16 | 0.06, 0.37, 0.23 |
| ppl_filter|smoothllm | 0.23 | 0.68 | 0.14 | 0.23, 0.19, 0.28 |
| ppl_filter|token_anomaly | 0.34 | 0.94 | 0.17 | 0.21, 0.53, 0.28 |
| probe|probe_b | 0.37 | 0.99 | 0.25 | 0.44, 0.23, 0.44 |
| probe|refusal_prime | 0.30 | 0.75 | 0.25 | 0.18, 0.48, 0.24 |
| probe|smoothllm | 0.24 | 0.57 | 0.16 | 0.34, 0.22, 0.15 |
| probe|token_anomaly | 0.41 | 0.94 | 0.29 | 0.38, 0.36, 0.48 |
| probe_b|refusal_prime | 0.17 | 0.54 | 0.11 | 0.21, 0.29, 0.02 |
| probe_b|smoothllm | 0.21 | 0.62 | 0.11 | 0.21, 0.30, 0.11 |
| probe_b|token_anomaly | 0.51 | 1.46 | 0.38 | 0.57, 0.32, 0.65 |
| refusal_prime|smoothllm | 0.10 | 0.30 | 0.05 | 0.19, -0.03, 0.15 |
| refusal_prime|token_anomaly | 0.39 | 1.07 | 0.36 | 0.34, 0.50, 0.34 |
| smoothllm|token_anomaly | 0.06 | 0.15 | -0.07 | 0.02, 0.13, 0.02 |

## E3 assembled stack

| seed | stack ASR | 95% CI | intersection | strongest layer | McNemar p (vs strongest) |
|---|---|---|---|---|---|
| 0 | 0.00 | [0.00, 0.04] | 0.01 | llamaguard 0.01 | 1.000 |
| 1 | 0.00 | [0.00, 0.04] | 0.00 | llamaguard 0.02 | 0.500 |
| 2 | 0.00 | [0.00, 0.04] | 0.00 | llamaguard 0.01 | 1.000 |

## How much is behaviour difficulty? (random-intercept logistic model)

Behaviour SD 1.81, ICC 0.50 (2400 observations). Predicted phi assumes the pair fails independently given the behaviour.

| pair | observed phi | predicted by difficulty alone | excess |
|---|---|---|---|
| llamaguard|ppl_filter | 0.12 | 0.15 | -0.03 |
| llamaguard|probe | 0.12 | 0.14 | -0.01 |
| llamaguard|probe_b | 0.17 | 0.15 | 0.02 |
| llamaguard|refusal_prime | 0.39 | 0.15 | 0.24 |
| llamaguard|smoothllm | 0.07 | 0.11 | -0.04 |
| llamaguard|token_anomaly | 0.27 | 0.16 | 0.11 |
| ppl_filter|probe | 0.36 | 0.31 | 0.05 |
| ppl_filter|probe_b | 0.36 | 0.30 | 0.06 |
| ppl_filter|refusal_prime | 0.22 | 0.23 | -0.02 |
| ppl_filter|smoothllm | 0.23 | 0.28 | -0.05 |
| ppl_filter|token_anomaly | 0.34 | 0.28 | 0.06 |
| probe|probe_b | 0.37 | 0.31 | 0.07 |
| probe|refusal_prime | 0.30 | 0.22 | 0.08 |
| probe|smoothllm | 0.24 | 0.33 | -0.10 |
| probe|token_anomaly | 0.41 | 0.27 | 0.14 |
| probe_b|refusal_prime | 0.17 | 0.23 | -0.06 |
| probe_b|smoothllm | 0.21 | 0.27 | -0.07 |
| probe_b|token_anomaly | 0.51 | 0.28 | 0.24 |
| refusal_prime|smoothllm | 0.10 | 0.18 | -0.08 |
| refusal_prime|token_anomaly | 0.39 | 0.23 | 0.16 |
| smoothllm|token_anomaly | 0.06 | 0.23 | -0.17 |

## Seed 0: diversity, strategies, objective, grader parsing

Non-pairwise diversity over the layers (Kuncheva & Whitaker 2003): theta 0.040, entropy 0.363, kw_variance 0.109, kappa 0.149

| defense | best-of-strategies ASR | attacks pooled |
|---|---|---|
| llamaguard | 0.02 | adaptive, static_gcg, static_plain |
| ppl_filter | 0.15 | adaptive, static_gcg, static_plain |
| probe | 0.34 | adaptive, static_gcg, static_plain |
| probe_b | 0.15 | adaptive, static_gcg, static_plain |
| refusal_prime | 0.04 | adaptive, static_gcg, static_plain |
| smoothllm | 0.53 | adaptive, static_gcg, static_plain |
| token_anomaly | 0.08 | adaptive, static_gcg, static_plain |
| undefended | 0.16 | adaptive, static_gcg, static_plain |
| stack | 0.00 | adaptive, static_gcg, static_plain |

| defense | AUROC of in-loop logprob for grader breach |
|---|---|
| llamaguard | 1.00 |
| ppl_filter | 0.84 |
| probe | 0.59 |
| probe_b | 0.86 |
| refusal_prime | 0.51 |
| smoothllm | 0.64 |
| token_anomaly | 0.74 |
| undefended | 0.79 |

| defense | judged | labels with an unparsed vote |
|---|---|---|
| llamaguard | 7 | 0 |
| ppl_filter | 200 | 0 |
| probe | 65 | 0 |
| probe_b | 256 | 0 |
| refusal_prime | 300 | 0 |
| smoothllm | 300 | 5 |
| token_anomaly | 300 | 0 |
| undefended | 300 | 0 |
| stack | 6 | 0 |

## Seed 1: diversity, strategies, objective, grader parsing

Non-pairwise diversity over the layers (Kuncheva & Whitaker 2003): theta 0.045, entropy 0.353, kw_variance 0.107, kappa 0.180

| defense | best-of-strategies ASR | attacks pooled |
|---|---|---|
| llamaguard | 0.02 | adaptive, static_gcg, static_plain |
| ppl_filter | 0.19 | adaptive, static_gcg, static_plain |
| probe | 0.27 | adaptive, static_gcg, static_plain |
| probe_b | 0.17 | adaptive, static_gcg, static_plain |
| refusal_prime | 0.07 | adaptive, static_gcg, static_plain |
| smoothllm | 0.49 | adaptive, static_gcg, static_plain |
| token_anomaly | 0.14 | adaptive, static_gcg, static_plain |
| undefended | 0.19 | adaptive, static_gcg, static_plain |
| stack | 0.00 | adaptive, static_gcg, static_plain |

| defense | AUROC of in-loop logprob for grader breach |
|---|---|
| llamaguard | 0.25 |
| ppl_filter | 0.89 |
| probe | 0.49 |
| probe_b | 0.86 |
| refusal_prime | 0.97 |
| smoothllm | 0.60 |
| token_anomaly | 0.85 |
| undefended | 0.79 |

| defense | judged | labels with an unparsed vote |
|---|---|---|
| llamaguard | 8 | 0 |
| ppl_filter | 200 | 0 |
| probe | 61 | 0 |
| probe_b | 256 | 0 |
| refusal_prime | 300 | 0 |
| smoothllm | 300 | 4 |
| token_anomaly | 300 | 0 |
| undefended | 300 | 0 |
| stack | 4 | 0 |

## Seed 2: diversity, strategies, objective, grader parsing

Non-pairwise diversity over the layers (Kuncheva & Whitaker 2003): theta 0.045, entropy 0.373, kw_variance 0.109, kappa 0.174

| defense | best-of-strategies ASR | attacks pooled |
|---|---|---|
| llamaguard | 0.01 | adaptive, static_gcg, static_plain |
| ppl_filter | 0.20 | adaptive, static_gcg, static_plain |
| probe | 0.32 | adaptive, static_gcg, static_plain |
| probe_b | 0.20 | adaptive, static_gcg, static_plain |
| refusal_prime | 0.05 | adaptive, static_gcg, static_plain |
| smoothllm | 0.47 | adaptive, static_gcg, static_plain |
| token_anomaly | 0.14 | adaptive, static_gcg, static_plain |
| undefended | 0.17 | adaptive, static_gcg, static_plain |
| stack | 0.00 | adaptive, static_gcg, static_plain |

| defense | AUROC of in-loop logprob for grader breach |
|---|---|
| llamaguard | 0.00 |
| ppl_filter | 0.89 |
| probe | 0.64 |
| probe_b | 0.92 |
| refusal_prime | 0.88 |
| smoothllm | 0.63 |
| token_anomaly | 0.88 |
| undefended | 0.86 |

| defense | judged | labels with an unparsed vote |
|---|---|---|
| llamaguard | 8 | 0 |
| ppl_filter | 200 | 0 |
| probe | 67 | 0 |
| probe_b | 256 | 0 |
| refusal_prime | 300 | 0 |
| smoothllm | 300 | 6 |
| token_anomaly | 300 | 0 |
| undefended | 300 | 0 |
| stack | 2 | 0 |

## Held-out member selection, seed 0, all layers (500 splits; test-half intersection residual)

| k | diversity-greedy | one per row | random subset |
|---|---|---|---|
| 1 | 0.01 | 0.01 | 0.18 |
| 2 | 0.01 | 0.01 | 0.06 |
| 3 | 0.01 | 0.01 | 0.02 |
| 4 | 0.01 | 0.01 | 0.01 |

Single best layer alone: 0.01

## Held-out member selection, seed 0, without llamaguard (500 splits; test-half intersection residual)

| k | diversity-greedy | one per row | random subset |
|---|---|---|---|
| 1 | 0.04 | 0.06 | 0.21 |
| 2 | 0.02 | 0.02 | 0.07 |
| 3 | 0.02 | 0.02 | 0.04 |
| 4 | 0.01 | 0.02 | 0.02 |

Single best layer alone: 0.04

## Held-out member selection, seed 1, all layers (500 splits; test-half intersection residual)

| k | diversity-greedy | one per row | random subset |
|---|---|---|---|
| 1 | 0.02 | 0.02 | 0.19 |
| 2 | 0.01 | 0.01 | 0.06 |
| 3 | 0.01 | 0.01 | 0.03 |
| 4 | 0.01 | 0.01 | 0.02 |

Single best layer alone: 0.02

## Held-out member selection, seed 1, without llamaguard (500 splits; test-half intersection residual)

| k | diversity-greedy | one per row | random subset |
|---|---|---|---|
| 1 | 0.07 | 0.08 | 0.21 |
| 2 | 0.05 | 0.05 | 0.08 |
| 3 | 0.04 | 0.04 | 0.05 |
| 4 | 0.03 | 0.03 | 0.04 |

Single best layer alone: 0.07

## Held-out member selection, seed 2, all layers (500 splits; test-half intersection residual)

| k | diversity-greedy | one per row | random subset |
|---|---|---|---|
| 1 | 0.01 | 0.01 | 0.19 |
| 2 | 0.01 | 0.01 | 0.06 |
| 3 | 0.01 | 0.01 | 0.03 |
| 4 | 0.00 | 0.00 | 0.01 |

Single best layer alone: 0.01

## Held-out member selection, seed 2, without llamaguard (500 splits; test-half intersection residual)

| k | diversity-greedy | one per row | random subset |
|---|---|---|---|
| 1 | 0.05 | 0.06 | 0.22 |
| 2 | 0.02 | 0.03 | 0.09 |
| 3 | 0.02 | 0.01 | 0.04 |
| 4 | 0.00 | 0.00 | 0.02 |

Single best layer alone: 0.05

## Benign false-refusal rate, seed 0, jbb set

| configuration | FRR | 95% CI | n |
|---|---|---|---|
| llamaguard | 0.54 | [0.44, 0.63] | 100 |
| ppl_filter | 0.50 | [0.40, 0.60] | 100 |
| probe_b | 0.68 | [0.58, 0.76] | 100 |
| probe | 0.53 | [0.43, 0.62] | 100 |
| refusal_prime | 0.99 | [0.95, 1.00] | 100 |
| smoothllm | 0.53 | [0.43, 0.62] | 100 |
| token_anomaly | 0.50 | [0.40, 0.60] | 100 |
| undefended | 0.50 | [0.40, 0.60] | 100 |
| stack | 1.00 | [0.96, 1.00] | 100 |

## Benign false-refusal rate, seed 0, orbench set

| configuration | FRR | 95% CI | n |
|---|---|---|---|
| llamaguard | 0.91 | [0.88, 0.93] | 400 |
| ppl_filter | 0.88 | [0.85, 0.91] | 400 |
| probe_b | 0.89 | [0.86, 0.92] | 400 |
| probe | 0.91 | [0.88, 0.94] | 400 |
| refusal_prime | 1.00 | [0.99, 1.00] | 400 |
| smoothllm | 0.83 | [0.79, 0.86] | 400 |
| token_anomaly | 0.88 | [0.85, 0.91] | 400 |
| undefended | 0.88 | [0.85, 0.91] | 400 |
| stack | 1.00 | [0.99, 1.00] | 400 |

## Benign false-refusal rate, seed 1, jbb set

| configuration | FRR | 95% CI | n |
|---|---|---|---|
| llamaguard | 0.54 | [0.44, 0.63] | 100 |
| ppl_filter | 0.50 | [0.40, 0.60] | 100 |
| probe_b | 0.68 | [0.58, 0.76] | 100 |
| probe | 0.53 | [0.43, 0.62] | 100 |
| refusal_prime | 0.99 | [0.95, 1.00] | 100 |
| smoothllm | 0.53 | [0.43, 0.62] | 100 |
| token_anomaly | 0.50 | [0.40, 0.60] | 100 |
| undefended | 0.50 | [0.40, 0.60] | 100 |
| stack | 1.00 | [0.96, 1.00] | 100 |

## Benign false-refusal rate, seed 1, orbench set

| configuration | FRR | 95% CI | n |
|---|---|---|---|
| llamaguard | 0.91 | [0.88, 0.93] | 400 |
| ppl_filter | 0.88 | [0.85, 0.91] | 400 |
| probe_b | 0.89 | [0.86, 0.92] | 400 |
| probe | 0.91 | [0.88, 0.94] | 400 |
| refusal_prime | 1.00 | [0.99, 1.00] | 400 |
| smoothllm | 0.83 | [0.79, 0.86] | 400 |
| token_anomaly | 0.88 | [0.85, 0.91] | 400 |
| undefended | 0.88 | [0.85, 0.91] | 400 |
| stack | 1.00 | [0.99, 1.00] | 400 |

## Benign false-refusal rate, seed 2, jbb set

| configuration | FRR | 95% CI | n |
|---|---|---|---|
| llamaguard | 0.54 | [0.44, 0.63] | 100 |
| ppl_filter | 0.50 | [0.40, 0.60] | 100 |
| probe_b | 0.68 | [0.58, 0.76] | 100 |
| probe | 0.53 | [0.43, 0.62] | 100 |
| refusal_prime | 0.99 | [0.95, 1.00] | 100 |
| smoothllm | 0.53 | [0.43, 0.62] | 100 |
| token_anomaly | 0.50 | [0.40, 0.60] | 100 |
| undefended | 0.50 | [0.40, 0.60] | 100 |
| stack | 1.00 | [0.96, 1.00] | 100 |

## Benign false-refusal rate, seed 2, orbench set

| configuration | FRR | 95% CI | n |
|---|---|---|---|
| llamaguard | 0.91 | [0.88, 0.93] | 400 |
| ppl_filter | 0.88 | [0.85, 0.91] | 400 |
| probe_b | 0.89 | [0.86, 0.92] | 400 |
| probe | 0.91 | [0.88, 0.94] | 400 |
| refusal_prime | 1.00 | [0.99, 1.00] | 400 |
| smoothllm | 0.83 | [0.79, 0.86] | 400 |
| token_anomaly | 0.88 | [0.85, 0.91] | 400 |
| undefended | 0.88 | [0.85, 0.91] | 400 |
| stack | 1.00 | [0.99, 1.00] | 400 |
