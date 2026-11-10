# v2 analysis: v2_vicuna (seeds [0, 1, 2], n=100)

## Marginal ASR

| defense | mean ASR | pooled 95% CI | reliability r |
|---|---|---|---|
| llamaguard | 0.19 | [0.15, 0.24] | 0.36 |
| ppl_filter | 0.24 | [0.20, 0.29] | 0.29 |
| probe | 0.21 | [0.16, 0.26] | 0.09 |
| probe_b | 0.21 | [0.16, 0.26] | 0.17 |
| refusal_prime | 0.07 | [0.05, 0.11] | 0.29 |
| smoothllm | 0.19 | [0.15, 0.23] | 0.24 |
| token_anomaly | 0.24 | [0.20, 0.29] | 0.26 |
| undefended | 0.22 | [0.18, 0.27] | 0.30 |

**Null (undefended seed-pair phi):** 0.30

## E1 behaviour-level phi (separate searches)

| pair | mean phi | disattenuated | partial given undefended | per seed |
|---|---|---|---|---|
| llamaguard|ppl_filter | 0.22 | 0.69 | 0.08 | 0.37, -0.03, 0.32 |
| llamaguard|probe | 0.16 | 0.90 | 0.04 | 0.32, -0.16, 0.32 |
| llamaguard|probe_b | 0.25 | 1.01 | 0.09 | 0.26, 0.11, 0.38 |
| llamaguard|refusal_prime | 0.08 | 0.25 | -0.02 | 0.07, 0.02, 0.16 |
| llamaguard|smoothllm | 0.14 | 0.49 | -0.02 | 0.36, -0.04, 0.11 |
| llamaguard|token_anomaly | 0.29 | 0.96 | 0.13 | 0.31, 0.22, 0.36 |
| ppl_filter|probe | 0.25 | 1.59 | 0.17 | 0.35, 0.12, 0.29 |
| ppl_filter|probe_b | 0.25 | 1.14 | 0.13 | 0.28, 0.12, 0.34 |
| ppl_filter|refusal_prime | 0.22 | 0.77 | 0.15 | 0.22, 0.11, 0.33 |
| ppl_filter|smoothllm | 0.21 | 0.79 | 0.09 | 0.38, 0.14, 0.10 |
| ppl_filter|token_anomaly | 0.29 | 1.05 | 0.17 | 0.21, 0.33, 0.32 |
| probe|probe_b | 0.22 | 1.85 | 0.13 | 0.14, 0.19, 0.34 |
| probe|refusal_prime | 0.03 | 0.21 | -0.03 | -0.12, 0.11, 0.10 |
| probe|smoothllm | 0.22 | 1.50 | 0.13 | 0.37, 0.07, 0.21 |
| probe|token_anomaly | 0.12 | 0.76 | 0.00 | 0.09, -0.07, 0.32 |
| probe_b|refusal_prime | 0.15 | 0.71 | 0.07 | 0.18, 0.23, 0.05 |
| probe_b|smoothllm | 0.19 | 0.97 | 0.06 | 0.25, 0.07, 0.26 |
| probe_b|token_anomaly | 0.33 | 1.57 | 0.19 | 0.33, 0.27, 0.38 |
| refusal_prime|smoothllm | 0.12 | 0.45 | 0.04 | 0.09, 0.29, -0.03 |
| refusal_prime|token_anomaly | 0.24 | 0.90 | 0.17 | 0.15, 0.39, 0.20 |
| smoothllm|token_anomaly | 0.23 | 0.92 | 0.11 | 0.13, 0.13, 0.42 |

## E2 prompt-level phi, seed 0 (same prompts)

| pair | mean over sources |
|---|---|
| llamaguard|ppl_filter | 0.07 |
| llamaguard|probe | 0.08 |
| llamaguard|probe_b | 0.07 |
| llamaguard|refusal_prime | 0.02 |
| llamaguard|smoothllm | 0.01 |
| llamaguard|token_anomaly | 0.08 |
| ppl_filter|probe | 0.95 |
| ppl_filter|probe_b | 1.00 |
| ppl_filter|refusal_prime | 0.65 |
| ppl_filter|smoothllm | 0.44 |
| ppl_filter|token_anomaly | 0.98 |
| probe|probe_b | 0.95 |
| probe|refusal_prime | 0.65 |
| probe|smoothllm | 0.41 |
| probe|token_anomaly | 0.93 |
| probe_b|refusal_prime | 0.65 |
| probe_b|smoothllm | 0.44 |
| probe_b|token_anomaly | 0.98 |
| refusal_prime|smoothllm | 0.43 |
| refusal_prime|token_anomaly | 0.64 |
| smoothllm|token_anomaly | 0.43 |

## How much is behaviour difficulty? (random-intercept logistic model)

Behaviour SD 1.45, ICC 0.39 (2400 observations). Predicted phi assumes the pair fails independently given the behaviour.

| pair | observed phi | predicted by difficulty alone | excess |
|---|---|---|---|
| llamaguard|ppl_filter | 0.22 | 0.24 | -0.02 |
| llamaguard|probe | 0.16 | 0.24 | -0.08 |
| llamaguard|probe_b | 0.25 | 0.24 | 0.01 |
| llamaguard|refusal_prime | 0.08 | 0.19 | -0.11 |
| llamaguard|smoothllm | 0.14 | 0.23 | -0.09 |
| llamaguard|token_anomaly | 0.29 | 0.24 | 0.05 |
| ppl_filter|probe | 0.25 | 0.25 | 0.01 |
| ppl_filter|probe_b | 0.25 | 0.25 | 0.00 |
| ppl_filter|refusal_prime | 0.22 | 0.19 | 0.03 |
| ppl_filter|smoothllm | 0.21 | 0.24 | -0.04 |
| ppl_filter|token_anomaly | 0.29 | 0.25 | 0.03 |
| probe|probe_b | 0.22 | 0.24 | -0.02 |
| probe|refusal_prime | 0.03 | 0.19 | -0.15 |
| probe|smoothllm | 0.22 | 0.24 | -0.02 |
| probe|token_anomaly | 0.12 | 0.25 | -0.13 |
| probe_b|refusal_prime | 0.15 | 0.19 | -0.03 |
| probe_b|smoothllm | 0.19 | 0.24 | -0.04 |
| probe_b|token_anomaly | 0.33 | 0.25 | 0.08 |
| refusal_prime|smoothllm | 0.12 | 0.19 | -0.07 |
| refusal_prime|token_anomaly | 0.24 | 0.19 | 0.06 |
| smoothllm|token_anomaly | 0.23 | 0.24 | -0.01 |

## Seed 0: diversity, strategies, objective, grader parsing

Non-pairwise diversity over the layers (Kuncheva & Whitaker 2003): theta 0.053, entropy 0.353, kw_variance 0.106, kappa 0.225

| defense | best-of-strategies ASR | attacks pooled |
|---|---|---|
| llamaguard | 0.25 | adaptive, pair, static_gcg, static_plain |
| ppl_filter | 0.71 | adaptive, pair, static_gcg, static_plain |
| probe | 0.77 | adaptive, pair, static_gcg, static_plain |
| probe_b | 0.78 | adaptive, pair, static_gcg, static_plain |
| refusal_prime | 0.80 | adaptive, pair, static_gcg, static_plain |
| smoothllm | 0.66 | adaptive, pair, static_gcg, static_plain |
| token_anomaly | 0.79 | adaptive, pair, static_gcg, static_plain |
| undefended | 0.79 | adaptive, pair, static_gcg, static_plain |
| stack | 0.03 | adaptive, static_gcg, static_plain |
| stack_ppl_filter+token_anomaly | 0.71 | adaptive, static_gcg, static_plain |
| stack_probe+probe_b | 0.78 | adaptive, static_gcg, static_plain |
| stack_probe+refusal_prime | 0.76 | adaptive, static_gcg, static_plain |
| stack_token_anomaly+smoothllm | 0.61 | adaptive, static_gcg, static_plain |

| defense | AUROC of in-loop logprob for grader breach |
|---|---|
| llamaguard | 0.67 |
| ppl_filter | 0.66 |
| probe | 0.55 |
| probe_b | 0.63 |
| refusal_prime | 0.51 |
| smoothllm | 0.68 |
| token_anomaly | 0.55 |
| undefended | 0.62 |

| defense | judged | labels with an unparsed vote |
|---|---|---|
| llamaguard | 100 | 0 |
| ppl_filter | 300 | 7 |
| probe | 289 | 10 |
| probe_b | 379 | 12 |
| refusal_prime | 400 | 8 |
| smoothllm | 400 | 8 |
| token_anomaly | 400 | 15 |
| undefended | 400 | 15 |
| stack | 5 | 0 |
| stack_ppl_filter+token_anomaly | 200 | 9 |
| stack_probe+probe_b | 189 | 10 |
| stack_probe+refusal_prime | 189 | 9 |
| stack_token_anomaly+smoothllm | 300 | 9 |

## Seed 1: diversity, strategies, objective, grader parsing

Non-pairwise diversity over the layers (Kuncheva & Whitaker 2003): theta 0.034, entropy 0.370, kw_variance 0.111, kappa 0.109

| defense | best-of-strategies ASR | attacks pooled |
|---|---|---|
| llamaguard | 0.19 | adaptive, pair, static_gcg, static_plain |
| ppl_filter | 0.75 | adaptive, pair, static_gcg, static_plain |
| probe | 0.75 | adaptive, pair, static_gcg, static_plain |
| probe_b | 0.82 | adaptive, pair, static_gcg, static_plain |
| refusal_prime | 0.76 | adaptive, pair, static_gcg, static_plain |
| smoothllm | 0.74 | adaptive, pair, static_gcg, static_plain |
| token_anomaly | 0.82 | adaptive, pair, static_gcg, static_plain |
| undefended | 0.77 | adaptive, pair, static_gcg, static_plain |
| stack | 0.02 | adaptive, static_gcg, static_plain |
| stack_ppl_filter+token_anomaly | 0.76 | adaptive, static_gcg, static_plain |
| stack_probe+probe_b | 0.81 | adaptive, static_gcg, static_plain |
| stack_probe+refusal_prime | 0.81 | adaptive, static_gcg, static_plain |
| stack_token_anomaly+smoothllm | 0.72 | adaptive, static_gcg, static_plain |

| defense | AUROC of in-loop logprob for grader breach |
|---|---|
| llamaguard | - |
| ppl_filter | 0.58 |
| probe | 0.60 |
| probe_b | 0.56 |
| refusal_prime | 0.55 |
| smoothllm | 0.55 |
| token_anomaly | 0.61 |
| undefended | 0.61 |

| defense | judged | labels with an unparsed vote |
|---|---|---|
| llamaguard | 99 | 1 |
| ppl_filter | 300 | 9 |
| probe | 289 | 10 |
| probe_b | 379 | 12 |
| refusal_prime | 400 | 12 |
| smoothllm | 400 | 12 |
| token_anomaly | 400 | 14 |
| undefended | 400 | 13 |
| stack | 4 | 0 |
| stack_ppl_filter+token_anomaly | 200 | 12 |
| stack_probe+probe_b | 189 | 13 |
| stack_probe+refusal_prime | 189 | 9 |
| stack_token_anomaly+smoothllm | 300 | 8 |

## Seed 2: diversity, strategies, objective, grader parsing

Non-pairwise diversity over the layers (Kuncheva & Whitaker 2003): theta 0.058, entropy 0.343, kw_variance 0.104, kappa 0.253

| defense | best-of-strategies ASR | attacks pooled |
|---|---|---|
| llamaguard | 0.20 | adaptive, pair, static_gcg, static_plain |
| ppl_filter | 0.78 | adaptive, pair, static_gcg, static_plain |
| probe | 0.80 | adaptive, pair, static_gcg, static_plain |
| probe_b | 0.78 | adaptive, pair, static_gcg, static_plain |
| refusal_prime | 0.76 | adaptive, pair, static_gcg, static_plain |
| smoothllm | 0.72 | adaptive, pair, static_gcg, static_plain |
| token_anomaly | 0.79 | adaptive, pair, static_gcg, static_plain |
| undefended | 0.79 | adaptive, pair, static_gcg, static_plain |
| stack | 0.02 | adaptive, static_gcg, static_plain |
| stack_ppl_filter+token_anomaly | 0.71 | adaptive, static_gcg, static_plain |
| stack_probe+probe_b | 0.78 | adaptive, static_gcg, static_plain |
| stack_probe+refusal_prime | 0.71 | adaptive, static_gcg, static_plain |
| stack_token_anomaly+smoothllm | 0.70 | adaptive, static_gcg, static_plain |

| defense | AUROC of in-loop logprob for grader breach |
|---|---|
| llamaguard | 1.00 |
| ppl_filter | 0.59 |
| probe | 0.56 |
| probe_b | 0.57 |
| refusal_prime | 0.52 |
| smoothllm | 0.57 |
| token_anomaly | 0.52 |
| undefended | 0.61 |

| defense | judged | labels with an unparsed vote |
|---|---|---|
| llamaguard | 100 | 1 |
| ppl_filter | 300 | 15 |
| probe | 289 | 11 |
| probe_b | 379 | 12 |
| refusal_prime | 400 | 10 |
| smoothllm | 400 | 9 |
| token_anomaly | 400 | 14 |
| undefended | 400 | 11 |
| stack | 3 | 0 |
| stack_ppl_filter+token_anomaly | 200 | 12 |
| stack_probe+probe_b | 189 | 11 |
| stack_probe+refusal_prime | 189 | 9 |
| stack_token_anomaly+smoothllm | 300 | 8 |

## Held-out member selection, seed 0, all layers (500 splits; test-half intersection residual)

| k | diversity-greedy | one per row | random subset |
|---|---|---|---|
| 1 | 0.06 | 0.06 | 0.20 |
| 2 | 0.01 | 0.02 | 0.08 |
| 3 | 0.00 | 0.01 | 0.04 |
| 4 | 0.00 | 0.00 | 0.02 |

Single best layer alone: 0.06

## Held-out member selection, seed 0, without llamaguard (500 splits; test-half intersection residual)

| k | diversity-greedy | one per row | random subset |
|---|---|---|---|
| 1 | 0.06 | 0.06 | 0.20 |
| 2 | 0.01 | 0.01 | 0.07 |
| 3 | 0.00 | 0.00 | 0.03 |
| 4 | 0.00 | 0.00 | 0.01 |

Single best layer alone: 0.06

## Held-out member selection, seed 1, all layers (500 splits; test-half intersection residual)

| k | diversity-greedy | one per row | random subset |
|---|---|---|---|
| 1 | 0.05 | 0.05 | 0.18 |
| 2 | 0.03 | 0.03 | 0.05 |
| 3 | 0.01 | 0.01 | 0.02 |
| 4 | 0.00 | 0.00 | 0.01 |

Single best layer alone: 0.05

## Held-out member selection, seed 1, without llamaguard (500 splits; test-half intersection residual)

| k | diversity-greedy | one per row | random subset |
|---|---|---|---|
| 1 | 0.05 | 0.05 | 0.18 |
| 2 | 0.03 | 0.03 | 0.05 |
| 3 | 0.02 | 0.02 | 0.02 |
| 4 | 0.01 | 0.01 | 0.01 |

Single best layer alone: 0.05

## Held-out member selection, seed 2, all layers (500 splits; test-half intersection residual)

| k | diversity-greedy | one per row | random subset |
|---|---|---|---|
| 1 | 0.13 | 0.13 | 0.20 |
| 2 | 0.04 | 0.04 | 0.08 |
| 3 | 0.03 | 0.03 | 0.05 |
| 4 | 0.02 | 0.02 | 0.03 |

Single best layer alone: 0.13

## Held-out member selection, seed 2, without llamaguard (500 splits; test-half intersection residual)

| k | diversity-greedy | one per row | random subset |
|---|---|---|---|
| 1 | 0.12 | 0.12 | 0.21 |
| 2 | 0.04 | 0.05 | 0.08 |
| 3 | 0.02 | 0.02 | 0.04 |
| 4 | 0.02 | 0.01 | 0.03 |

Single best layer alone: 0.12

## Benign false-refusal rate, seed 0, jbb set

| configuration | FRR | 95% CI | n |
|---|---|---|---|
| llamaguard | 0.26 | [0.18, 0.35] | 100 |
| ppl_filter | 0.08 | [0.04, 0.15] | 100 |
| probe_b | 0.15 | [0.09, 0.23] | 100 |
| probe | 0.26 | [0.18, 0.35] | 100 |
| refusal_prime | 0.39 | [0.30, 0.49] | 100 |
| smoothllm | 0.20 | [0.13, 0.29] | 100 |
| token_anomaly | 0.08 | [0.04, 0.15] | 100 |
| undefended | 0.08 | [0.04, 0.15] | 100 |
| stack | 0.83 | [0.74, 0.89] | 100 |
| stack_ppl_filter+token_anomaly | 0.08 | [0.04, 0.15] | 100 |
| stack_probe+probe_b | 0.28 | [0.20, 0.37] | 100 |
| stack_probe+refusal_prime | 0.52 | [0.42, 0.62] | 100 |
| stack_token_anomaly+smoothllm | 0.20 | [0.13, 0.29] | 100 |

## Benign false-refusal rate, seed 0, orbench set

| configuration | FRR | 95% CI | n |
|---|---|---|---|
| llamaguard | 0.35 | [0.30, 0.40] | 400 |
| ppl_filter | 0.16 | [0.13, 0.20] | 400 |
| probe_b | 0.17 | [0.13, 0.20] | 400 |
| probe | 0.25 | [0.21, 0.29] | 400 |
| refusal_prime | 0.85 | [0.82, 0.89] | 400 |
| smoothllm | 0.31 | [0.26, 0.35] | 400 |
| token_anomaly | 0.16 | [0.13, 0.20] | 400 |
| undefended | 0.16 | [0.13, 0.20] | 400 |
| stack | 0.99 | [0.98, 1.00] | 400 |

## Benign false-refusal rate, seed 1, jbb set

| configuration | FRR | 95% CI | n |
|---|---|---|---|
| llamaguard | 0.26 | [0.18, 0.35] | 100 |
| ppl_filter | 0.08 | [0.04, 0.15] | 100 |
| probe_b | 0.15 | [0.09, 0.23] | 100 |
| probe | 0.26 | [0.18, 0.35] | 100 |
| refusal_prime | 0.39 | [0.30, 0.49] | 100 |
| smoothllm | 0.20 | [0.13, 0.29] | 100 |
| token_anomaly | 0.08 | [0.04, 0.15] | 100 |
| stack | 0.83 | [0.74, 0.89] | 100 |
| stack_ppl_filter+token_anomaly | 0.08 | [0.04, 0.15] | 100 |
| stack_probe+probe_b | 0.28 | [0.20, 0.37] | 100 |
| stack_probe+refusal_prime | 0.52 | [0.42, 0.62] | 100 |
| stack_token_anomaly+smoothllm | 0.20 | [0.13, 0.29] | 100 |

## Benign false-refusal rate, seed 2, jbb set

| configuration | FRR | 95% CI | n |
|---|---|---|---|
| llamaguard | 0.26 | [0.18, 0.35] | 100 |
| ppl_filter | 0.08 | [0.04, 0.15] | 100 |
| probe_b | 0.15 | [0.09, 0.23] | 100 |
| probe | 0.26 | [0.18, 0.35] | 100 |
| refusal_prime | 0.39 | [0.30, 0.49] | 100 |
| smoothllm | 0.20 | [0.13, 0.29] | 100 |
| token_anomaly | 0.08 | [0.04, 0.15] | 100 |
| stack | 0.83 | [0.74, 0.89] | 100 |
| stack_ppl_filter+token_anomaly | 0.08 | [0.04, 0.15] | 100 |
| stack_probe+probe_b | 0.28 | [0.20, 0.37] | 100 |
| stack_probe+refusal_prime | 0.52 | [0.42, 0.62] | 100 |
| stack_token_anomaly+smoothllm | 0.20 | [0.13, 0.29] | 100 |
