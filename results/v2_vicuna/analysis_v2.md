# v2 analysis: v2_vicuna (seeds [0, 1, 2], n=100)

## Marginal ASR

| defense | mean ASR | pooled 95% CI | reliability r |
|---|---|---|---|
| llamaguard | 0.03 | [0.02, 0.06] | 0.54 |
| ppl_filter | 0.72 | [0.67, 0.77] | 0.58 |
| probe | 0.73 | [0.68, 0.78] | 0.63 |
| probe_b | 0.75 | [0.70, 0.80] | 0.65 |
| refusal_prime | 0.75 | [0.70, 0.80] | 0.61 |
| smoothllm | 0.65 | [0.59, 0.70] | 0.42 |
| token_anomaly | 0.74 | [0.69, 0.79] | 0.65 |
| undefended | 0.74 | [0.69, 0.79] | 0.62 |

**Null (undefended seed-pair phi):** 0.62

## E1 behaviour-level phi (separate searches)

| pair | mean phi | disattenuated | partial given undefended | per seed |
|---|---|---|---|---|
| llamaguard|ppl_filter | 0.11 | 0.20 | 0.09 | 0.12, 0.11, 0.10 |
| llamaguard|probe | 0.06 | 0.11 | 0.01 | 0.10, -0.02, 0.11 |
| llamaguard|probe_b | 0.06 | 0.10 | 0.00 | -0.03, 0.09, 0.10 |
| llamaguard|refusal_prime | 0.06 | 0.10 | 0.02 | 0.09, 0.10, -0.02 |
| llamaguard|smoothllm | 0.05 | 0.10 | 0.01 | 0.02, 0.12, 0.01 |
| llamaguard|token_anomaly | 0.01 | 0.02 | -0.05 | -0.03, -0.03, 0.10 |
| ppl_filter|probe | 0.58 | 0.97 | 0.07 | 0.54, 0.67, 0.53 |
| ppl_filter|probe_b | 0.58 | 0.95 | 0.13 | 0.65, 0.55, 0.55 |
| ppl_filter|refusal_prime | 0.63 | 1.05 | 0.21 | 0.70, 0.62, 0.57 |
| ppl_filter|smoothllm | 0.53 | 1.08 | 0.12 | 0.55, 0.69, 0.35 |
| ppl_filter|token_anomaly | 0.60 | 0.99 | 0.17 | 0.55, 0.69, 0.57 |
| probe|probe_b | 0.65 | 1.02 | 0.27 | 0.64, 0.58, 0.72 |
| probe|refusal_prime | 0.63 | 1.01 | 0.22 | 0.67, 0.65, 0.57 |
| probe|smoothllm | 0.49 | 0.95 | 0.00 | 0.53, 0.53, 0.40 |
| probe|token_anomaly | 0.63 | 0.99 | 0.22 | 0.59, 0.62, 0.69 |
| probe_b|refusal_prime | 0.63 | 1.00 | 0.26 | 0.68, 0.57, 0.65 |
| probe_b|smoothllm | 0.55 | 1.06 | 0.20 | 0.62, 0.52, 0.52 |
| probe_b|token_anomaly | 0.61 | 0.94 | 0.23 | 0.64, 0.47, 0.71 |
| refusal_prime|smoothllm | 0.53 | 1.04 | 0.14 | 0.49, 0.62, 0.48 |
| refusal_prime|token_anomaly | 0.60 | 0.95 | 0.20 | 0.68, 0.55, 0.57 |
| smoothllm|token_anomaly | 0.53 | 1.00 | 0.13 | 0.44, 0.59, 0.54 |

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

## E3 assembled stack

| seed | stack ASR | 95% CI | intersection | strongest layer | McNemar p (vs strongest) |
|---|---|---|---|---|---|
| 0 | 0.03 | [0.01, 0.08] | 0.01 | llamaguard 0.03 | 1.000 |
| 1 | 0.02 | [0.01, 0.07] | 0.02 | llamaguard 0.03 | 1.000 |
| 2 | 0.02 | [0.01, 0.07] | 0.02 | llamaguard 0.03 | 1.000 |

## How much is behaviour difficulty? (random-intercept logistic model)

Behaviour SD 2.73, ICC 0.69 (2400 observations). Predicted phi assumes the pair fails independently given the behaviour.

| pair | observed phi | predicted by difficulty alone | excess |
|---|---|---|---|
| llamaguard|ppl_filter | 0.11 | 0.15 | -0.04 |
| llamaguard|probe | 0.06 | 0.14 | -0.08 |
| llamaguard|probe_b | 0.06 | 0.14 | -0.08 |
| llamaguard|refusal_prime | 0.06 | 0.14 | -0.08 |
| llamaguard|smoothllm | 0.05 | 0.18 | -0.13 |
| llamaguard|token_anomaly | 0.01 | 0.14 | -0.13 |
| ppl_filter|probe | 0.58 | 0.49 | 0.09 |
| ppl_filter|probe_b | 0.58 | 0.49 | 0.09 |
| ppl_filter|refusal_prime | 0.63 | 0.49 | 0.14 |
| ppl_filter|smoothllm | 0.53 | 0.49 | 0.04 |
| ppl_filter|token_anomaly | 0.60 | 0.49 | 0.11 |
| probe|probe_b | 0.65 | 0.49 | 0.16 |
| probe|refusal_prime | 0.63 | 0.49 | 0.14 |
| probe|smoothllm | 0.49 | 0.49 | 0.00 |
| probe|token_anomaly | 0.63 | 0.49 | 0.14 |
| probe_b|refusal_prime | 0.63 | 0.49 | 0.14 |
| probe_b|smoothllm | 0.55 | 0.48 | 0.07 |
| probe_b|token_anomaly | 0.61 | 0.49 | 0.12 |
| refusal_prime|smoothllm | 0.53 | 0.48 | 0.05 |
| refusal_prime|token_anomaly | 0.60 | 0.49 | 0.11 |
| smoothllm|token_anomaly | 0.53 | 0.48 | 0.04 |

## Seed 0: diversity, strategies, objective, grader parsing

Non-pairwise diversity over the layers (Kuncheva & Whitaker 2003): theta 0.099, entropy 0.447, kw_variance 0.138, kappa 0.321

| defense | best-of-strategies ASR | attacks pooled |
|---|---|---|
| llamaguard | 0.03 | adaptive, static_gcg, static_plain |
| ppl_filter | 0.68 | adaptive, static_gcg, static_plain |
| probe | 0.77 | adaptive, static_gcg, static_plain |
| probe_b | 0.75 | adaptive, static_gcg, static_plain |
| refusal_prime | 0.80 | adaptive, static_gcg, static_plain |
| smoothllm | 0.62 | adaptive, static_gcg, static_plain |
| token_anomaly | 0.78 | adaptive, static_gcg, static_plain |
| undefended | 0.78 | adaptive, static_gcg, static_plain |
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
| llamaguard | 8 | 0 |
| ppl_filter | 200 | 7 |
| probe | 189 | 10 |
| probe_b | 279 | 11 |
| refusal_prime | 300 | 8 |
| smoothllm | 300 | 8 |
| token_anomaly | 300 | 14 |
| undefended | 300 | 14 |
| stack | 5 | 0 |
| stack_ppl_filter+token_anomaly | 200 | 9 |
| stack_probe+probe_b | 189 | 10 |
| stack_probe+refusal_prime | 189 | 9 |
| stack_token_anomaly+smoothllm | 300 | 9 |

## Seed 1: diversity, strategies, objective, grader parsing

Non-pairwise diversity over the layers (Kuncheva & Whitaker 2003): theta 0.096, entropy 0.430, kw_variance 0.136, kappa 0.318

| defense | best-of-strategies ASR | attacks pooled |
|---|---|---|
| llamaguard | 0.03 | adaptive, static_gcg, static_plain |
| ppl_filter | 0.73 | adaptive, static_gcg, static_plain |
| probe | 0.73 | adaptive, static_gcg, static_plain |
| probe_b | 0.80 | adaptive, static_gcg, static_plain |
| refusal_prime | 0.76 | adaptive, static_gcg, static_plain |
| smoothllm | 0.69 | adaptive, static_gcg, static_plain |
| token_anomaly | 0.80 | adaptive, static_gcg, static_plain |
| undefended | 0.77 | adaptive, static_gcg, static_plain |
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
| llamaguard | 7 | 0 |
| ppl_filter | 200 | 9 |
| probe | 189 | 10 |
| probe_b | 279 | 10 |
| refusal_prime | 300 | 12 |
| smoothllm | 300 | 9 |
| token_anomaly | 300 | 12 |
| undefended | 300 | 11 |
| stack | 4 | 0 |
| stack_ppl_filter+token_anomaly | 200 | 12 |
| stack_probe+probe_b | 189 | 13 |
| stack_probe+refusal_prime | 189 | 9 |
| stack_token_anomaly+smoothllm | 300 | 8 |

## Seed 2: diversity, strategies, objective, grader parsing

Non-pairwise diversity over the layers (Kuncheva & Whitaker 2003): theta 0.094, entropy 0.453, kw_variance 0.140, kappa 0.302

| defense | best-of-strategies ASR | attacks pooled |
|---|---|---|
| llamaguard | 0.03 | adaptive, static_gcg, static_plain |
| ppl_filter | 0.76 | adaptive, static_gcg, static_plain |
| probe | 0.78 | adaptive, static_gcg, static_plain |
| probe_b | 0.77 | adaptive, static_gcg, static_plain |
| refusal_prime | 0.74 | adaptive, static_gcg, static_plain |
| smoothllm | 0.68 | adaptive, static_gcg, static_plain |
| token_anomaly | 0.79 | adaptive, static_gcg, static_plain |
| undefended | 0.77 | adaptive, static_gcg, static_plain |
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
| llamaguard | 8 | 0 |
| ppl_filter | 200 | 14 |
| probe | 189 | 11 |
| probe_b | 279 | 12 |
| refusal_prime | 300 | 9 |
| smoothllm | 300 | 8 |
| token_anomaly | 300 | 13 |
| undefended | 300 | 9 |
| stack | 3 | 0 |
| stack_ppl_filter+token_anomaly | 200 | 12 |
| stack_probe+probe_b | 189 | 11 |
| stack_probe+refusal_prime | 189 | 9 |
| stack_token_anomaly+smoothllm | 300 | 8 |

## E3 jointly attacked subset: ppl_filter, token_anomaly

| seed | joint ASR | 95% CI | intersection of members | McNemar p (joint vs intersection) |
|---|---|---|---|---|
| 0 | 0.71 | [0.61, 0.79] | 0.61 | 0.021 |
| 1 | 0.75 | [0.66, 0.82] | 0.68 | 0.167 |
| 2 | 0.71 | [0.61, 0.79] | 0.67 | 0.541 |

## E3 jointly attacked subset: probe, probe_b

| seed | joint ASR | 95% CI | intersection of members | McNemar p (joint vs intersection) |
|---|---|---|---|---|
| 0 | 0.73 | [0.64, 0.81] | 0.67 | 0.210 |
| 1 | 0.77 | [0.68, 0.84] | 0.67 | 0.021 |
| 2 | 0.74 | [0.65, 0.82] | 0.68 | 0.070 |

## E3 jointly attacked subset: probe, refusal_prime

| seed | joint ASR | 95% CI | intersection of members | McNemar p (joint vs intersection) |
|---|---|---|---|---|
| 0 | 0.75 | [0.66, 0.82] | 0.71 | 0.424 |
| 1 | 0.81 | [0.72, 0.87] | 0.66 | 0.001 |
| 2 | 0.71 | [0.61, 0.79] | 0.64 | 0.189 |

## E3 jointly attacked subset: token_anomaly, smoothllm

| seed | joint ASR | 95% CI | intersection of members | McNemar p (joint vs intersection) |
|---|---|---|---|---|
| 0 | 0.59 | [0.49, 0.68] | 0.54 | 0.500 |
| 1 | 0.70 | [0.60, 0.78] | 0.63 | 0.248 |
| 2 | 0.68 | [0.58, 0.76] | 0.60 | 0.134 |

## Held-out member selection, seed 0, all layers (500 splits; test-half intersection residual)

| k | diversity-greedy | one per row | random subset |
|---|---|---|---|
| 1 | 0.03 | 0.03 | 0.62 |
| 2 | 0.03 | 0.02 | 0.46 |
| 3 | 0.02 | 0.02 | 0.34 |
| 4 | 0.02 | 0.02 | 0.24 |

Single best layer alone: 0.03

## Held-out member selection, seed 0, without llamaguard (500 splits; test-half intersection residual)

| k | diversity-greedy | one per row | random subset |
|---|---|---|---|
| 1 | 0.63 | 0.63 | 0.72 |
| 2 | 0.56 | 0.55 | 0.63 |
| 3 | 0.51 | 0.53 | 0.58 |
| 4 | 0.49 | 0.53 | 0.54 |

Single best layer alone: 0.63

## Held-out member selection, seed 1, all layers (500 splits; test-half intersection residual)

| k | diversity-greedy | one per row | random subset |
|---|---|---|---|
| 1 | 0.03 | 0.03 | 0.63 |
| 2 | 0.03 | 0.03 | 0.47 |
| 3 | 0.03 | 0.03 | 0.35 |
| 4 | 0.02 | 0.02 | 0.25 |

Single best layer alone: 0.03

## Held-out member selection, seed 1, without llamaguard (500 splits; test-half intersection residual)

| k | diversity-greedy | one per row | random subset |
|---|---|---|---|
| 1 | 0.71 | 0.71 | 0.73 |
| 2 | 0.64 | 0.65 | 0.65 |
| 3 | 0.60 | 0.59 | 0.60 |
| 4 | 0.57 | 0.54 | 0.55 |

Single best layer alone: 0.71

## Held-out member selection, seed 2, all layers (500 splits; test-half intersection residual)

| k | diversity-greedy | one per row | random subset |
|---|---|---|---|
| 1 | 0.03 | 0.03 | 0.62 |
| 2 | 0.02 | 0.02 | 0.46 |
| 3 | 0.02 | 0.02 | 0.34 |
| 4 | 0.02 | 0.02 | 0.24 |

Single best layer alone: 0.03

## Held-out member selection, seed 2, without llamaguard (500 splits; test-half intersection residual)

| k | diversity-greedy | one per row | random subset |
|---|---|---|---|
| 1 | 0.68 | 0.68 | 0.72 |
| 2 | 0.59 | 0.60 | 0.63 |
| 3 | 0.53 | 0.54 | 0.57 |
| 4 | 0.50 | 0.49 | 0.53 |

Single best layer alone: 0.68

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
