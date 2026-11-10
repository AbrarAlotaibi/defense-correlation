# v2 analysis: v2_vicuna_nolg (seeds [0, 1, 2], n=100)

## Marginal ASR

| defense | mean ASR | pooled 95% CI | reliability r |
|---|---|---|---|

**Null (undefended seed-pair phi):** needs >= 2 seeds

## E1 behaviour-level phi (separate searches)

| pair | mean phi | disattenuated | partial given undefended | per seed |
|---|---|---|---|---|

## Seed 0: diversity, strategies, objective, grader parsing


| defense | best-of-strategies ASR | attacks pooled |
|---|---|---|
| stack | 0.60 | adaptive, static_gcg, static_plain |

| defense | judged | labels with an unparsed vote |
|---|---|---|
| stack | 111 | 8 |

## Seed 1: diversity, strategies, objective, grader parsing


| defense | best-of-strategies ASR | attacks pooled |
|---|---|---|
| stack | 0.57 | adaptive, static_gcg, static_plain |

| defense | judged | labels with an unparsed vote |
|---|---|---|
| stack | 111 | 4 |

## Seed 2: diversity, strategies, objective, grader parsing


| defense | best-of-strategies ASR | attacks pooled |
|---|---|---|
| stack | 0.55 | adaptive, static_gcg, static_plain |

| defense | judged | labels with an unparsed vote |
|---|---|---|
| stack | 111 | 9 |

## Benign false-refusal rate, seed 0, jbb set

| configuration | FRR | 95% CI | n |
|---|---|---|---|
| stack | 0.82 | [0.73, 0.88] | 100 |

## Benign false-refusal rate, seed 1, jbb set

| configuration | FRR | 95% CI | n |
|---|---|---|---|
| stack | 0.82 | [0.73, 0.88] | 100 |

## Benign false-refusal rate, seed 2, jbb set

| configuration | FRR | 95% CI | n |
|---|---|---|---|
| stack | 0.82 | [0.73, 0.88] | 100 |
