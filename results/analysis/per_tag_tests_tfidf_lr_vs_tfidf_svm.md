# Per-tag paired tests: tfidf_lr minus tfidf_svm (validation)

Paired artist bootstrap, 1000 resamples; per-tag AP differences corrected with Benjamini-Hochberg (FDR 0.05). Thresholds tuned on validation.

- **Macro-F1:** +0.004 [-0.000, +0.009], p = 0.056
- **MAP:** +0.009 [+0.007, +0.012], p = 0.000
- **Tags with a reliable AP difference:** 38 of 112 (tfidf_lr better on 38, tfidf_svm better on 0)

## By category

| Category | Tags | Significant | Mean AP diff |
|---|---|---|---|
| instrumentation/production | 7 | 4 | +0.013 |
| genre | 71 | 25 | +0.010 |
| theme | 4 | 2 | +0.009 |
| mood/emotion | 13 | 5 | +0.008 |
| era | 10 | 2 | +0.007 |
| other | 7 | 0 | +0.001 |

## Largest differences

| Tag | Category | AP diff | 95% CI | q (BH) |
|---|---|---|---|---|
| pop punk | genre | +0.064 | [+0.007, +0.108] | 0.052 |
| folk rock | genre | +0.048 | [+0.006, +0.087] | 0.000 |
| emo | genre | +0.045 | [+0.023, +0.079] | 0.000 |
| melodic death metal | genre | +0.044 | [+0.006, +0.096] | 0.024 |
| folk | genre | +0.041 | [+0.016, +0.069] | 0.000 |
| heavy metal | genre | -0.026 | [-0.055, +0.014] | 0.398 |
| power metal | genre | -0.031 | [-0.084, +0.004] | 0.167 |
| trap | genre | -0.032 | [-0.090, +0.023] | 0.412 |
| symphonic metal | genre | -0.033 | [-0.086, +0.014] | 0.398 |
| thrash metal | genre | -0.038 | [-0.066, +0.001] | 0.134 |
