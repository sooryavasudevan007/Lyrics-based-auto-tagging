# Leakage check: artist-disjoint vs. song-level split (validation)

Same TF-IDF + LR model, vocabulary and hyperparameters; only the split differs. Thresholds tuned on each validation split.

| Split | Val artists also in train | Macro-F1 | MAP | P@10 | ROC-AUC |
|---|---|---|---|---|---|
| artist | 0 | 0.165 [0.156, 0.169] | 0.122 [0.117, 0.131] | 0.274 | 0.759 |
| song | 3,972 | 0.189 [0.182, 0.193] | 0.144 [0.140, 0.151] | 0.388 | 0.778 |

**Song-level minus artist-disjoint:** macro_F1 +0.024, MAP +0.022, ROC-AUC +0.018, P@10 +0.113

## ROC-AUC by category

| Category | Artist | Song | Gap |
|---|---|---|---|
| other | 0.647 | 0.686 | +0.039 |
| era | 0.693 | 0.717 | +0.023 |
| mood/emotion | 0.703 | 0.720 | +0.017 |
| genre | 0.790 | 0.807 | +0.017 |
| instrumentation/production | 0.731 | 0.746 | +0.015 |
| mood+theme | 0.727 | 0.742 | +0.015 |
| theme | 0.806 | 0.813 | +0.008 |
