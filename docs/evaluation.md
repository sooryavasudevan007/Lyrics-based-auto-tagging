# Shared evaluation harness (Soorya, Objective 2 PI5)

Every model is scored with `src/evaluate.py` so results are comparable
(README Section 3.3). Interface: a score matrix in, a metrics table out.

```python
from src.data import load_split, load_categories
from src.evaluate import evaluate, tune_thresholds, bootstrap_ci

val = load_split("val")
S = model.predict_proba(X_val)            # (songs x K), higher = more likely
res = evaluate(val.Y, S, val.ids, val.tags, load_categories(), thresholds=0.5)
res.summary        # micro/macro P, R, F1, Hamming loss, MAP, P@10, R-precision, ROC-AUC
res.per_tag        # one row per tag
res.per_category   # one row per category, plus the "mood+theme" group used for H1

th = tune_thresholds(val.Y, S)            # README 3.4 (per tag; global for < 20 positives)
res_tuned = evaluate(val.Y, S, val.ids, val.tags, load_categories(), thresholds=th)
ci = bootstrap_ci(val.Y, S, S >= 0.5, val.ids, val.artists, n_boot=1000)
```

Checked against scikit-learn: micro/macro F1, Hamming loss and MAP match
exactly; P@10 and R-precision match hand calculation; constant scores give
AP equal to tag prevalence and ROC-AUC 0.5.

Rules: tune on `val`, never on `test`. The test split is scored once, at the
end, after all choices are frozen and committed.

## Baselines

```bash
python scripts/run_baselines.py                    # ~10-30 min on Music4All-EN
python scripts/run_baselines.py --models tfidf_lr  # one model
python scripts/run_baselines.py --bootstrap 1000   # adds 95% CIs (slower)
```

Outputs in `results/baselines/`: `summary_val.md` / `.csv` (one row per model
and threshold setting), `per_category_val.csv`, and `<model>_<thresholds>_per_tag.csv`.
