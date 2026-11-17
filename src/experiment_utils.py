"""Alignment, provenance, and train-only fitting used by the experiment pipelines."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from scipy.sparse import issparse
from joblib import parallel_backend

from .evaluate import evaluate, tune_thresholds


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    def convert(obj):
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, Path):
            return str(obj)
        raise TypeError(type(obj).__name__)
    # JSON null, rather than non-standard NaN, for unavailable quantities.
    def finite(obj):
        if isinstance(obj, dict):
            return {str(k): finite(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple, np.ndarray)):
            return [finite(v) for v in obj]
        if isinstance(obj, (float, np.floating)) and not np.isfinite(obj):
            return None
        return obj
    Path(path).write_text(json.dumps(finite(value), indent=2, default=convert) + "\n")


def ensure_train_val(train, val):
    if not train.ids or not val.ids:
        raise ValueError("Training and validation splits must both contain songs")
    if train.tags != val.tags:
        raise ValueError("Tag column order differs")
    if len(set(train.ids)) != len(train.ids) or len(set(val.ids)) != len(val.ids):
        raise ValueError("Duplicate song IDs")
    if set(train.ids) & set(val.ids):
        raise ValueError("Song overlap between training and validation")
    if {a.strip().casefold() for a in train.artists} & {a.strip().casefold() for a in val.artists}:
        raise ValueError("Artist leakage between training and validation")


def fit_lr(X, Y, seed=575, regularization=1.0):
    if issparse(X):
        # Canonicalize before workers share the matrix: scipy reductions may
        # otherwise try to sort read-only joblib memmaps during sklearn checks.
        X.sum_duplicates()
        X.sort_indices()
    with parallel_backend("threading", n_jobs=4):
        fitted = OneVsRestClassifier(LogisticRegression(
            C=regularization, solver="liblinear", max_iter=1000, random_state=seed), n_jobs=4).fit(X, Y)
    if any(np.max(getattr(est, "n_iter_", [0])) >= 1000 for est in fitted.estimators_):
        raise RuntimeError("Logistic regression reached its iteration limit; do not report this as a converged run")
    return fitted


def save_scores(path, split, scores, thresholds, **extra):
    np.savez_compressed(path, ids=np.asarray(split.ids, dtype=str),
                        tags=np.asarray(split.tags, dtype=str), scores=np.asarray(scores, np.float32),
                        thresholds=np.asarray(thresholds, np.float32), **extra)


def load_scores(path, split):
    with np.load(path, allow_pickle=False) as z:
        required = {"ids", "tags", "scores", "thresholds"}
        if not required.issubset(z.files):
            raise ValueError("Scores need explicit ids, tags, scores, thresholds")
        ids, tags = z["ids"].tolist(), z["tags"].tolist()
        if len(set(ids)) != len(ids) or tags != split.tags or set(ids) != set(split.ids):
            raise ValueError("Score artifact IDs or tag order do not match the requested split")
        score, thresholds = z["scores"], z["thresholds"]
        if score.shape != (len(ids), len(tags)) or thresholds.shape != (len(tags),):
            raise ValueError("Score artifact shape mismatch")
        if not np.isfinite(score).all() or not np.isfinite(thresholds).all():
            raise ValueError("Non-finite scores or thresholds")
        order = {sid: i for i, sid in enumerate(ids)}
        return score[[order[sid] for sid in split.ids]], thresholds


def report_model(out, name, split, scores, categories, fit_seconds=0):
    thresholds = tune_thresholds(split.Y, scores)
    save_scores(out / f"{name}_val_scores.npz", split, scores, thresholds)
    rows = []
    for variant, th in [("fixed_0.5", 0.5), ("tuned_on_val", thresholds)]:
        result = evaluate(split.Y, scores, split.ids, split.tags, categories, thresholds=th)
        result.per_tag.to_csv(out / f"{name}_{variant}_per_tag.csv", index=False)
        result.per_category.to_csv(out / f"{name}_{variant}_per_category.csv", index=False)
        rows.append({"model": name, "thresholds": variant, "fit_s": fit_seconds, **result.summary})
    return rows, thresholds


def subset_split(split, ids):
    from .data import Split
    lookup = {sid: i for i, sid in enumerate(split.ids)}
    if len(ids) != len(set(ids)) or any(sid not in lookup for sid in ids):
        raise ValueError("Subset IDs must be unique and present in this split")
    idx = [lookup[sid] for sid in ids]
    return Split(list(ids), [split.artists[i] for i in idx], [split.texts[i] for i in idx],
                 split.Y[idx], split.tags)


def coefficient_importance(model, names, tags, categories):
    rows = []
    for tag, est in zip(tags, model.estimators_):
        if not hasattr(est, "coef_"):
            continue  # constant labels have no fitted coefficient vector
        for feature, coefficient in zip(names, est.coef_.ravel()):
            rows.append({"tag": tag, "category": categories.get(tag, "other"),
                         "feature": feature, "coefficient": float(coefficient),
                         "abs_coefficient": float(abs(coefficient))})
    return pd.DataFrame(rows)
