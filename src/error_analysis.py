"""Multi-label error analysis without treating unannotated tags as certain negatives.

Confused pairs count a missed observed label alongside an extra predicted label;
they are co-occurring errors, not mutually exclusive class substitutions.
"""
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .stylistic import extract_features


def sample_errors(split, scores, thresholds, categories, per_category=12, seed=575):
    pred = scores >= thresholds
    rng = np.random.default_rng(seed)
    rows = []
    for category in sorted(set(categories.get(t, "other") for t in split.tags)):
        cols = [j for j, t in enumerate(split.tags) if categories.get(t, "other") == category]
        # Keep the category-balanced sample independent of error severity.
        candidates = np.flatnonzero((pred[:, cols] != split.Y[:, cols]).any(axis=1))
        chosen = rng.choice(candidates, size=min(per_category, len(candidates)), replace=False)
        for i in sorted(chosen):
            false_pos = [split.tags[j] for j in cols if pred[i, j] and not split.Y[i, j]]
            false_neg = [split.tags[j] for j in cols if not pred[i, j] and split.Y[i, j]]
            rows.append({"id": split.ids[i], "artist": split.artists[i], "category": category,
                         "observed_tags": json.dumps([t for j, t in enumerate(split.tags) if split.Y[i, j]]),
                         "predicted_tags": json.dumps([t for j, t in enumerate(split.tags) if pred[i, j]]),
                         "extra_tags": json.dumps(false_pos), "missed_tags": json.dumps(false_neg),
                         "pattern": "", "review_notes": "", "reviewed": False})
    return pd.DataFrame(rows, columns=["id", "artist", "category", "observed_tags", "predicted_tags",
                                      "extra_tags", "missed_tags", "pattern", "review_notes", "reviewed"])


def confused_pairs(Y, pred, tags):
    fn = (np.asarray(Y, bool) & ~np.asarray(pred, bool)).astype(np.int64)
    fp = (~np.asarray(Y, bool) & np.asarray(pred, bool)).astype(np.int64)
    counts = fn.T @ fp
    rows = []
    for i, j in zip(*np.nonzero(counts)):
        if i == j:
            continue
        fraction = float(counts[i, j] / max(1, fn[:, i].sum()))
        marginal = float(fp[:, j].mean())
        rows.append({"missed_tag": tags[i], "extra_tag": tags[j], "count": int(counts[i, j]),
                     "fraction_of_missed": fraction, "extra_tag_fp_rate": marginal,
                     "conditional_lift": fraction / marginal if marginal else np.nan})
    return pd.DataFrame(rows, columns=["missed_tag", "extra_tag", "count", "fraction_of_missed",
                                      "extra_tag_fp_rate", "conditional_lift"]).sort_values(
        ["count", "missed_tag", "extra_tag"], ascending=[False, True, True])


def tag_error_rates(Y, pred, tags, thresholds, categories):
    Y, pred = np.asarray(Y, bool), np.asarray(pred, bool)
    return pd.DataFrame({"tag": tags, "category": [categories.get(t, "other") for t in tags],
                         "n_observed": Y.sum(axis=0), "n_predicted": pred.sum(axis=0),
                         "n_extra": (~Y & pred).sum(axis=0), "n_missed": (Y & ~pred).sum(axis=0),
                         "threshold": thresholds}).sort_values(["n_extra", "tag"], ascending=[False, True])


def song_quality(Y, pred):
    Y, pred = np.asarray(Y, bool), np.asarray(pred, bool)
    tp = (Y & pred).sum(axis=1)
    den = Y.sum(axis=1) + pred.sum(axis=1)
    # Zero-label/zero-prediction rows have undefined sample F1, not perfect F1.
    f1 = np.divide(2 * tp, den, out=np.full(len(Y), np.nan), where=den > 0)
    return pd.DataFrame({"sample_F1": f1, "hamming_loss": (Y != pred).mean(axis=1),
                         "n_observed": Y.sum(axis=1), "n_predicted": pred.sum(axis=1)})


def quality_correlations(features, quality, artists, n_boot=200, seed=575):
    rng = np.random.default_rng(seed)
    codes, unique = pd.factorize(pd.Series(artists).str.strip().str.casefold())
    groups = [np.flatnonzero(codes == c) for c in range(len(unique))]
    rows = []
    for feature in ["n_words", "n_lines", "repeated_line_ratio", "chorus_repetition"]:
        x = features[feature].to_numpy()
        for metric in ["sample_F1", "hamming_loss"]:
            y = quality[metric].to_numpy()
            def correlation(idx):
                xx, yy = x[idx], y[idx]
                ok = np.isfinite(xx) & np.isfinite(yy)
                if ok.sum() < 3 or np.ptp(xx[ok]) == 0 or np.ptp(yy[ok]) == 0:
                    return np.nan
                return float(spearmanr(xx[ok], yy[ok]).statistic)
            rho = correlation(np.arange(len(x)))
            draws = [correlation(np.concatenate([groups[i] for i in rng.integers(len(groups), size=len(groups))]))
                     for _ in range(n_boot)]
            draws = np.asarray(draws)
            draws = draws[np.isfinite(draws)]
            lo, hi = np.quantile(draws, [.025, .975]) if len(draws) else (np.nan, np.nan)
            rows.append({"feature": feature, "metric": metric, "spearman_rho": rho,
                         "artist_bootstrap_low": lo, "artist_bootstrap_high": hi,
                         "valid_bootstraps": len(draws), "n_finite_songs": int(np.isfinite(y).sum())})
    return pd.DataFrame(rows)


def length_repetition_groups(features, quality):
    combined = pd.concat([features.reset_index(drop=True), quality.reset_index(drop=True)], axis=1)
    rows = []
    for feature in ["n_words", "repeated_line_ratio"]:
        try:
            groups = pd.qcut(combined[feature], 4, duplicates="drop")
        except ValueError:
            continue
        for label, group in combined.groupby(groups, observed=True):
            rows.append({"feature": feature, "bin": str(label), "n_songs": len(group),
                         "mean_sample_F1": group.sample_F1.mean(),
                         "mean_hamming_loss": group.hamming_loss.mean()})
    return pd.DataFrame(rows)


def explain_tfidf(split, sample, vectorizer, model, max_terms=8):
    """Exact signed term contributions for the selected FP/FN decisions."""
    rows = []
    indices = {sid: i for i, sid in enumerate(split.ids)}
    names = vectorizer.get_feature_names_out()
    X = vectorizer.transform(split.texts)
    for row in sample.itertuples(index=False):
        i = indices[row.id]
        for kind, tag_list in [("extra", row.extra_tags), ("missed", row.missed_tags)]:
            for tag in json.loads(tag_list):
                j = split.tags.index(tag)
                est = model.estimators_[j]
                if not hasattr(est, "coef_"):
                    continue
                nz = X[i].indices
                contributions = X[i].data * est.coef_.ravel()[nz]
                order = np.argsort(-np.abs(contributions))[:max_terms]
                for k in order:
                    rows.append({"id": row.id, "category": row.category, "error_type": kind,
                                 "tag": tag, "term": names[nz[k]],
                                 "contribution": float(contributions[k]),
                                 "intercept": float(est.intercept_[0])})
    return pd.DataFrame(rows, columns=["id", "category", "error_type", "tag", "term", "contribution", "intercept"])
