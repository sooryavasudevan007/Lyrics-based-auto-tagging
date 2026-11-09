"""Shared evaluation harness (README Section 3.3). Every model in the project
is scored with this module so that results are directly comparable.

Interface: a score matrix in, a metrics table out.

    from src.evaluate import evaluate
    res = evaluate(Y_true, S, ids=val.ids, tags=val.tags, categories=load_categories(),
                   thresholds=0.5)
    res.summary        # dict of aggregate metrics
    res.per_tag        # DataFrame, one row per tag
    res.per_category   # DataFrame, one row per category (+ "mood+theme")

S is (songs x K) with higher = more likely. Annotation metrics use
S >= thresholds (a scalar or one value per tag), or explicit 0/1 predictions
passed as `pred`. Retrieval metrics use S directly.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

P_AT_K = 10
GROUPS = {"mood+theme": ("mood/emotion", "theme")}   # README: H1 analysis group


@dataclass
class Result:
    summary: dict
    per_tag: pd.DataFrame
    per_category: pd.DataFrame


# ------------------------------------------------------------------ ranking
def _rank_order(S, ids):
    """Row order per tag: score descending, ties broken by song id (README 3.3)."""
    id_rank = np.argsort(np.argsort(np.asarray(ids, dtype=object)))  # position in sorted ids
    # lexsort sorts by last key first: primary -score, secondary id
    return np.stack([np.lexsort((id_rank, -S[:, j])) for j in range(S.shape[1])], axis=1)


def retrieval_metrics(Y, S, ids, k=P_AT_K, with_auc=True):
    """Per-tag non-interpolated AP, P@k, R-precision, ROC-AUC.

    AP(t) = (1/R_t) * sum_i P(i) * rel(i). With no tied scores this equals
    sklearn's average_precision_score; ties are broken by song id.
    """
    Y = np.asarray(Y, dtype=bool)
    order = _rank_order(S, ids)
    rel = np.take_along_axis(Y, order, axis=0)                 # (n, K) relevance by rank
    hits = np.cumsum(rel, axis=0)
    ranks = np.arange(1, len(Y) + 1)[:, None]
    R = Y.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        ap = (rel * hits / ranks).sum(axis=0) / R
        p_at_k = hits[min(k, len(Y)) - 1] / min(k, len(Y))
        r_prec = np.array([hits[r - 1, j] / r if r > 0 else np.nan for j, r in enumerate(R)])
    auc = np.array([roc_auc_score(Y[:, j], S[:, j]) if with_auc and 0 < R[j] < len(Y) else np.nan
                    for j in range(Y.shape[1])])
    return {"AP": ap, f"P@{k}": p_at_k, "R-precision": r_prec, "ROC-AUC": auc}


# ------------------------------------------------------------------ annotation
def annotation_counts(Y, P):
    Y = np.asarray(Y, dtype=bool); P = np.asarray(P, dtype=bool)
    tp = (Y & P).sum(axis=0); fp = (~Y & P).sum(axis=0); fn = (Y & ~P).sum(axis=0)
    return tp, fp, fn


def _prf(tp, fp, fn):
    with np.errstate(invalid="ignore", divide="ignore"):
        p = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        r = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
        f = np.where(p + r > 0, 2 * p * r / (p + r), 0.0)
    return p, r, f


def annotation_metrics(Y, P):
    tp, fp, fn = annotation_counts(Y, P)
    p, r, f = _prf(tp, fp, fn)
    mp, mr, mf = _prf(tp.sum(), fp.sum(), fn.sum())
    hamming = float((np.asarray(Y, bool) != np.asarray(P, bool)).mean())
    return {"precision": p, "recall": r, "F1": f}, {
        "micro_P": float(mp), "micro_R": float(mr), "micro_F1": float(mf),
        "macro_P": float(p.mean()), "macro_R": float(r.mean()), "macro_F1": float(f.mean()),
        "hamming_loss": hamming}


# ------------------------------------------------------------------ main entry
def evaluate(Y, S, ids, tags, categories=None, thresholds=0.5, pred=None, k=P_AT_K):
    Y = np.asarray(Y)
    S = np.asarray(S, dtype=float)
    assert Y.shape == S.shape == (len(ids), len(tags)), "Y, S, ids, tags do not line up"
    if pred is None:
        pred = S >= np.broadcast_to(np.asarray(thresholds, dtype=float), (len(tags),))
    per_tag_ann, summary = annotation_metrics(Y, pred)
    ret = retrieval_metrics(Y, S, ids, k)

    per_tag = pd.DataFrame({"tag": tags, "n_pos": Y.sum(axis=0), **per_tag_ann, **ret})
    if categories is not None:
        per_tag.insert(1, "category", [categories.get(t, "other") for t in tags])
    summary.update({
        "MAP": float(np.nanmean(ret["AP"])),
        f"P@{k}": float(np.nanmean(ret[f"P@{k}"])),
        "R-precision": float(np.nanmean(ret["R-precision"])),
        "ROC-AUC": float(np.nanmean(ret["ROC-AUC"])),
        "n_songs": len(ids), "K": len(tags)})

    per_cat = pd.DataFrame()
    if categories is not None:
        rows = []
        cats = per_tag["category"]
        groups = {c: [c] for c in sorted(cats.unique())}
        groups.update({g: list(members) for g, members in GROUPS.items()})
        for name, members in groups.items():
            m = cats.isin(members).values
            if m.any():
                rows.append({"category": name, "n_tags": int(m.sum()),
                             "macro_F1": per_tag.loc[m, "F1"].mean(),
                             "MAP": per_tag.loc[m, "AP"].mean(),
                             "ROC-AUC": per_tag.loc[m, "ROC-AUC"].mean(),
                             f"P@{k}": per_tag.loc[m, f"P@{k}"].mean()})
        per_cat = pd.DataFrame(rows)
    return Result(summary, per_tag, per_cat)


# ------------------------------------------------------------------ thresholds
def tune_thresholds(Y_val, S_val, min_pos=20, grid=None):
    """README 3.4: per-tag threshold maximising that tag's validation F1;
    tags with < min_pos validation positives share one global threshold
    chosen to maximise validation macro-F1 over those tags."""
    Y = np.asarray(Y_val, dtype=bool)
    S = np.asarray(S_val, dtype=float)
    grid = np.linspace(0.01, 0.99, 99) if grid is None else np.asarray(grid)
    f1 = np.zeros((len(grid), Y.shape[1]))
    for g, th in enumerate(grid):
        _, _, f1[g] = _prf(*annotation_counts(Y, S >= th))
    th = grid[f1.argmax(axis=0)]
    rare = Y.sum(axis=0) < min_pos
    if rare.any():
        th[rare] = grid[f1[:, rare].mean(axis=1).argmax()]
    return th


# ------------------------------------------------------------------ bootstrap
def bootstrap_ci(Y, S, pred, ids, artists, n_boot=1000, seed=0, alpha=0.05):
    """95% CIs for macro-F1 and MAP, resampling *artists* with replacement
    (README 3.3: respects the artist-disjoint design)."""
    rng = np.random.default_rng(seed)
    Y = np.asarray(Y, bool); P = np.asarray(pred, bool); S = np.asarray(S, float)
    ids = np.asarray(ids, dtype=object)
    codes, uniq = pd.factorize(pd.Series(artists).str.lower())
    rows_of = [np.flatnonzero(codes == a) for a in range(len(uniq))]
    stats = []
    for _ in range(n_boot):
        pick = np.concatenate([rows_of[a] for a in rng.integers(0, len(uniq), len(uniq))])
        _, s = annotation_metrics(Y[pick], P[pick])
        # duplicated songs get unique ids so tie-breaking stays well defined
        boot_ids = np.char.add(ids[pick].astype(str), np.arange(len(pick)).astype(str))
        ap = retrieval_metrics(Y[pick], S[pick], boot_ids, with_auc=False)["AP"]
        stats.append((s["macro_F1"], np.nanmean(ap)))
    stats = np.array(stats)
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)], axis=0)
    return {"macro_F1_CI": (float(lo[0]), float(hi[0])), "MAP_CI": (float(lo[1]), float(hi[1]))}
