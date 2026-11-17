"""Mansoor Ali, Objectives 2 and 3 on TF-IDF features (validation split only).

  Obj 2  PI1  tag co-occurrence matrix (train)                 -> cooccurrence.png/.csv
         PI2  classifier chain, Logistic Regression base
         PI3  random chain orders vs frequency order
         PI4  ensemble of chains vs binary relevance (paired artist bootstrap)
         PI5  per-tag F1 grouped by tag frequency (rare vs frequent)
  Obj 3  PI1  validation protocol (artist-disjoint train/val from the frozen splits)
         PI2  grid search of LR regularisation strength C (selected by val macro-F1)
         PI3  per-tag threshold tuning vs fixed 0.5
         PI4  final comparison table (summary_chains.md)

Usage: python scripts/run_mansoor_chains.py [--n-chains 5] [--n-boot 200]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import ClassifierChain

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C                                              # noqa: E402
from src.data import load_split, load_categories                         # noqa: E402
from src.evaluate import (evaluate, tune_thresholds, annotation_metrics,   # noqa: E402
                          retrieval_metrics)
from src.models import make_vectorizer                                   # noqa: E402

OUT = C.ROOT / "results" / "mansoor"
FIG = OUT / "figures"
plt.rcParams.update({"figure.dpi": 200, "font.size": 8, "font.family": "serif",
                     "axes.spines.top": False, "axes.spines.right": False})


def log(m):
    print(m, flush=True)


def lr(C_=1.0):
    return LogisticRegression(C=C_, solver="liblinear")


def row(name, tag, res, fit_s, pred_s):
    return {"model": name, "thresholds": tag, **res.summary,
            "fit_s": round(fit_s, 1), "predict_s": round(pred_s, 2)}


def paired_bootstrap(Y, S_a, S_b, th_a, th_b, ids, artists, n_boot, seed=0):
    """Differences (b - a) in macro-F1 and MAP over artist-resampled val sets."""
    rng = np.random.default_rng(seed)
    Y = np.asarray(Y, bool); ids = np.asarray(ids, dtype=object)
    Pa, Pb = S_a >= th_a, S_b >= th_b
    codes, uniq = pd.factorize(pd.Series(artists).str.lower())
    rows_of = [np.flatnonzero(codes == a) for a in range(len(uniq))]
    d = []
    for _ in range(n_boot):
        pick = np.concatenate([rows_of[a] for a in rng.integers(0, len(uniq), len(uniq))])
        bid = np.char.add(ids[pick].astype(str), np.arange(len(pick)).astype(str))
        out = []
        for S, P in ((S_a, Pa), (S_b, Pb)):
            _, s = annotation_metrics(Y[pick], P[pick])
            ap = retrieval_metrics(Y[pick], S[pick], bid, with_auc=False)["AP"]
            out.append((s["macro_F1"], np.nanmean(ap)))
        d.append((out[1][0] - out[0][0], out[1][1] - out[0][1]))
    d = np.array(d)
    lo, hi = np.percentile(d, [2.5, 97.5], axis=0)
    return {"macro_F1_diff_CI": [float(lo[0]), float(hi[0])],
            "MAP_diff_CI": [float(lo[1]), float(hi[1])],
            "macro_F1_diff_mean": float(d[:, 0].mean()), "MAP_diff_mean": float(d[:, 1].mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-chains", type=int, default=5)
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--c-grid", type=float, nargs="+", default=[0.3, 1.0, 3.0, 10.0])
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)

    train, val = load_split("train"), load_split("val")
    cats = load_categories()
    K = len(train.tags)
    log(f"train {len(train.ids):,}  val {len(val.ids):,}  K={K}")

    # ------------------------------------------------ Obj 2 PI1: co-occurrence
    Yt = train.Y.astype(np.float32)
    co = Yt.T @ Yt
    d = np.sqrt(np.diag(co))
    lift = co / np.outer(d, d)                       # cosine-normalised co-occurrence
    np.fill_diagonal(lift, 0)
    pd.DataFrame(co.astype(int), index=train.tags, columns=train.tags).to_csv(OUT / "cooccurrence.csv")
    order = np.argsort(-Yt.sum(0))[:40]
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    im = ax.imshow(lift[np.ix_(order, order)], cmap="viridis")
    ax.set_xticks(range(40)); ax.set_yticks(range(40))
    ax.set_xticklabels([train.tags[i] for i in order], rotation=90, fontsize=4)
    ax.set_yticklabels([train.tags[i] for i in order], fontsize=4)
    fig.colorbar(im, ax=ax, fraction=0.046, label="cosine co-occurrence")
    fig.tight_layout(); fig.savefig(FIG / "cooccurrence.png"); plt.close(fig)
    iu = np.triu_indices(K, 1)
    top = np.argsort(-lift[iu])[:15]
    pairs = [(train.tags[iu[0][i]], train.tags[iu[1][i]], round(float(lift[iu][i]), 3)) for i in top]
    pd.DataFrame(pairs, columns=["tag_a", "tag_b", "cosine"]).to_csv(OUT / "top_cooccurring_pairs.csv", index=False)
    log(f"top pairs: {pairs[:5]}")

    # ------------------------------------------------ features
    t = time.time()
    vec = make_vectorizer()
    X_tr = vec.fit_transform(train.texts).astype(np.float32)
    X_va = vec.transform(val.texts).astype(np.float32)
    for X in (X_tr, X_va):          # avoid read-only memmap errors in joblib workers
        X.sum_duplicates(); X.sort_indices()
    log(f"TF-IDF {X_tr.shape[1]:,} features ({time.time() - t:.0f}s)")

    rows, per_tag = [], {}
    cat_rows = []

    def score(name, S, fit_s, pred_s):
        th = tune_thresholds(val.Y, S)
        for tag, thr in (("fixed_0.5", 0.5), ("tuned_on_val", th)):
            res = evaluate(val.Y, S, val.ids, val.tags, cats, thresholds=thr)
            rows.append(row(name, tag, res, fit_s, pred_s))
            per_tag[(name, tag)] = res.per_tag
            cat_rows.append(res.per_category.assign(model=name, thresholds=tag))
            log(f"  [{name} | {tag}] macro-F1 {res.summary['macro_F1']:.3f} "
                f"MAP {res.summary['MAP']:.3f} AUC {res.summary['ROC-AUC']:.3f}")
        return th

    # ------------------------------------------------ Obj 3 PI2: grid over C
    grid = []
    scores_br = {}
    for c in args.c_grid:
        cache = OUT / f"grid_C{c:g}_val_scores.npy"
        if cache.exists():                       # resume after an interrupted run
            S = np.load(cache).astype(float); fit_s = json.load(open(OUT / f"grid_C{c:g}_fit_s.json"))
        else:
            t = time.time()
            m = OneVsRestClassifier(lr(c), n_jobs=2).fit(X_tr, train.Y)
            fit_s = time.time() - t
            S = m.predict_proba(X_va)
            np.save(cache, S.astype(np.float32)); json.dump(fit_s, open(OUT / f"grid_C{c:g}_fit_s.json", "w"))
        th = tune_thresholds(val.Y, S)
        res = evaluate(val.Y, S, val.ids, val.tags, cats, thresholds=th)
        grid.append({"C": c, "macro_F1_tuned": res.summary["macro_F1"], "MAP": res.summary["MAP"],
                     "ROC-AUC": res.summary["ROC-AUC"], "fit_s": round(fit_s, 1)})
        scores_br[c] = (S, fit_s)
        log(f"C={c}: macro-F1(tuned) {res.summary['macro_F1']:.3f} MAP {res.summary['MAP']:.3f}")
    grid = pd.DataFrame(grid)
    grid.to_csv(OUT / "grid_C.csv", index=False)
    best_c = float(grid.sort_values(["macro_F1_tuned", "MAP"], ascending=False).iloc[0]["C"])
    log(f"best C = {best_c}")
    S_br, fit_br = scores_br[best_c]
    th_br = score(f"br_lr_C{best_c:g}", S_br, fit_br, 0.0)

    # ------------------------------------------------ Obj 2 PI2-PI4: chains
    freq = np.argsort(-train.Y.sum(0))            # most frequent first
    chain_scores = {}

    def run_chain(name, order, seed):
        cache = OUT / f"{name}_val_scores.npy"
        meta = OUT / f"{name}_times.json"
        if cache.exists() and meta.exists():     # resume after an interrupted run
            S = np.load(cache).astype(float); fit_s, pred_s = json.load(open(meta))
        else:
            t = time.time()
            ch = ClassifierChain(lr(best_c), order=order, cv=None, random_state=seed).fit(X_tr, train.Y)
            fit_s = time.time() - t
            t = time.time()
            S = ch.predict_proba(X_va)
            pred_s = time.time() - t
            np.save(cache, S.astype(np.float32)); json.dump([fit_s, pred_s], open(meta, "w"))
        chain_scores[name] = S
        log(f"{name}: fit {fit_s:.0f}s")
        score(name, S, fit_s, pred_s)

    run_chain("chain_freq_order", freq, 0)
    rng = np.random.default_rng(C.SEED)
    for i in range(args.n_chains):
        run_chain(f"chain_random_{i}", rng.permutation(K), i + 1)

    rand_names = [n for n in chain_scores if n.startswith("chain_random")]
    S_ens = np.mean([chain_scores[n] for n in rand_names], axis=0)
    th_ens = score(f"chain_ensemble_{len(rand_names)}", S_ens, 0.0, 0.0)
    np.save(OUT / "br_val_scores.npy", S_br.astype(np.float32))
    np.save(OUT / "chain_ensemble_val_scores.npy", S_ens.astype(np.float32))

    # ------------------------------------------------ Obj 2 PI4: paired bootstrap
    log("paired artist bootstrap: BR vs chain ensemble ...")
    pb = paired_bootstrap(val.Y, S_br, S_ens, th_br, th_ens, val.ids, val.artists, args.n_boot, C.SEED)
    log(str(pb))
    json.dump(pb, open(OUT / "paired_bootstrap_br_vs_ensemble.json", "w"), indent=1)

    # ------------------------------------------------ Obj 2 PI5: rare vs frequent
    br_f = per_tag[(f"br_lr_C{best_c:g}", "tuned_on_val")].set_index("tag")
    en_f = per_tag[(f"chain_ensemble_{len(rand_names)}", "tuned_on_val")].set_index("tag")
    ntr = pd.Series(train.Y.sum(0), index=train.tags)
    band = pd.qcut(ntr.rank(method="first"), 3, labels=["rare", "medium", "frequent"])
    cmp_ = pd.DataFrame({"train_count": ntr, "band": band,
                         "F1_br": br_f["F1"], "F1_chain_ens": en_f["F1"],
                         "AP_br": br_f["AP"], "AP_chain_ens": en_f["AP"]})
    cmp_["dF1"] = cmp_["F1_chain_ens"] - cmp_["F1_br"]
    cmp_["dAP"] = cmp_["AP_chain_ens"] - cmp_["AP_br"]
    cmp_.to_csv(OUT / "rare_vs_frequent_per_tag.csv")
    bands = cmp_.groupby("band", observed=True)[["F1_br", "F1_chain_ens", "dF1", "AP_br",
                                                  "AP_chain_ens", "dAP"]].mean().round(4)
    bands.to_csv(OUT / "rare_vs_frequent.csv")
    log("\n" + bands.to_string())

    # ------------------------------------------------ outputs
    summ = pd.DataFrame(rows)
    summ.to_csv(OUT / "summary_chains.csv", index=False)
    pd.concat(cat_rows).to_csv(OUT / "per_category_chains.csv", index=False)
    cols = ["model", "thresholds", "micro_F1", "macro_F1", "hamming_loss", "MAP", "P@10",
            "R-precision", "ROC-AUC", "fit_s", "predict_s"]
    s = summ[cols].round(3)
    md = ["# Mansoor: BR vs classifier chains (TF-IDF, validation split)", "",
          "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    md += ["| " + " | ".join(str(v) for v in r) + " |" for r in s.itertuples(index=False)]
    (OUT / "summary_chains.md").write_text("\n".join(md) + "\n")

    fig, ax = plt.subplots(figsize=(3.3, 2.1))
    names = ["BR"] + ["chain\n(freq)"] + ["chain\nrand."] + ["chain\nens."]
    tuned = summ[summ.thresholds == "tuned_on_val"].set_index("model")["macro_F1"]
    rnd = np.mean([tuned[n] for n in rand_names])
    vals = [tuned[f"br_lr_C{best_c:g}"], tuned["chain_freq_order"], rnd,
            tuned[f"chain_ensemble_{len(rand_names)}"]]
    ax.bar(names, vals, color=["#4C72B0", "#DD8452", "#8172B3", "#55A868"])
    ax.set_ylabel("macro-F1 (tuned, val)"); ax.set_ylim(min(vals) - 0.02, max(vals) + 0.01)
    fig.tight_layout(); fig.savefig(FIG / "chains_macroF1.png"); plt.close(fig)
    log(f"saved to {OUT}")


if __name__ == "__main__":
    main()
