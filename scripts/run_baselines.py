"""Untuned baselines (Soorya, Objective 2), scored on the VALIDATION split.

Models (README 3.2):
  popularity   predict the n most frequent training tags for every song
  random       random scores; annotation = Bernoulli(training prevalence)
  tfidf_lr     TF-IDF + one-vs-rest Logistic Regression
  tfidf_svm    TF-IDF + one-vs-rest Linear SVM, sigmoid-calibrated (5-fold, by artist)
  tfidf_nb     TF-IDF + one-vs-rest Multinomial NB, sigmoid-calibrated (5-fold, by artist)

Annotation metrics are reported at the fixed 0.5 threshold and with per-tag
thresholds tuned on validation (README 3.4; optimistic, documentation only).
The test split is never touched here.

Usage:
  python scripts/run_baselines.py                 # all models
  python scripts/run_baselines.py --models tfidf_lr popularity
  python scripts/run_baselines.py --bootstrap 1000   # adds artist-bootstrap CIs (slow)
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C                                            # noqa: E402
from src.data import load_split, load_categories                       # noqa: E402
from src.evaluate import evaluate, tune_thresholds, bootstrap_ci       # noqa: E402
from src.models import TFIDF, make_vectorizer, make_model             # noqa: E402

RESULTS = C.ROOT / "results" / "baselines"
ALL = ["popularity", "random", "tfidf_lr", "tfidf_svm", "tfidf_nb"]


def to_markdown(df):
    head = "| " + " | ".join(df.columns) + " |"
    sep = "|" + "---|" * len(df.columns)
    body = ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join([head, sep, *body])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=ALL, choices=ALL)
    ap.add_argument("--bootstrap", type=int, default=0, help="artist-bootstrap resamples (0 = off)")
    args = ap.parse_args()

    train, val = load_split("train"), load_split("val")
    cats = load_categories()
    prev = train.Y.mean(axis=0)
    n_top = int(np.median(train.Y.sum(axis=1)))
    print(f"train {len(train.ids):,} songs, val {len(val.ids):,} songs, K = {len(train.tags)}")

    X_tr = X_va = None
    vec_time = 0.0
    if any(m.startswith("tfidf") for m in args.models):
        t = time.time()
        vec = make_vectorizer()
        X_tr = vec.fit_transform(train.texts)
        X_va = vec.transform(val.texts)
        vec_time = time.time() - t
        print(f"TF-IDF: {X_tr.shape[1]:,} features ({vec_time:.0f}s)")

    RESULTS.mkdir(parents=True, exist_ok=True)
    rows, cat_rows = [], []
    rng = np.random.default_rng(C.SEED)
    for name in args.models:
        print(f"\n== {name}")
        t0 = time.time()
        if name == "popularity":
            S = np.tile(prev, (len(val.ids), 1))
            pred = np.zeros_like(val.Y, dtype=bool)
            pred[:, np.argsort(-prev)[:n_top]] = True
            fit_s = pred_s = 0.0
        elif name == "random":
            S = rng.random(val.Y.shape)
            pred = rng.random(val.Y.shape) < prev
            fit_s = pred_s = 0.0
        else:
            model = make_model(name, train.artists)
            model.fit(X_tr, train.Y)
            fit_s = time.time() - t0
            t1 = time.time()
            S = model.predict_proba(X_va)
            pred_s = time.time() - t1
            pred = None

        variants = [("fixed_0.5", pred if pred is not None else S >= 0.5)]
        if pred is None:
            th = tune_thresholds(val.Y, S)
            variants.append(("tuned_on_val", S >= th))
            np.save(RESULTS / f"{name}_val_thresholds.npy", th)
        np.save(RESULTS / f"{name}_val_scores.npy", S.astype(np.float32))

        for label, P in variants:
            res = evaluate(val.Y, S, val.ids, val.tags, cats, pred=P)
            row = {"model": name, "thresholds": label, **res.summary,
                   "fit_s": round(fit_s, 1), "predict_s": round(pred_s, 2)}
            if args.bootstrap and label == "fixed_0.5":
                row.update(bootstrap_ci(val.Y, S, P, val.ids, val.artists,
                                        n_boot=args.bootstrap, seed=C.SEED))
            rows.append(row)
            res.per_tag.to_csv(RESULTS / f"{name}_{label}_per_tag.csv", index=False)
            pc = res.per_category.assign(model=name, thresholds=label)
            cat_rows.append(pc)
            print(f"  [{label}] micro-F1 {res.summary['micro_F1']:.3f}  macro-F1 "
                  f"{res.summary['macro_F1']:.3f}  MAP {res.summary['MAP']:.3f}  "
                  f"AUC {res.summary['ROC-AUC']:.3f}  P@10 {res.summary['P@10']:.3f}")
        print(f"  fit {fit_s:.0f}s, predict {pred_s:.1f}s")

    summary = pd.DataFrame(rows)
    summary.to_csv(RESULTS / "summary_val.csv", index=False)
    pd.concat(cat_rows).to_csv(RESULTS / "per_category_val.csv", index=False)
    json.dump({"tfidf": {k: str(v) for k, v in TFIDF.items()}, "tfidf_seconds": round(vec_time, 1),
               "popularity_n_tags": n_top, "split": "val"},
              open(RESULTS / "run_info.json", "w"), indent=1)

    cols = ["model", "thresholds", "micro_F1", "macro_F1", "hamming_loss", "MAP",
            "P@10", "R-precision", "ROC-AUC", "fit_s", "predict_s"]
    md = ["# Baselines on the validation split", "",
          "Untuned models; `tuned_on_val` thresholds are chosen on the same split, so those "
          "annotation scores are optimistic (README 3.4). Retrieval metrics do not depend on "
          "thresholds.", "",
          to_markdown(summary[cols].round(3))]
    (RESULTS / "summary_val.md").write_text("\n".join(md) + "\n")
    print("\n" + "\n".join(md))
    print(f"\nSaved to {RESULTS}")


if __name__ == "__main__":
    main()
