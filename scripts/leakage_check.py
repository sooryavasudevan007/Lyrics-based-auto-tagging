"""Leakage check (README 3.5; Soorya, Objective 1 PI4).

Trains the same TF-IDF + Logistic Regression baseline twice, with the same
frozen K-tag vocabulary and hyperparameters, so that only the split differs:

  artist  - the artist-disjoint split used everywhere in the project
  song    - a song-level stratified split, where an artist's songs can be in
            both train and validation

If the song-level scores are clearly higher, models can exploit an artist's
recurring vocabulary, and the artist-disjoint protocol is needed.

Validation only; the test split is never used.
  python scripts/leakage_check.py               # ~3-5 min
  python scripts/leakage_check.py --n-boot 0    # no confidence intervals
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C                                   # noqa: E402
from src.data import load_split, load_categories              # noqa: E402
from src.evaluate import evaluate, tune_thresholds, bootstrap_ci  # noqa: E402
from src.models import make_vectorizer, make_model            # noqa: E402

OUT = C.ROOT / "results" / "analysis"


def run(level, cats, n_boot):
    tr, va = load_split("train", level=level), load_split("val", level=level)
    t = time.time()
    vec = make_vectorizer()
    X = vec.fit_transform(tr.texts)
    model = make_model("tfidf_lr", tr.artists).fit(X, tr.Y)
    S = model.predict_proba(vec.transform(va.texts))
    th = tune_thresholds(va.Y, S)
    res = evaluate(va.Y, S, va.ids, va.tags, cats, thresholds=th)
    shared = len(set(a.lower() for a in tr.artists) & set(a.lower() for a in va.artists))
    row = {"split": level, "train_songs": len(tr.ids), "val_songs": len(va.ids),
           "val_artists_also_in_train": shared,
           **{k: res.summary[k] for k in ("macro_F1", "micro_F1", "MAP", "P@10", "ROC-AUC")},
           "fit_s": round(time.time() - t, 1)}
    draws = None
    if n_boot:
        ci = bootstrap_ci(va.Y, S, S >= th, va.ids, va.artists, n_boot=n_boot, seed=C.SEED)
        row.update({"macro_F1_CI": ci["macro_F1_CI"], "MAP_CI": ci["MAP_CI"]})
    pc = res.per_category.assign(split=level)
    print(f"  {level:6s}: macro-F1 {row['macro_F1']:.3f}  MAP {row['MAP']:.3f}  "
          f"AUC {row['ROC-AUC']:.3f}  ({shared:,} val artists also in train)")
    return row, pc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=500)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    cats = load_categories()

    rows, pcs = [], []
    for level in ("artist", "song"):
        r, pc = run(level, cats, args.n_boot)
        rows.append(r); pcs.append(pc)
    a, s = rows
    gap = {k: s[k] - a[k] for k in ("macro_F1", "MAP", "ROC-AUC", "P@10")}

    pc = pd.concat(pcs)
    piv = pc.pivot(index="category", columns="split", values="ROC-AUC")
    piv["gap"] = piv["song"] - piv["artist"]

    out = {"artist_disjoint": a, "song_level": s, "song_minus_artist": gap}
    json.dump(out, open(OUT / "leakage_check.json", "w"), indent=1, default=float)
    pc.round(4).to_csv(OUT / "leakage_check_per_category.csv", index=False)

    def ci(r, k):
        c = r.get(f"{k}_CI")
        return f" [{c[0]:.3f}, {c[1]:.3f}]" if c else ""

    md = ["# Leakage check: artist-disjoint vs. song-level split (validation)", "",
          "Same TF-IDF + LR model, vocabulary and hyperparameters; only the split differs. "
          "Thresholds tuned on each validation split.", "",
          "| Split | Val artists also in train | Macro-F1 | MAP | P@10 | ROC-AUC |",
          "|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['split']} | {r['val_artists_also_in_train']:,} | {r['macro_F1']:.3f}"
                  f"{ci(r, 'macro_F1')} | {r['MAP']:.3f}{ci(r, 'MAP')} | {r['P@10']:.3f} | "
                  f"{r['ROC-AUC']:.3f} |")
    md += ["", "**Song-level minus artist-disjoint:** " +
           ", ".join(f"{k} {v:+.3f}" for k, v in gap.items()), "",
           "## ROC-AUC by category", "", "| Category | Artist | Song | Gap |", "|---|---|---|---|"]
    md += [f"| {c} | {r.artist:.3f} | {r.song:.3f} | {r.gap:+.3f} |"
           for c, r in piv.sort_values("gap", ascending=False).iterrows()]
    (OUT / "leakage_check.md").write_text("\n".join(md) + "\n")
    print("\n" + "\n".join(md))
    print(f"\nSaved to {OUT}")


if __name__ == "__main__":
    main()
