"""Per-tag analysis of the TF-IDF + Logistic Regression baseline (Soorya, Objective 3).

  PI1  per-tag F1 and ROC-AUC, sorted best to worst      -> per_tag_sorted.csv, per_tag_auc.png
  PI2  most informative TF-IDF terms for selected tags    -> top_terms.md / .csv
  PI3  results by tag category                            -> auc_by_category.png, per_category.csv
  PI4  learning curves on growing training subsets        -> learning_curve.csv / .png
  +    artist-bootstrap 95% CIs for macro-F1 and MAP       -> ci.json

Validation split only. Run after build_dataset.py:
  python scripts/analyze_baselines.py            # ~5-10 min
  python scripts/analyze_baselines.py --n-boot 0 --fractions 1.0   # quick
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C                                       # noqa: E402
from src.data import load_split, load_categories                  # noqa: E402
from src.evaluate import evaluate, tune_thresholds, bootstrap_ci  # noqa: E402
from src.models import make_vectorizer, make_model                # noqa: E402

OUT = C.ROOT / "results" / "analysis"
FIG = OUT / "figures"
CAT_ORDER = ["genre", "theme", "instrumentation/production", "era", "mood/emotion", "other"]
COLORS = {"genre": "#4C72B0", "theme": "#DD8452", "instrumentation/production": "#55A868",
          "era": "#8172B3", "mood/emotion": "#C44E52", "other": "#937860"}
TERM_TAGS = ["rap", "death metal", "christian", "christmas", "summer", "party", "love",
             "sad", "romantic", "happy", "80s", "female vocalists"]
plt.rcParams.update({"figure.dpi": 200, "font.size": 8, "axes.spines.top": False,
                     "axes.spines.right": False, "font.family": "serif"})


def fit_lr(texts, Y, artists):
    vec = make_vectorizer()
    X = vec.fit_transform(texts)
    model = make_model("tfidf_lr", artists).fit(X, Y)
    return vec, model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--fractions", type=float, nargs="+", default=[0.1, 0.25, 0.5, 1.0])
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)

    train, val = load_split("train"), load_split("val")
    cats = load_categories()

    # ------------------------------------------------ full model
    t = time.time()
    vec, model = fit_lr(train.texts, train.Y, train.artists)
    fit_s = time.time() - t
    S = model.predict_proba(vec.transform(val.texts))
    th = tune_thresholds(val.Y, S)
    res = evaluate(val.Y, S, val.ids, val.tags, cats, thresholds=th)
    print(f"Full model: fit {fit_s:.0f}s, macro-F1 {res.summary['macro_F1']:.3f}, "
          f"MAP {res.summary['MAP']:.3f}, AUC {res.summary['ROC-AUC']:.3f}")

    # ------------------------------------------------ PI1: per-tag, sorted
    pt = res.per_tag.sort_values("ROC-AUC", ascending=False).reset_index(drop=True)
    pt["prevalence"] = pt["n_pos"] / len(val.ids)
    pt["AP_lift"] = pt["AP"] / pt["prevalence"]          # AP relative to a random ranking
    pt.round(4).to_csv(OUT / "per_tag_sorted.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 2.4))
    ax.bar(range(len(pt)), pt["ROC-AUC"], color=[COLORS.get(c, "grey") for c in pt["category"]],
           width=0.85)
    ax.axhline(0.5, color="black", lw=0.6, ls="--")
    ax.set(xlim=(-1, len(pt)), ylim=(0.4, 1.0), ylabel="ROC-AUC", xticks=[],
           xlabel=f"{len(pt)} tags, sorted by ROC-AUC")
    for i in list(range(3)) + list(range(len(pt) - 3, len(pt))):
        ax.annotate(pt.loc[i, "tag"], (i, pt.loc[i, "ROC-AUC"]), rotation=90, fontsize=6,
                    ha="center", va="bottom", xytext=(0, 2), textcoords="offset points")
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[c]) for c in CAT_ORDER if c in set(pt["category"])]
    ax.legend(handles, [c for c in CAT_ORDER if c in set(pt["category"])], ncol=6, fontsize=6,
              frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(); fig.savefig(FIG / "per_tag_auc.png"); plt.close(fig)

    # ------------------------------------------------ PI3: category view
    res.per_category.round(4).to_csv(OUT / "per_category.csv", index=False)
    present = [c for c in CAT_ORDER if c in set(pt["category"])]
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    rng = np.random.default_rng(0)
    for i, c in enumerate(present):
        v = pt.loc[pt["category"] == c, "ROC-AUC"].values
        ax.scatter(i + rng.uniform(-0.18, 0.18, len(v)), v, s=8, color=COLORS[c], alpha=0.8)
        ax.hlines(np.median(v), i - 0.3, i + 0.3, color="black", lw=1)
        ax.annotate(f"n={len(v)}", (i, 0.42), ha="center", fontsize=6)
    ax.axhline(0.5, color="grey", lw=0.5, ls="--")
    ax.set_xticks(range(len(present)),
                  [c.replace("instrumentation/production", "instr./prod.").replace("mood/emotion", "mood")
                   for c in present], rotation=30, ha="right")
    ax.set(ylabel="Per-tag ROC-AUC", ylim=(0.4, 1.0))
    fig.tight_layout(); fig.savefig(FIG / "auc_by_category.png"); plt.close(fig)

    # ------------------------------------------------ PI2: informative terms
    terms = np.array(vec.get_feature_names_out())
    rows = []
    for tag in [t for t in TERM_TAGS if t in val.tags]:
        j = val.tags.index(tag)
        coef = model.estimators_[j].coef_.ravel()
        top = np.argsort(-coef)[:8]
        rows.append({"tag": tag, "category": cats.get(tag, "other"),
                     "AUC": round(float(pt.set_index("tag").loc[tag, "ROC-AUC"]), 3),
                     "top_terms": ", ".join(terms[top])})
    tt = pd.DataFrame(rows)
    tt.to_csv(OUT / "top_terms.csv", index=False)
    md = ["| Tag | Category | AUC | Highest-weighted terms |", "|---|---|---|---|"]
    md += [f"| {r.tag} | {r.category} | {r.AUC} | {r.top_terms} |" for r in tt.itertuples()]
    (OUT / "top_terms.md").write_text("\n".join(md) + "\n")

    # ------------------------------------------------ PI4: learning curves
    lc = []
    codes, uniq = pd.factorize(pd.Series(train.artists).str.lower())
    perm = np.random.default_rng(C.SEED).permutation(len(uniq))
    for f in sorted(args.fractions):
        keep_art = set(perm[: max(1, int(round(f * len(uniq))))])
        idx = np.flatnonzero([c in keep_art for c in codes])
        t = time.time()
        if f >= 1.0:
            v, m = vec, model
        else:
            v, m = fit_lr([train.texts[i] for i in idx], train.Y[idx],
                          [train.artists[i] for i in idx])
        fs = time.time() - t if f < 1.0 else fit_s
        t = time.time()
        Sf = m.predict_proba(v.transform(val.texts))
        ps = time.time() - t
        r = evaluate(val.Y, Sf, val.ids, val.tags, cats, thresholds=tune_thresholds(val.Y, Sf))
        pc = r.per_category.set_index("category")
        lc.append({"fraction": f, "train_songs": len(idx), "fit_s": round(fs, 1),
                   "predict_s": round(ps, 1), "macro_F1": r.summary["macro_F1"],
                   "MAP": r.summary["MAP"], "ROC-AUC": r.summary["ROC-AUC"],
                   "AUC_genre": pc.loc["genre", "ROC-AUC"] if "genre" in pc.index else np.nan,
                   "AUC_mood+theme": pc.loc["mood+theme", "ROC-AUC"] if "mood+theme" in pc.index else np.nan})
        print(f"  fraction {f:.2f}: {len(idx):,} songs, fit {fs:.0f}s, MAP {r.summary['MAP']:.3f}, "
              f"AUC {r.summary['ROC-AUC']:.3f}")
    lc = pd.DataFrame(lc)
    lc.round(4).to_csv(OUT / "learning_curve.csv", index=False)
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    ax.plot(lc["train_songs"], lc["ROC-AUC"], "o-", color="black", label="all tags", ms=3)
    ax.plot(lc["train_songs"], lc["AUC_genre"], "s--", color=COLORS["genre"], label="genre", ms=3)
    ax.plot(lc["train_songs"], lc["AUC_mood+theme"], "^--", color=COLORS["mood/emotion"],
            label="mood + theme", ms=3)
    ax.set(xscale="log", xlabel="Training songs (log scale)", ylabel="Macro ROC-AUC (val)")
    ax.minorticks_off()
    ax.set_xticks(lc["train_songs"], [f"{n/1000:.1f}k" if n >= 1000 else str(n) for n in lc["train_songs"]])
    ax.legend(frameon=False, fontsize=6)
    fig.tight_layout(); fig.savefig(FIG / "learning_curve.png"); plt.close(fig)

    # ------------------------------------------------ bootstrap CIs
    out = {"model": "tfidf_lr", "thresholds": "tuned_on_val", **{k: res.summary[k] for k in
           ("macro_F1", "micro_F1", "MAP", "ROC-AUC", "P@10")}, "fit_s": round(fit_s, 1)}
    if args.n_boot:
        t = time.time()
        out.update(bootstrap_ci(val.Y, S, S >= th, val.ids, val.artists,
                                n_boot=args.n_boot, seed=C.SEED))
        out["n_boot"] = args.n_boot
        print(f"Bootstrap ({args.n_boot}) took {time.time() - t:.0f}s: {out}")
    json.dump(out, open(OUT / "ci.json", "w"), indent=1)

    print("\nTop 10 tags by AUC:\n", pt[["tag", "category", "n_pos", "F1", "AP", "ROC-AUC", "AP_lift"]]
          .head(10).round(3).to_string(index=False))
    print("\nBottom 10:\n", pt[["tag", "category", "n_pos", "F1", "AP", "ROC-AUC", "AP_lift"]]
          .tail(10).round(3).to_string(index=False))
    print("\nInformative terms:\n" + (OUT / "top_terms.md").read_text())
    print("Learning curve:\n" + lc.round(3).to_string(index=False))
    print(f"\nSaved to {OUT}")


if __name__ == "__main__":
    main()
