"""Per-tag significance tests between two baselines (Soorya, Objective 3 PI5).

Paired artist-bootstrap of the per-tag AP difference (model A minus model B)
on the validation split, with Benjamini-Hochberg correction across the K
tags (FDR 0.05), plus the aggregate macro-F1 and MAP differences.

Uses the score matrices saved by run_baselines.py when present
(results/baselines/<model>_val_scores.npy), otherwise refits the models.

  python scripts/per_tag_tests.py                      # tfidf_lr vs tfidf_svm, ~5-10 min
  python scripts/per_tag_tests.py --a tfidf_lr --b tfidf_nb --n-boot 1000
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C                                   # noqa: E402
from src.data import load_split, load_categories              # noqa: E402
from src.evaluate import tune_thresholds, paired_bootstrap    # noqa: E402
from src.models import make_vectorizer, make_model            # noqa: E402

BASE = C.ROOT / "results" / "baselines"
OUT = C.ROOT / "results" / "analysis"


def scores(name, train, val, cache):
    f = BASE / f"{name}_val_scores.npy"
    if f.exists():
        S = np.load(f)
        if S.shape == val.Y.shape:
            print(f"  {name}: using saved scores {f.name}")
            return S.astype(float)
    print(f"  {name}: refitting")
    if "X" not in cache:
        vec = make_vectorizer()
        cache["X"] = vec.fit_transform(train.texts)
        cache["Xv"] = vec.transform(val.texts)
    return make_model(name, train.artists).fit(cache["X"], train.Y).predict_proba(cache["Xv"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="tfidf_lr")
    ap.add_argument("--b", default="tfidf_svm")
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    train, val = load_split("train"), load_split("val")
    cats = load_categories()
    cache = {}
    S_a, S_b = scores(args.a, train, val, cache), scores(args.b, train, val, cache)
    P_a, P_b = S_a >= tune_thresholds(val.Y, S_a), S_b >= tune_thresholds(val.Y, S_b)

    print(f"Paired bootstrap ({args.n_boot} artist resamples) ...")
    r = paired_bootstrap(val.Y, S_a, P_a, S_b, P_b, val.ids, val.artists,
                         n_boot=args.n_boot, seed=C.SEED, per_tag=True)
    pt = r.pop("per_tag")
    pt.insert(0, "tag", val.tags)
    pt.insert(1, "category", [cats.get(t, "other") for t in val.tags])
    pt.insert(2, "n_pos", val.Y.sum(axis=0))
    pt["significant_BH"] = pt["q_BH"] < 0.05
    pt["winner"] = np.where(~pt["significant_BH"], "no reliable difference",
                            np.where(pt["AP_diff"] > 0, args.a, args.b))
    pt = pt.sort_values("AP_diff", ascending=False)
    tag = f"{args.a}_vs_{args.b}"
    pt.round(4).to_csv(OUT / f"per_tag_tests_{tag}.csv", index=False)

    n_sig = int(pt["significant_BH"].sum())
    by_cat = (pt.groupby("category")
                .agg(tags=("tag", "size"), sig=("significant_BH", "sum"),
                     mean_AP_diff=("AP_diff", "mean"))
                .sort_values("mean_AP_diff", ascending=False))
    json.dump({"a": args.a, "b": args.b, **r, "n_tags": len(pt),
               "n_significant_BH": n_sig,
               "a_better": int(((pt["AP_diff"] > 0) & pt["significant_BH"]).sum()),
               "b_better": int(((pt["AP_diff"] < 0) & pt["significant_BH"]).sum())},
              open(OUT / f"per_tag_tests_{tag}.json", "w"), indent=1, default=float)

    md = [f"# Per-tag paired tests: {args.a} minus {args.b} (validation)", "",
          f"Paired artist bootstrap, {args.n_boot} resamples; per-tag AP differences "
          "corrected with Benjamini-Hochberg (FDR 0.05). Thresholds tuned on validation.", "",
          f"- **Macro-F1:** {r['macro_F1']['diff']:+.3f} "
          f"[{r['macro_F1']['CI'][0]:+.3f}, {r['macro_F1']['CI'][1]:+.3f}], p = {r['macro_F1']['p']:.3f}",
          f"- **MAP:** {r['MAP']['diff']:+.3f} [{r['MAP']['CI'][0]:+.3f}, {r['MAP']['CI'][1]:+.3f}], "
          f"p = {r['MAP']['p']:.3f}",
          f"- **Tags with a reliable AP difference:** {n_sig} of {len(pt)} "
          f"({args.a} better on {int(((pt['AP_diff'] > 0) & pt['significant_BH']).sum())}, "
          f"{args.b} better on {int(((pt['AP_diff'] < 0) & pt['significant_BH']).sum())})", "",
          "## By category", "", "| Category | Tags | Significant | Mean AP diff |", "|---|---|---|---|"]
    md += [f"| {c} | {int(x.tags)} | {int(x.sig)} | {x.mean_AP_diff:+.3f} |" for c, x in by_cat.iterrows()]
    md += ["", "## Largest differences", "", "| Tag | Category | AP diff | 95% CI | q (BH) |",
           "|---|---|---|---|---|"]
    show = pd.concat([pt.head(5), pt.tail(5)])
    md += [f"| {x.tag} | {x.category} | {x.AP_diff:+.3f} | [{x.CI_low:+.3f}, {x.CI_high:+.3f}] | "
           f"{x.q_BH:.3f} |" for x in show.itertuples()]
    (OUT / f"per_tag_tests_{tag}.md").write_text("\n".join(md) + "\n")
    print("\n" + "\n".join(md))
    print(f"\nSaved to {OUT}")


if __name__ == "__main__":
    main()
