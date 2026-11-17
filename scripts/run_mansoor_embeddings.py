"""Mansoor, Objective 1 (and Objective 3 PI4-PI5): sentence-embedding taggers on the
validation split, compared with Soorya's TF-IDF + LR baseline.

  PI2  one-vs-rest LR on all-MiniLM-L6-v2 embeddings vs TF-IDF (paired artist bootstrap; RQ2)
  PI3  embedding time and memory (second model: --extra-emb, e.g. mean-pooled static word vectors)
  PI4  long lyrics: truncation (first window) vs mean vs max pooling over windows
  PI5' late fusion of the TF-IDF and embedding scores (weight chosen on validation)

Needs: scripts/embed_lyrics.py output in data/processed/emb_minilm and
results/mansoor/ from scripts/run_mansoor_chains.py (cached TF-IDF scores).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C                                              # noqa: E402
from src.data import load_split, load_categories                         # noqa: E402
from src.evaluate import evaluate, tune_thresholds                       # noqa: E402
from run_mansoor_chains import paired_bootstrap                          # noqa: E402

OUT = C.ROOT / "results" / "mansoor"


def log(m):
    print(m, flush=True)


def pooled(emb_dir, ids_wanted):
    z = np.load(emb_dir / "chunks_meta.npz", allow_pickle=True)
    song_idx, chunk_idx, ids = z["song_idx"], z["chunk_idx"], z["ids"]
    E = np.load(emb_dir / "chunk_emb.npy", mmap_mode="r")
    n_songs = len(ids)
    first = np.zeros((n_songs, E.shape[1]), np.float32)
    sm = np.zeros_like(first); mx = np.full_like(first, -np.inf); cnt = np.zeros(n_songs)
    for s in range(0, len(song_idx), 20000):
        e = np.asarray(E[s:s + 20000], dtype=np.float32)
        si, ci = song_idx[s:s + 20000], chunk_idx[s:s + 20000]
        f = ci == 0
        first[si[f]] = e[f]
        np.add.at(sm, si, e); np.add.at(cnt, si, 1)
        np.maximum.at(mx, si, e)
    mean = sm / cnt[:, None]
    pos = {i: k for k, i in enumerate(ids)}
    rows = np.array([pos[i] for i in ids_wanted])
    return {"first": first[rows], "mean": mean[rows], "max": mx[rows]}, int((cnt > 1).sum())


class JointLR:
    """K independent L2-regularised logistic regressions fitted jointly with L-BFGS.

    The summed binary cross-entropy of all tags plus an L2 penalty is separable by tag, so
    this is the same model as OneVsRestClassifier(LogisticRegression(C=C)) (sklearn objective
    0.5*|w|^2 + C*sum(loss)), but fitted in seconds on dense features instead of 112 liblinear runs.
    """
    def __init__(self, C, max_iter=300):
        self.C, self.max_iter = C, max_iter

    def fit(self, X, Y):
        import torch
        torch.manual_seed(C_SEED)
        X = torch.tensor(X, dtype=torch.float32); Y = torch.tensor(Y, dtype=torch.float32)
        n, d = X.shape
        W = torch.zeros(d, Y.shape[1], requires_grad=True)
        b = torch.zeros(Y.shape[1], requires_grad=True)
        opt = torch.optim.LBFGS([W, b], lr=1.0, max_iter=self.max_iter, history_size=20,
                                tolerance_grad=1e-6, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            z = X @ W + b
            loss = torch.nn.functional.binary_cross_entropy_with_logits(z, Y, reduction="sum") \
                + 0.5 / self.C * (W ** 2).sum()
            loss = loss / n
            loss.backward()
            return loss
        opt.step(closure)
        self.W, self.b = W.detach(), b.detach()
        return self

    def predict_proba(self, X):
        import torch
        z = torch.tensor(X, dtype=torch.float32) @ self.W + self.b
        return torch.sigmoid(z).numpy().astype(float)


C_SEED = C.SEED


def fit_lr(Xtr, Ytr, c):
    t = time.time()
    m = JointLR(c).fit(Xtr, Ytr)
    return m, time.time() - t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb-dir", default=str(C.PROCESSED_DIR / "emb_minilm"))
    ap.add_argument("--c-grid", type=float, nargs="+", default=[0.1, 1.0, 10.0])
    ap.add_argument("--n-boot", type=int, default=200)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    train, val = load_split("train"), load_split("val")
    cats = load_categories()
    emb_dir = Path(args.emb_dir)
    Ptr, n_multi = pooled(emb_dir, train.ids)
    Pva, _ = pooled(emb_dir, val.ids)
    info = json.load(open(emb_dir / "info.json")) if (emb_dir / "info.json").exists() else {}
    mem_mb = (emb_dir / "chunk_emb.npy").stat().st_size / 1e6
    log(f"{n_multi:,} songs have >1 window; chunk embeddings {mem_mb:.0f} MB; info {info}")

    grid = pd.read_csv(OUT / "grid_C.csv")
    best_c = float(grid.sort_values(["macro_F1_tuned", "MAP"], ascending=False).iloc[0]["C"])
    S_tfidf = np.load(OUT / f"grid_C{best_c:g}_val_scores.npy").astype(float)
    th_tfidf = tune_thresholds(val.Y, S_tfidf)

    rows, scores, fits = [], {}, {}

    def record(name, S, fit_s):
        th = tune_thresholds(val.Y, S)
        for tag, thr in (("fixed_0.5", 0.5), ("tuned_on_val", th)):
            r = evaluate(val.Y, S, val.ids, val.tags, cats, thresholds=thr)
            rows.append({"model": name, "thresholds": tag, **r.summary, "fit_s": round(fit_s, 1)})
            if tag == "tuned_on_val":
                r.per_tag.to_csv(OUT / f"emb_{name}_per_tag.csv", index=False)
                r.per_category.assign(model=name).to_csv(OUT / f"emb_{name}_per_category.csv", index=False)
                log(f"  [{name}] macro-F1 {r.summary['macro_F1']:.3f} MAP {r.summary['MAP']:.3f} "
                    f"AUC {r.summary['ROC-AUC']:.3f} P@10 {r.summary['P@10']:.3f}")
        return th

    # TF-IDF reference row (cached scores from run_mansoor_chains.py)
    record(f"tfidf_lr_C{best_c:g}", S_tfidf, float(json.load(open(OUT / f"grid_C{best_c:g}_fit_s.json"))))

    # PI4: pooling variants, small C grid on the first-window features
    for pool in ("first", "mean", "max"):
        sc = StandardScaler().fit(Ptr[pool])
        Xtr, Xva = sc.transform(Ptr[pool]), sc.transform(Pva[pool])
        best = None
        for c in args.c_grid:
            m, fit_s = fit_lr(Xtr, train.Y, c)
            S = m.predict_proba(Xva)
            th = tune_thresholds(val.Y, S)
            mf = evaluate(val.Y, S, val.ids, val.tags, cats, thresholds=th).summary["macro_F1"]
            log(f"  {pool} C={c}: tuned macro-F1 {mf:.3f}")
            if best is None or mf > best[0]:
                best = (mf, c, S, fit_s)
        _, c, S, fit_s = best
        name = f"minilm_{pool}_C{c:g}"
        scores[pool] = (name, S)
        record(name, S, fit_s)
        np.save(OUT / f"{name}_val_scores.npy", S.astype(np.float32))


    # PI3: second embedding model (static GloVe vectors, mean-pooled) for comparison
    gz = C.PROCESSED_DIR / "glove_mean.npz"
    if gz.exists():
        g = np.load(gz, allow_pickle=True)
        gpos = {i: k for k, i in enumerate(g["ids"])}
        Gtr = g["E"][[gpos[i] for i in train.ids]]; Gva = g["E"][[gpos[i] for i in val.ids]]
        sc = StandardScaler().fit(Gtr)
        best = None
        for c in args.c_grid:
            m, fit_s = fit_lr(sc.transform(Gtr), train.Y, c)
            S = m.predict_proba(sc.transform(Gva))
            mf = evaluate(val.Y, S, val.ids, val.tags, cats,
                          thresholds=tune_thresholds(val.Y, S)).summary["macro_F1"]
            log(f"  glove C={c}: tuned macro-F1 {mf:.3f}")
            if best is None or mf > best[0]:
                best = (mf, c, S, fit_s)
        _, c, S, fit_s = best
        record(f"glove100_mean_C{c:g}", S, fit_s)

    # pick the best pooling by validation macro-F1 (MAP tie-break)
    tuned = pd.DataFrame(rows).query("thresholds == 'tuned_on_val'").set_index("model")
    best_pool = max(scores, key=lambda p: (tuned.loc[scores[p][0], "macro_F1"], tuned.loc[scores[p][0], "MAP"]))
    name_e, S_emb = scores[best_pool]
    th_emb = tune_thresholds(val.Y, S_emb)
    log(f"best pooling: {best_pool} ({name_e})")

    # RQ2 paired bootstrap (embedding minus TF-IDF)
    pb = paired_bootstrap(val.Y, S_tfidf, S_emb, th_tfidf, th_emb, val.ids, val.artists, args.n_boot, C.SEED)
    pb["best_pooling"] = best_pool
    json.dump(pb, open(OUT / "paired_bootstrap_emb_vs_tfidf.json", "w"), indent=1)
    log(f"RQ2 paired bootstrap (emb - tfidf): {pb}")

    # late fusion
    best_f = None
    for w in (0.3, 0.5, 0.7):
        S = w * S_tfidf + (1 - w) * S_emb
        th = tune_thresholds(val.Y, S)
        r = evaluate(val.Y, S, val.ids, val.tags, cats, thresholds=th)
        if best_f is None or (r.summary["macro_F1"], r.summary["MAP"]) > best_f[0]:
            best_f = ((r.summary["macro_F1"], r.summary["MAP"]), w, S)
    _, w, S_fuse = best_f
    record(f"late_fusion_w{w}", S_fuse, 0.0)
    th_f = tune_thresholds(val.Y, S_fuse)
    pbf = paired_bootstrap(val.Y, S_tfidf, S_fuse, th_tfidf, th_f, val.ids, val.artists, args.n_boot, C.SEED)
    pbe = paired_bootstrap(val.Y, S_emb, S_fuse, th_emb, th_f, val.ids, val.artists, args.n_boot, C.SEED)
    json.dump({"fusion_minus_tfidf": pbf, "fusion_minus_embedding": pbe, "w_tfidf": w},
              open(OUT / "paired_bootstrap_fusion.json", "w"), indent=1)
    log(f"fusion - tfidf: {pbf}\nfusion - emb: {pbe}")

    s = pd.DataFrame(rows)
    s.to_csv(OUT / "summary_embeddings.csv", index=False)
    cols = ["model", "thresholds", "micro_F1", "macro_F1", "hamming_loss", "MAP", "P@10",
            "R-precision", "ROC-AUC", "fit_s"]
    md = ["# Mansoor: embeddings vs TF-IDF (validation split)", "",
          "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    md += ["| " + " | ".join(str(v) for v in r) + " |" for r in s[cols].round(3).itertuples(index=False)]
    (OUT / "summary_embeddings.md").write_text("\n".join(md) + "\n")
    json.dump({"embedding_info": info, "chunk_emb_MB": round(mem_mb, 1), "songs_multi_window": n_multi},
              open(OUT / "embedding_cost.json", "w"), indent=1)


if __name__ == "__main__":
    main()
