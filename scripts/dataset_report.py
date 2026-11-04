"""Dataset statistics and plots (Soorya, Objective 1 PI2-PI4).

Writes reports/dataset_stats.md (the table for README Section 2) and figures
in reports/figures/. Run after build_dataset.py.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C                                  # noqa: E402
from src.data import load_split, load_vocab, load_categories  # noqa: E402

plt.rcParams.update({"figure.dpi": 150, "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False})


def main():
    info = json.loads((C.SPLITS_DIR / "build_info.json").read_text())
    vocab = load_vocab()
    cats = load_categories()
    parts = {s: load_split(s) for s in ("train", "val", "test")}
    Y_all = np.vstack([p.Y for p in parts.values()])
    lengths = np.array([len(t.split()) for p in parts.values() for t in p.texts])
    artists = {s: set(p.artists) for s, p in parts.items()}

    # ------------------------------------------------------------- checks
    overlap = {f"{a}&{b}": len(artists[a] & artists[b])
               for a, b in (("train", "val"), ("train", "test"), ("val", "test"))}
    assert all(v == 0 for v in overlap.values()), f"Artist leakage across splits: {overlap}"

    prev = pd.DataFrame({s: p.Y.mean(axis=0) for s, p in parts.items()}, index=vocab)
    prev["max_abs_diff_vs_train"] = (prev[["val", "test"]].sub(prev["train"], axis=0)).abs().max(axis=1)
    sizes = {s: len(p.ids) for s, p in parts.items()}
    total = sum(sizes.values())

    cat_counts = pd.Series([cats.get(t, "other") for t in vocab]).value_counts()
    tags_per_song = Y_all.sum(axis=1)

    # ------------------------------------------------------------- table
    lines = ["# Dataset statistics", "",
             "## Filtering funnel", "", "| Step | Songs |", "|---|---|"]
    lines += [f"| {name} | {n:,} |" for name, n in info["funnel"]]
    lines += ["", "## Summary (README Section 2 table)", "", "| Statistic | Value |", "|---|---|",
              f"| Songs after language and length filters | {total:,} |",
              f"| Artists after filtering | {info['n_artists']:,} |",
              "| Train / validation / test songs | " + " / ".join(f"{sizes[s]:,}" for s in sizes)
              + " (" + " / ".join(f"{100*sizes[s]/total:.1f}%" for s in sizes) + ") |",
              f"| Final vocabulary size *K* | {len(vocab)} (top {C.TOP_K} + "
              f"{len(info['added_for_category'])} added for mood/theme) |",
              "| Tags per category | " + ", ".join(f"{c}: {n}" for c, n in cat_counts.items()) + " |",
              f"| Median tags per song (in vocabulary) | {np.median(tags_per_song):.0f} |",
              f"| Median lyric length (words) | {np.median(lengths):.0f} |",
              "", "## Split checks", "",
              f"- Artist overlap between splits: {overlap} (must all be 0)",
              f"- Largest per-tag prevalence difference vs. train: "
              f"{prev['max_abs_diff_vs_train'].max():.3f} "
              f"(median {prev['max_abs_diff_vs_train'].median():.3f})",
              f"- Tags merged as spelling variants: {info['n_tag_merges']}",
              f"- Tags added to reach category minimums: {len(info['added_for_category'])}"
              + (f" ({', '.join(info['added_for_category'])})" if info['added_for_category'] else ""),
              f"- Tags dropped for < {C.MIN_POS_EVAL} val/test positives: {len(info['dropped_tags'])}"]
    C.REPORTS_DIR.mkdir(exist_ok=True)
    (C.REPORTS_DIR / "dataset_stats.md").write_text("\n".join(lines) + "\n")
    prev.round(4).to_csv(C.REPORTS_DIR / "tag_prevalence_by_split.csv")

    # ------------------------------------------------------------- figures
    C.FIG_DIR.mkdir(parents=True, exist_ok=True)
    order = np.argsort(-Y_all.sum(axis=0))

    fig, ax = plt.subplots(figsize=(7, 3))
    ax.bar(range(len(vocab)), Y_all.sum(axis=0)[order], color="#4C72B0", width=0.9)
    ax.set(xlabel="Tag rank", ylabel="Songs", title="Tag frequency (final vocabulary)")
    for i in range(min(10, len(vocab))):
        ax.annotate(vocab[order[i]], (i, Y_all.sum(axis=0)[order[i]]), rotation=60,
                    fontsize=6, ha="left", va="bottom")
    fig.tight_layout(); fig.savefig(C.FIG_DIR / "tag_frequency.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.hist(np.clip(lengths, 0, 1000), bins=50, color="#4C72B0")
    ax.set(xlabel="Words per song (clipped at 1000)", ylabel="Songs", title="Lyric length")
    fig.tight_layout(); fig.savefig(C.FIG_DIR / "lyric_length.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.hist(tags_per_song, bins=range(0, int(tags_per_song.max()) + 2), color="#4C72B0")
    ax.set(xlabel="Vocabulary tags per song", ylabel="Songs", title="Tags per song")
    fig.tight_layout(); fig.savefig(C.FIG_DIR / "tags_per_song.png"); plt.close(fig)

    top = order[:30]
    Ytop = Y_all[:, top].astype(float)
    co = Ytop.T @ Ytop
    d = np.sqrt(np.diag(co))
    cos = co / np.outer(d, d)
    np.fill_diagonal(cos, np.nan)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cos, cmap="viridis")
    names = [vocab[i] for i in top]
    ax.set_xticks(range(len(names)), names, rotation=90, fontsize=6)
    ax.set_yticks(range(len(names)), names, fontsize=6)
    ax.set_title("Tag co-occurrence (cosine), 30 most frequent tags")
    fig.colorbar(im, ax=ax, shrink=0.7)
    fig.tight_layout(); fig.savefig(C.FIG_DIR / "tag_cooccurrence.png"); plt.close(fig)

    print((C.REPORTS_DIR / "dataset_stats.md").read_text())
    print(f"Figures written to {C.FIG_DIR}")


if __name__ == "__main__":
    main()
