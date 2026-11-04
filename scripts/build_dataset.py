"""Build the frozen lyrics-tag dataset (README Section 2, leakage-free protocol).

Steps
  1. Label-independent filtering: English, lyrics with >= MIN_TOKENS tokens;
     rule-based tag normalisation.
  2. Artist-disjoint 70/15/15 split by iterative stratification on the
     CANDIDATE_TAGS most frequent tags (plus a song-level split for the
     leakage check).
  3. Final vocabulary: top-K tags by *training* frequency, topped up so that
     mood/emotion and theme reach MIN_PER_CATEGORY tags; drop tags with
     < MIN_POS_EVAL positives in validation or test.
  4. Tag categories: keyword lexicon, corrected by the manual review in
     splits/category_overrides.csv.

Usage
  python scripts/build_dataset.py --m4a-dir /path/to/music4all
"""
import argparse
import gzip
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C                                   # noqa: E402
from src.m4a_io import load_raw, clean_lyrics, tokens          # noqa: E402
from src.tags import build_merge_map, apply_normalisation, load_overrides, load_synonyms, categorise  # noqa: E402
from src.stratify import iterative_stratification              # noqa: E402

SPLITS = list(C.SPLIT_FRACTIONS)


def log(msg):
    print(msg, flush=True)


def step1_filter(df: pd.DataFrame):
    funnel = [("All songs in Music4All", len(df))]
    df = df[df["lang"].str.lower() == C.LANGUAGE].copy()
    funnel.append((f"Language = {C.LANGUAGE}", len(df)))
    df = df[df["lyrics_raw"].notna()].copy()
    funnel.append(("Lyrics file present", len(df)))
    df["lyrics"] = df["lyrics_raw"].map(clean_lyrics)
    df["n_tokens"] = df["lyrics"].map(lambda t: len(tokens(t)))
    df = df[df["n_tokens"] >= C.MIN_TOKENS].copy()
    funnel.append((f">= {C.MIN_TOKENS} tokens after cleaning", len(df)))

    merge = build_merge_map(df["tags_raw"])
    synonyms = load_synonyms(C.TAG_SYNONYMS)
    df["tags"] = apply_normalisation(df["tags_raw"], merge, synonyms)
    df = df[df["tags"].map(len) > 0].copy()
    funnel.append(("At least one tag", len(df)))
    df = df.drop(columns=["lyrics_raw"]).reset_index(drop=True)
    n_merged = sum(1 for k, v in merge.items() if k != v) + len(synonyms)
    return df, funnel, merge, n_merged


def label_matrix(tag_lists, vocab):
    idx = {t: i for i, t in enumerate(vocab)}
    Y = np.zeros((len(tag_lists), len(vocab)), dtype=np.int8)
    for r, tags in enumerate(tag_lists):
        for t in tags:
            if t in idx:
                Y[r, idx[t]] = 1
    return Y


def step2_split(df: pd.DataFrame):
    counts = Counter(t for tags in df["tags"] for t in tags)
    candidates = [t for t, _ in counts.most_common(C.CANDIDATE_TAGS)]
    Y_song = label_matrix(df["tags"], candidates)

    # Artist-disjoint: one row per artist, label counts summed over its songs
    artist_codes, artists = pd.factorize(df["artist"].str.strip().str.lower())
    n_art = len(artists)
    Y_art = np.zeros((n_art, len(candidates)), dtype=np.int32)
    np.add.at(Y_art, artist_codes, Y_song)
    w_art = np.bincount(artist_codes, minlength=n_art)
    fr = list(C.SPLIT_FRACTIONS.values())
    art_split = iterative_stratification(Y_art, w_art, fr, seed=C.SEED)
    split_artist = np.array(SPLITS)[art_split[artist_codes]]

    # Song-level split (for the leakage check only)
    song_split = iterative_stratification(Y_song, np.ones(len(df)), fr, seed=C.SEED)
    split_song = np.array(SPLITS)[song_split]
    return split_artist, split_song, n_art


def step3_vocab(df, split_col, overrides):
    """Top-K by training frequency, topped up for under-represented categories,
    then the fixed val/test minimum-positives rule."""
    train_counts = Counter(t for tags in df.loc[df[split_col] == "train", "tags"] for t in tags)
    ranked = [t for t, _ in train_counts.most_common()]
    top = ranked[:C.TOP_K]
    cat = {t: categorise(t, overrides)[0] for t in ranked}

    added = []
    for c, need in C.MIN_PER_CATEGORY.items():
        have = sum(cat[t] == c for t in top)
        for t in ranked[C.TOP_K:]:
            if have >= need or train_counts[t] < C.MIN_TRAIN_COUNT_EXTRA:
                break
            if cat[t] == c:
                top.append(t); added.append(t); have += 1

    Y = label_matrix(df["tags"], top)
    keep, dropped = [], []
    for j, t in enumerate(top):
        pos = {s: int(Y[(df[split_col] == s).values, j].sum()) for s in ("val", "test")}
        (keep if min(pos.values()) >= C.MIN_POS_EVAL else dropped).append(t)
    added = [t for t in added if t in keep]
    return keep, dropped, added, train_counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m4a-dir", default=str(C.M4A_DIR))
    args = ap.parse_args()

    log("Loading Music4All ...")
    raw = load_raw(Path(args.m4a_dir))
    log(f"  {len(raw):,} songs loaded")

    log("Step 1: filtering and tag normalisation ...")
    df, funnel, merge, n_merged = step1_filter(raw)
    for name, n in funnel:
        log(f"  {name:<40} {n:>8,}")

    log("Step 2: artist-disjoint iterative stratification ...")
    df["split"], df["split_song_level"], n_art = step2_split(df)

    log("Step 3: freezing the vocabulary on the training split ...")
    overrides = load_overrides(C.CATEGORY_OVERRIDES)
    vocab, dropped, added, train_counts = step3_vocab(df, "split", overrides)
    log(f"  K = {len(vocab)}: top {C.TOP_K} by training frequency + {len(added)} added "
        f"to reach category minimums; {len(dropped)} dropped (< {C.MIN_POS_EVAL} val/test positives)")

    log("Step 4: tag categories (lexicon + manual overrides) ...")
    rows = []
    for t in vocab:
        c, src = categorise(t, overrides)
        rows.append({"tag": t, "train_count": train_counts[t], "category": c, "source": src,
                     "reason": overrides.get(t, ("", ""))[1], "added_for_category": t in added})
    cats = pd.DataFrame(rows)
    log("  " + ", ".join(f"{k}: {v}" for k, v in cats["category"].value_counts().items()))

    # ---------------------------------------------------------- write outputs
    C.SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    C.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    df[["id", "artist", "split", "split_song_level"]].to_csv(C.SPLITS_DIR / "splits.csv", index=False)
    (C.SPLITS_DIR / "vocab.json").write_text(json.dumps(vocab, indent=1))
    cats.to_csv(C.SPLITS_DIR / "tag_categories.csv", index=False)
    pd.Series(merge, name="canonical").rename_axis("variant").reset_index() \
        .query("variant != canonical").to_csv(C.SPLITS_DIR / "tag_merges.csv", index=False)
    json.dump({
        "config": {k: getattr(C, k) for k in
                   ("LANGUAGE", "MIN_TOKENS", "SPLIT_FRACTIONS", "CANDIDATE_TAGS",
                    "TOP_K", "MIN_POS_EVAL", "MIN_PER_CATEGORY",
                    "MIN_TRAIN_COUNT_EXTRA", "SEED")},
        "funnel": funnel, "n_artists": int(n_art), "n_tag_merges": n_merged,
        "dropped_tags": dropped, "added_for_category": added,
    }, open(C.SPLITS_DIR / "build_info.json", "w"), indent=1)

    with gzip.open(C.PROCESSED_DIR / "songs.jsonl.gz", "wt", encoding="utf-8") as f:
        for r in df[["id", "artist", "song", "lyrics", "n_tokens", "tags", "split",
                     "split_song_level"]].itertuples(index=False):
            f.write(json.dumps(r._asdict(), ensure_ascii=False) + "\n")

    log("Done. Committed outputs are in splits/; lyrics are in data/processed/ (not committed).")
    log("Next: python scripts/dataset_report.py")


if __name__ == "__main__":
    main()
