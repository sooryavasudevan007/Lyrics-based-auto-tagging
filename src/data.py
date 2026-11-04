"""Shared loader: every team member gets the same songs, splits and tags.

    from src.data import load_split
    train = load_split("train")
    train.texts   # list of lyric strings
    train.Y       # (n_songs, K) 0/1 numpy array
    train.ids     # song ids
    train.tags    # the K tag names, column order of Y
"""
from dataclasses import dataclass
import gzip
import json

import numpy as np
import pandas as pd

from . import config as C


@dataclass
class Split:
    ids: list
    artists: list
    texts: list
    Y: np.ndarray
    tags: list


def load_vocab():
    return json.loads((C.SPLITS_DIR / "vocab.json").read_text())


def load_categories():
    """tag -> category (lexicon suggestion corrected by manual overrides)."""
    df = pd.read_csv(C.SPLITS_DIR / "tag_categories.csv", keep_default_na=False)
    return dict(zip(df["tag"], df["category"]))


def _songs():
    with gzip.open(C.PROCESSED_DIR / "songs.jsonl.gz", "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def load_split(split: str, level: str = "artist") -> Split:
    """split: train | val | test. level: 'artist' (default) or 'song' (leakage check only)."""
    vocab = load_vocab()
    idx = {t: i for i, t in enumerate(vocab)}
    key = "split" if level == "artist" else "split_song_level"
    rows = [r for r in _songs() if r[key] == split]
    Y = np.zeros((len(rows), len(vocab)), dtype=np.int8)
    for i, r in enumerate(rows):
        for t in r["tags"]:
            if t in idx:
                Y[i, idx[t]] = 1
    return Split([r["id"] for r in rows], [r["artist"] for r in rows],
                 [r["lyrics"] for r in rows], Y, vocab)
