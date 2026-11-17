"""Deterministic lyric statistics, repetition, and CMUdict end-rhyme features.

No fitted statistics or labels enter extraction. Scaling is fitted on train only.
Chorus repetition is a structural proxy, not automatic chorus identification.
"""
from collections import Counter
from functools import lru_cache
import re

import numpy as np
import pandas as pd

from .m4a_io import clean_lyrics, tokens

FEATURE_NAMES = [
    "n_chars", "n_words", "n_unique_words", "type_token_ratio", "n_lines",
    "mean_word_chars", "std_word_chars", "mean_line_words", "std_line_words",
    "repeated_line_ratio", "chorus_repetition", "repeated_word_bigram_ratio",
    "end_rhyme_rate", "rhyme_pair_coverage", "exclamation_per_word",
]


@lru_cache(maxsize=1)
def pronunciation_dictionary():
    try:
        import cmudict
    except ImportError as exc:
        raise RuntimeError("Install requirements.txt for phonetic rhyme extraction") from exc
    return cmudict.dict()


@lru_cache(maxsize=100_000)
def rhyme_keys(word):
    """All rhyme tails from the final stressed vowel; preserve alternate readings."""
    keys = set()
    for phones in pronunciation_dictionary().get(word, []):
        vowels = [i for i, p in enumerate(phones) if p[-1:] in {"1", "2"}]
        if not vowels:
            vowels = [i for i, p in enumerate(phones) if p[-1:].isdigit()]
        if vowels:
            keys.add(tuple(re.sub(r"\d", "", p) for p in phones[vowels[-1]:]))
    return frozenset(keys)


def extract_one(text):
    text = clean_lyrics(text)
    words = tokens(text)
    line_words = [tokens(line) for line in text.splitlines()]
    line_words = [line for line in line_words if line]
    lines = [" ".join(line) for line in line_words]
    n, nl = len(words), len(lines)
    word_lens = [len(w) for w in words]
    line_lens = [len(line) for line in line_words]
    # A repeated two-line block is a transparent proxy for recurring chorus text.
    blocks = Counter(zip(lines, lines[1:]))
    covered = set()
    for i, block in enumerate(zip(lines, lines[1:])):
        if blocks[block] > 1:
            covered.update((i, i + 1))
    bigrams = list(zip(words, words[1:]))
    endings = [line[-1] for line in line_words]
    eligible = comparable = rhyming = 0
    for i, word in enumerate(endings):
        for other in endings[i + 1:i + 3]:
            if word == other:  # exact repetition is not evidence of phonetic rhyme
                continue
            eligible += 1
            ka, kb = rhyme_keys(word), rhyme_keys(other)
            if ka and kb:
                comparable += 1
                rhyming += bool(ka & kb)
    vals = [
        len(text), n, len(set(words)), len(set(words)) / max(1, n), nl,
        np.mean(word_lens) if n else 0, np.std(word_lens) if n else 0,
        np.mean(line_lens) if nl else 0, np.std(line_lens) if nl else 0,
        (nl - len(set(lines))) / max(1, nl), len(covered) / max(1, nl),
        (len(bigrams) - len(set(bigrams))) / max(1, len(bigrams)),
        rhyming / max(1, comparable), comparable / max(1, eligible),
        text.count("!") / max(1, n),
    ]
    return np.asarray(vals, dtype=np.float32)


def extract_features(texts, ids=None):
    """One finite row per input song, retaining the original order."""
    values = np.stack([extract_one(t) for t in texts]) if texts else np.empty((0, len(FEATURE_NAMES)))
    if not np.isfinite(values).all():
        raise ValueError("Non-finite stylistic feature")
    out = pd.DataFrame(values, columns=FEATURE_NAMES)
    if ids is not None:
        if len(ids) != len(out) or len(set(ids)) != len(ids):
            raise ValueError("Feature IDs must be unique and align with text rows")
        out.insert(0, "id", ids)
    return out
