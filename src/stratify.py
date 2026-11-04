"""Iterative stratification (Sechidis et al., 2011) for weighted groups.

Each row is one *artist*. Its label vector holds, for every tag, the number of
that artist's songs carrying the tag, and its weight is its song count. Whole
artists are assigned to splits, so the split is artist-disjoint, while label
balance is tracked in songs.
"""
import numpy as np


def iterative_stratification(Y: np.ndarray, weights: np.ndarray, fractions, seed=0):
    """Return an array of split indices (0..len(fractions)-1), one per row.

    Y: (n_rows, n_labels) non-negative counts. weights: (n_rows,) sizes.
    """
    rng = np.random.default_rng(seed)
    Y = np.asarray(Y, dtype=float)
    w = np.asarray(weights, dtype=float)
    r = np.asarray(fractions, dtype=float)
    r = r / r.sum()
    n, L = Y.shape

    need_size = r * w.sum()                       # desired songs per split
    need_lab = np.outer(r, Y.sum(axis=0))         # desired positives per split/label
    assign = np.full(n, -1)
    remaining = np.ones(n, dtype=bool)

    def pick(need):
        # Most-needed split; ties -> most-needed overall size; ties -> random
        best = np.flatnonzero(need == need.max())
        if len(best) > 1:
            sizes = need_size[best]
            best = best[sizes == sizes.max()]
        return rng.choice(best)

    while remaining.any():
        pos_left = (Y[remaining] > 0).sum(axis=0)
        active = np.flatnonzero(pos_left > 0)
        if len(active) == 0:
            break
        # label with the fewest remaining rows (ties broken randomly)
        m = pos_left[active].min()
        lab = rng.choice(active[pos_left[active] == m])
        rows = np.flatnonzero(remaining & (Y[:, lab] > 0))
        # place big artists first so later, smaller ones can fix the balance
        rows = rows[np.argsort(-Y[rows, lab] + rng.random(len(rows)) * 1e-3)]
        for i in rows:
            j = pick(need_lab[:, lab])
            assign[i] = j
            remaining[i] = False
            need_lab[j] -= Y[i]
            need_size[j] -= w[i]

    for i in np.flatnonzero(remaining):           # rows with no labels
        j = pick(need_size)
        assign[i] = j
        need_size[j] -= w[i]
    return assign
