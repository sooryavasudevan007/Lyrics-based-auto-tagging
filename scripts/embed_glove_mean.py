"""Second, cheaper embedding model (Mansoor Obj.1 PI3): mean-pooled static GloVe vectors
(glove-wiki-gigaword-100, 400k words, from the gensim-data GitHub release).

Usage: python scripts/embed_glove_mean.py /path/to/glove-wiki-gigaword-100.gz
"""
import gzip, json, re, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C

t = time.time()
words, vecs = {}, []
with gzip.open(sys.argv[1], "rt", encoding="utf-8") as f:
    next(f)
    for i, line in enumerate(f):
        w, *v = line.rstrip().split(" ")
        words[w] = i; vecs.append(np.asarray(v, dtype=np.float32))
V = np.stack(vecs)
tok = re.compile(r"[a-z']+")
ids, E = [], []
with gzip.open(C.PROCESSED_DIR / "songs.jsonl.gz", "rt", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        idx = [words[w] for w in tok.findall(r["lyrics"].lower()) if w in words]
        E.append(V[idx].mean(0) if idx else np.zeros(V.shape[1], np.float32)); ids.append(r["id"])
np.savez(C.PROCESSED_DIR / "glove_mean.npz", ids=np.array(ids, dtype=object), E=np.stack(E))
json.dump({"model": "glove-wiki-gigaword-100 mean-pooled", "dim": V.shape[1],
           "seconds": round(time.time() - t, 1)}, open(C.PROCESSED_DIR / "glove_mean_info.json", "w"))
print(f"{len(ids):,} songs in {time.time() - t:.0f}s")
