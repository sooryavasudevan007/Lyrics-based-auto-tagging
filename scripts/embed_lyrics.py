"""Sentence-embedding extraction for all lyrics (Mansoor, Objective 1 PI1, PI3, PI4).

Each lyric is tokenised without truncation and cut into windows of --chunk
tokens (at most --max-chunks). Every window is encoded with an
all-MiniLM-L6-v2 ONNX model (mean pooling over tokens, L2-normalised), and the
per-window vectors are cached so that "first window" (= truncation), "mean of
windows" and "max of windows" pooling can all be derived without re-encoding.

The script is resumable: progress is checkpointed to --out every 100 batches.

Usage:
  python scripts/embed_lyrics.py --model-dir /path/to/onnx_model_dir --out data/processed/emb_minilm
"""
import argparse
import gzip
import json
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C                                  # noqa: E402

CLS, SEP = 101, 102


def log(m):
    print(m, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True, help="dir with model.onnx and tokenizer.json")
    ap.add_argument("--out", default=str(C.PROCESSED_DIR / "emb_minilm"))
    ap.add_argument("--chunk", type=int, default=256, help="tokens per window incl. [CLS]/[SEP]")
    ap.add_argument("--max-chunks", type=int, default=3)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    md = Path(args.model_dir)

    tok = Tokenizer.from_file(str(md / "tokenizer.json"))
    tok.no_padding(); tok.no_truncation()

    meta_f = out / "chunks_meta.npz"
    if meta_f.exists():
        z = np.load(meta_f, allow_pickle=True)
        song_idx, chunk_idx, ids, lens = z["song_idx"], z["chunk_idx"], z["ids"], z["lens"]
        tokens = None
    else:
        log("tokenising ...")
        t = time.time()
        with gzip.open(C.PROCESSED_DIR / "songs.jsonl.gz", "rt", encoding="utf-8") as f:
            rows = [json.loads(l) for l in f]
        ids = np.array([r["id"] for r in rows], dtype=object)
        texts = [r["lyrics"] for r in rows]
        del rows
        body = args.chunk - 2
        song_l, chunk_l, tok_l = [], [], []
        for s in range(0, len(texts), 2000):
            for i, e in enumerate(tok.encode_batch(texts[s:s + 2000], add_special_tokens=False)):
                seq = e.ids
                n = max(1, min(args.max_chunks, -(-len(seq) // body)))
                for c in range(n):
                    w = seq[c * body:(c + 1) * body]
                    if c > 0 and len(w) < 16:          # drop tiny tail windows
                        continue
                    song_l.append(s + i); chunk_l.append(c); tok_l.append([CLS] + w + [SEP])
        song_idx = np.array(song_l, dtype=np.int32); chunk_idx = np.array(chunk_l, dtype=np.int8)
        lens = np.array([len(x) for x in tok_l], dtype=np.int32)
        flat = np.concatenate([np.array(x, dtype=np.int32) for x in tok_l])
        np.savez(meta_f, song_idx=song_idx, chunk_idx=chunk_idx, ids=ids, lens=lens)
        np.save(out / "tokens_flat.npy", flat)
        log(f"{len(texts):,} songs -> {len(song_idx):,} windows ({time.time() - t:.0f}s); "
            f"{(chunk_idx > 0).sum():,} extra windows")
        del tok_l

    flat = np.load(out / "tokens_flat.npy", mmap_mode="r")
    offsets = np.concatenate([[0], np.cumsum(lens)])
    n = len(song_idx)
    emb_f = out / "chunk_emb.npy"
    done_f = out / "done.npy"
    emb = np.lib.format.open_memmap(emb_f, mode="r+" if emb_f.exists() else "w+",
                                    dtype=np.float16, shape=(n, 384))
    done = np.load(done_f) if done_f.exists() else np.zeros(n, dtype=bool)

    so = ort.SessionOptions(); so.intra_op_num_threads = args.threads
    sess = ort.InferenceSession(str(md / "model.onnx"), so, providers=["CPUExecutionProvider"])
    order = np.argsort(-lens, kind="stable")             # long first; batches have little padding
    todo = order[~done[order]]
    log(f"{len(todo):,} of {n:,} windows left")
    t0 = time.time(); k = 0
    for b in range(0, len(todo), args.batch):
        idx = todo[b:b + args.batch]
        L = int(lens[idx].max())
        ii = np.zeros((len(idx), L), dtype=np.int64); mm = np.zeros_like(ii)
        for r, j in enumerate(idx):
            seq = flat[offsets[j]:offsets[j + 1]]
            ii[r, :len(seq)] = seq; mm[r, :len(seq)] = 1
        h = sess.run(None, {"input_ids": ii, "attention_mask": mm,
                            "token_type_ids": np.zeros_like(ii)})[0]
        e = (h * mm[..., None]).sum(1) / mm.sum(1, keepdims=True)
        e /= np.linalg.norm(e, axis=1, keepdims=True)
        emb[idx] = e.astype(np.float16); done[idx] = True
        k += 1
        if k % 100 == 0:
            emb.flush(); np.save(done_f, done)
            rate = (b + len(idx)) / (time.time() - t0)
            log(f"  {done.sum():,}/{n:,} windows  {rate:.0f}/s  eta {(len(todo) - b) / rate / 60:.0f} min")
    emb.flush(); np.save(done_f, done)
    json.dump({"model": "all-MiniLM-L6-v2 (fp32 ONNX, Xenova export)", "chunk_tokens": args.chunk,
               "max_chunks": args.max_chunks, "n_windows": int(n), "seconds": round(time.time() - t0, 1),
               "threads": args.threads}, open(out / "info.json", "w"), indent=1)
    log(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
