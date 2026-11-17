"""Restore local lyrics using the committed splits, vocabulary and tag merges.

Unlike build_dataset.py this never regenerates or changes the frozen protocol.
Accepts a lyrics directory or the full-release lyrics.zip (no extraction needed).
"""
import argparse
import gzip
import json
from pathlib import Path
import sys
import zipfile

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C
from src.experiment_utils import sha256, write_json
from src.m4a_io import clean_lyrics, tokens, _read_table
from src.tags import normalise_one


def prepare(raw_dir, lyrics_path, output_dir, splits_dir):
    raw_dir, lyrics_path = Path(raw_dir), Path(lyrics_path)
    output_dir, splits_dir = Path(output_dir), Path(splits_dir)
    frozen = pd.read_csv(splits_dir / "splits.csv", dtype=str, keep_default_na=False)
    if frozen.id.duplicated().any():
        raise ValueError("Duplicate frozen song IDs")
    tags = _read_table(raw_dir / "id_tags.csv").set_index("id")
    if tags.index.duplicated().any():
        raise ValueError("Duplicate IDs in raw tag table")
    missing = set(frozen.id) - set(tags.index)
    if missing:
        raise ValueError(f"Raw tag table is missing {len(missing)} frozen songs")
    merge = pd.read_csv(splits_dir / "tag_merges.csv", keep_default_na=False)
    merge = dict(zip(merge.variant, merge.canonical))
    syn = pd.read_csv(splits_dir / "tag_synonyms.csv", keep_default_na=False, comment="#")
    syn = dict(zip(syn.variant.map(normalise_one), syn.canonical.map(normalise_one)))
    archive = zipfile.ZipFile(lyrics_path) if lyrics_path.is_file() else None
    try:
        if archive:
            names = {Path(n).stem: n for n in archive.namelist()
                     if n.endswith(".txt") and "__MACOSX" not in n}
        records = []
        for r in frozen.itertuples(index=False):
            if archive:
                if r.id not in names:
                    raise ValueError(f"Lyrics missing for frozen ID {r.id}")
                text = archive.read(names[r.id]).decode("utf-8", errors="replace")
            else:
                text = (lyrics_path / f"{r.id}.txt").read_text(encoding="utf-8", errors="replace")
            text = clean_lyrics(text)
            n = len(tokens(text))
            if n < C.MIN_TOKENS:
                raise ValueError(f"Frozen ID {r.id} has only {n} tokens in this copy")
            normalized = []
            for t in str(tags.loc[r.id].iloc[0]).split(","):
                t = normalise_one(t)
                if t:
                    t = merge.get(t, t)
                    normalized.append(syn.get(t, t))
            records.append({"id": r.id, "artist": r.artist, "lyrics": text,
                            "tags": list(dict.fromkeys(normalized)), "n_tokens": n,
                            "split": r.split, "split_song_level": r.split_song_level})
    finally:
        if archive:
            archive.close()
    # Atomic output: no half-written processed corpus on failure.
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / "songs.jsonl.gz"
    temp = target.with_suffix(".tmp")
    with gzip.open(temp, "wt", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    temp.replace(target)
    provenance = {"songs": len(records), "split_counts": frozen.split.value_counts().to_dict(),
                  "splits_sha256": sha256(splits_dir / "splits.csv"),
                  "vocab_sha256": sha256(splits_dir / "vocab.json"),
                  "raw_tags_sha256": sha256(raw_dir / "id_tags.csv"),
                  "lyrics_sha256": sha256(lyrics_path) if lyrics_path.is_file() else None,
                  "test_evaluated": False}
    write_json(output_dir / "provenance.json", provenance)
    print(json.dumps(provenance, indent=2), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-dir", type=Path, required=True)
    ap.add_argument("--lyrics", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, default=C.PROCESSED_DIR)
    ap.add_argument("--splits-dir", type=Path, default=C.SPLITS_DIR)
    args = ap.parse_args()
    prepare(args.raw_dir, args.lyrics, args.output_dir, args.splits_dir)
