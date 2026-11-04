"""Read the raw Music4All files into one DataFrame (one row per song)."""
from pathlib import Path
import re
import pandas as pd

from . import config as C


def _read_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=C.SEP, dtype=str, keep_default_na=False, quoting=3)
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def _split_list(s: str) -> list[str]:
    return [t.strip() for t in s.split(",") if t.strip()] if s else []


def read_lyrics(lyrics_dir: Path, ids) -> dict[str, str]:
    out = {}
    for i in ids:
        f = lyrics_dir / f"{i}.txt"
        if f.exists():
            out[i] = f.read_text(encoding="utf-8", errors="replace")
    return out


def load_raw(m4a_dir: Path) -> pd.DataFrame:
    """Columns: id, artist, song, lang, tags_raw (list), genres_raw (list), lyrics_raw (str or NA)."""
    m4a_dir = Path(m4a_dir)
    missing = [f for f in C.FILES.values() if not (m4a_dir / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing in {m4a_dir}: {missing}. Run scripts/inspect_m4a.py and "
            "adjust FILES in src/config.py.")

    info = _read_table(m4a_dir / C.FILES["information"])[["id", "artist", "song"]]
    tags = _read_table(m4a_dir / C.FILES["tags"])
    genres = _read_table(m4a_dir / C.FILES["genres"])
    lang = _read_table(m4a_dir / C.FILES["lang"])

    df = info.merge(lang[["id", "lang"]], on="id", how="left")
    df = df.merge(tags.rename(columns={tags.columns[1]: "tags_raw"}), on="id", how="left")
    df = df.merge(genres.rename(columns={genres.columns[1]: "genres_raw"}), on="id", how="left")
    for col in ("tags_raw", "genres_raw"):
        df[col] = df[col].fillna("").map(_split_list)
    df["lang"] = df["lang"].fillna("")

    lyr = read_lyrics(m4a_dir / C.LYRICS_DIR, df["id"])
    df["lyrics_raw"] = df["id"].map(lyr)
    return df


_SECTION = re.compile(r"\[[^\]]*\]|\([^)]*(chorus|verse|bridge|x\d)[^)]*\)", re.I)
_TOKEN = re.compile(r"[a-z]+(?:'[a-z]+)?")


def clean_lyrics(text: str) -> str:
    """Remove section markers like [Chorus] or (x2); normalise whitespace; keep line breaks."""
    text = _SECTION.sub(" ", text)
    lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in text.splitlines()]
    return "\n".join(ln for ln in lines if ln)


def tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())
