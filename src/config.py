"""Project-wide settings. Change values here, not inside the scripts."""
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]

# Where the raw Music4All copy lives. Set the M4A_DIR environment variable
# or pass --m4a-dir on the command line; this is only the fallback.
M4A_DIR = Path(os.environ.get("M4A_DIR", ROOT / "data" / "music4all"))

# Outputs
PROCESSED_DIR = ROOT / "data" / "processed"   # lyrics text: NOT committed (copyright)
SPLITS_DIR = ROOT / "splits"                   # ids, splits, vocabulary: committed
REPORTS_DIR = ROOT / "reports"
FIG_DIR = REPORTS_DIR / "figures"

# Raw file names (tab-separated). Edit if your copy differs.
FILES = {
    "information": "id_information.csv",  # id, artist, song, album_name
    "tags": "id_tags.csv",                # id, tags (comma-separated)
    "genres": "id_genres.csv",            # id, genres (comma-separated)
    "lang": "id_lang.csv",                # id, lang
}
SEP = "\t"
LYRICS_DIR = "lyrics"   # one <id>.txt file per song

# Filtering (README Section 2)
LANGUAGE = "en"
MIN_TOKENS = 50

# Leakage-free protocol (README Section 2)
SPLIT_FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}
CANDIDATE_TAGS = 200        # broad set used only to balance the split
TOP_K = 100                 # final vocabulary, chosen on train frequency
MIN_POS_EVAL = 10           # drop tags with < 10 positives in val or test

# Category top-up (README Section 2 risk mitigation): genre tags dominate the
# top-K, so extra tags of these categories are added, in training-frequency
# order, until each has at least this many tags (if enough are frequent enough).
MIN_PER_CATEGORY = {"mood/emotion": 15, "theme": 15}
MIN_TRAIN_COUNT_EXTRA = 100  # an added tag needs >= 100 training songs
CATEGORY_OVERRIDES = ROOT / "splits" / "category_overrides.csv"   # manual review
TAG_SYNONYMS = ROOT / "splits" / "tag_synonyms.csv"   # manual merges of same-meaning tags
SEED = 575
