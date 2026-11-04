# Data pipeline (Soorya, Objective 1)

Builds the frozen dataset that every model in the project uses.

## Run

```bash
pip install -r requirements.txt

# 1. Check your Music4All folder matches src/config.py (file names, columns)
python scripts/inspect_m4a.py --m4a-dir /path/to/music4all

# 2. Filter, normalise tags, split, freeze the vocabulary (a few minutes)
python scripts/build_dataset.py --m4a-dir /path/to/music4all

# 3. Statistics table and figures
python scripts/dataset_report.py
```

## Outputs

| Path | Contents | In Git? |
|---|---|---|
| `splits/splits.csv` | song id, artist, artist-disjoint split, song-level split (leakage check) | yes |
| `splits/vocab.json` | the final *K* tags, in column order | yes |
| `splits/category_overrides.csv` | manual category decisions with reasons (edit this) | yes |
| `splits/tag_categories.csv` | category per tag, its source (lexicon/manual), and whether it was added to top up mood/theme | yes |
| `splits/tag_merges.csv` | spelling variants merged into one tag | yes |
| `splits/tag_synonyms.csv` | manual merges of different words with the same meaning (edit this) | yes |
| `splits/build_info.json` | settings, filtering funnel, dropped tags | yes |
| `data/processed/songs.jsonl.gz` | cleaned lyrics + tags | **no** (copyright) |
| `reports/dataset_stats.md`, `reports/figures/` | statistics and plots | yes |

Teammates need their own Music4All copy and run step 2 once; because the
seed and settings are fixed, they get identical splits (check with
`git diff splits/`).

## Using the data in your code

```python
from src.data import load_split, load_categories
train, val = load_split("train"), load_split("val")
# train.texts: lyrics, train.Y: (songs x K) 0/1 matrix, train.tags: tag names
cats = load_categories()        # tag -> category
leak = load_split("test", level="song")   # song-level split, leakage check only
```

## Tag categories (Objective 1 PI5)

Categories are assigned in two documented steps, using the tag text only:

1. A keyword lexicon (`src/tags.py`) suggests a category for every tag.
2. A person reviews `splits/tag_categories.csv`; any correction goes into
   `splits/category_overrides.csv` (tag, category, reason) and wins over the
   lexicon on the next build.

Because genre tags dominate the 100 most frequent tags, the build tops up
mood/emotion and theme to 15 tags each (settings in `src/config.py`), adding
the next most frequent training tags of those categories with at least 100
training songs. Added tags are flagged in `tag_categories.csv`.
