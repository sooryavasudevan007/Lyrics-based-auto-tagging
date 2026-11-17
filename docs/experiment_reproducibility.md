# Experiment reproducibility

This contribution implements the three objectives and 15 PIs in README Section 8.
It uses the committed artist-disjoint splits and the shared `src.evaluate` harness.
All exploratory model results are on **validation**. No command in this contribution
fits to, selects against, or evaluates the test labels.

## Setup

Use Python 3.10 or later, and install FFmpeg on your machine for MP3 decoding.

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -p 'test_experiments.py' -v
```

The tests use temporary synthetic fixtures. Their scores are never research results.
They exercise ID alignment, tag-column order, artist leakage, phonetic rhyme,
repetition, multi-label error pairs, multipart ZIP integrity, and the full experiment
workflow in an isolated directory.

## Restore the existing frozen dataset

If `data/processed/songs.jsonl.gz` is already present, skip this step. Otherwise use
the team's Music4All copy. Supply its tag table and either its lyrics directory or
`lyrics.zip` from the full release:

```bash
python scripts/prepare_frozen_data.py \
  --raw-dir /path/to/music4all \
  --lyrics /path/to/music4all/lyrics
```

The restorer matches all songs by Music4All ID, uses the committed normalization
and synonym maps, and writes in frozen split-file order. Missing lyrics, missing
tags, duplicate IDs or lyrics failing the existing minimum-length rule cause an
error. It does **not** run splitting or vocabulary selection again. Provenance,
including split and vocabulary hashes, is saved with the local processed corpus.

## Objective 1: stylistic features

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/run_experiments.py style
```

Fifteen features are extracted for every frozen song, including test songs, using
only the lyric text. Computing a fixed feature vector is label-independent; test
predictions and metrics are not produced.

| Feature | Definition |
| --- | --- |
| `n_chars` | Characters after the existing cleaning rules |
| `n_words`, `n_unique_words`, `type_token_ratio` | Token count, distinct-token count, and their ratio |
| `n_lines` | Nonempty tokenized lyric lines |
| `mean_word_chars`, `std_word_chars` | Word-length mean and population standard deviation |
| `mean_line_words`, `std_line_words` | Tokens per line: mean and population standard deviation |
| `repeated_line_ratio` | `(number of lines - number of distinct normalized lines) / number of lines` |
| `chorus_repetition` | Fraction of lines covered by a repeated contiguous two-line block |
| `repeated_word_bigram_ratio` | Fraction of token bigrams beyond their distinct occurrences |
| `end_rhyme_rate` | Rhyming distinct line-end word pairs divided by comparable pairs, in a two-line window |
| `rhyme_pair_coverage` | Fraction of eligible distinct-word pairs for which both words have CMUdict pronunciations |
| `exclamation_per_word` | Exclamation marks divided by token count |

Chorus repetition is a documented structural proxy. It does not identify a chorus
semantically. End rhyme uses CMUdict phonemes from the final stressed vowel and
accepts alternative pronunciations. Repeating the same end word does not count as
rhyme. Unrecognized words reduce coverage instead of silently becoming spelling
rhymes. Empty inputs yield finite features.

`StandardScaler` is fitted on the training songs only. The models are:

1. Stylistic features + one-vs-rest Logistic Regression.
2. The unchanged TF-IDF representation + Logistic Regression, rerun as a control.
3. TF-IDF concatenated with standardized stylistic features + Logistic Regression.
   The stylistic block is L2-normalized per song before concatenation, balancing
   its row norm with the already normalized TF-IDF block. This fixed rule also
   improves the conditioning of the combined optimization problem.

All use `C=1`, liblinear, seed 575, and a maximum of 1,000 iterations. The maximum
is raised from the original baseline's default to allow convergence for combined
features; regularization and text-vectorizer settings are unchanged. Signed
coefficients and mean absolute standardized coefficients by category identify
which features are associated with each tag/category. Association is not causation,
and correlated features can share coefficient weight.

Outputs are in `results/Harish/stylistic/`: feature matrix, score artifacts with IDs
and tag names, fixed/tuned metrics, category deltas, importance tables and run metadata.

## Objective 2: error analysis

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/run_experiments.py errors \
  --per-category 6 --bootstrap 200
```

The default model is the completed stylistic experiment's best lyrics model by
validation macro-F1, with MAP and model name as tie-breakers. To analyze another
model, supply `--scores /path/to/ID-aligned_val_scores.npz`. A score artifact must
explicitly contain `ids`, `tags`, `scores` and `thresholds`; silent row-order
assumptions and tag-column mismatches are rejected.

The reproducible sample contains up to six song/error-category rows for each
category, sampled uniformly among that category's erroneous songs with seed 575.
A song can appear under more than one category. `review_sample.csv` is the review
worksheet. `pattern`, `review_notes` and `reviewed` are filled after inspecting the
corresponding text in `private_review/sample_lyrics.jsonl`, which is ignored by Git.
Rerunning with the same score artifact preserves existing review annotations.

`confused_pairs.csv` counts songs where one observed tag is missed and another tag
is predicted without an observed annotation. Multi-label tags are not mutually
exclusive: these counts are **co-occurring missed/extra labels**, and an extra tag
can be an annotation omission rather than a definite musical mistake.

Prediction quality is described by per-song F1 and Hamming loss, with descriptive
Spearman correlations against lyric length and repetition. Confidence intervals
resample artists (200 draws), respecting the grouping of the frozen protocol.
There are also length/repetition quartile tables. These exploratory relationships
are not causal tests and do not adjust for tag prevalence or genre.

For selected erroneous tags, exact signed TF-IDF contributions show words that
support or oppose a decision. The saved model is reused, or refitted on the same
training data, and must reproduce the recorded scores within `1e-4`. Combined-model
word contributions cover the text columns; its stylistic terms and intercept also
contribute. No SHAP or embedding-model explanation is claimed.

## Objective 3: matched audio comparison

```bash
python scripts/run_experiments.py audio-manifest

# If the audio is already available, place <song_id>.mp3 files in this directory:
python scripts/fetch_audio_subset.py --workers 24

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/run_experiments.py audio \
  --audio-dir data/music4all/audios --workers 8
```

The completed numeric audio feature table is also committed. To reproduce model
training without downloading/decoding the clips again, use:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/run_experiments.py audio \
  --audio-features results/Harish/audio/features_all.csv.gz
```

This option requires every fixed subset ID exactly once and the complete feature
schema. Provenance distinguishes reused numeric features from local waveform
extraction; it does not count a feature table as locally available audio.

The manifest fixes **10,000 songs: 6,857 train, 1,555 validation, 1,588 test**.
Within each split songs are selected by a seeded SHA256 ranking of song IDs.
Quotas are proportional to frozen split sizes, using largest remainders. Labels,
model results and clip availability do not affect selection. A different set of
IDs cannot silently overwrite an existing manifest.

The original authors' access route is
<https://sites.google.com/view/contact4music4all>, which provides a contact address
for requesting the full release. The optional downloader instead uses a public
**third-party mirror** at <https://huggingface.co/datasets/Leon299/music4all>, pinned
to revision `a391160e3e17f351d5ab2d05439a7d3d7f0440eb`. These sources must not be
described as equivalent provenance. The downloader reads archive ranges and verifies
every extracted clip's ZIP CRC32 and uncompressed length, then records per-file
SHA256 checksums. Successful extraction proves the selected files match that
archive; it does not independently authenticate the mirror against an authors'
checksum. Downloads are resumable. Audio and raw lyrics are never added to Git.

FFmpeg decodes clips to mono 22,050 Hz, taking at most 30 seconds. The 53 summary
features are the means and standard deviations of 20 MFCCs, spectral centroid,
bandwidth, 85% rolloff, flatness, RMS and zero-crossing rate, plus duration. MFCCs
use a Hann STFT (2,048 samples; 512 hop), 40 HTK mel filters, log powers and an
orthonormal type-II DCT. Standardization and classifiers are fitted on train only.

The audio and lyrics models train on the **same 6,857 songs** and score the **same
1,555 validation songs**, with the same frozen 112-tag column order. Stylistic,
TF-IDF and combined lyrics candidates are each retrained on that subset; the
validation macro-F1 winner is used for the per-category comparison. All candidates'
metrics are retained so selection is visible. Category tables and modality
disagreement examples include observed support and song IDs. Missing clips are
reported explicitly; the sample is never silently reduced to easier songs.

MFCC summaries are a limited audio reference, not a bound on the best achievable
audio performance. Pretrained audio embeddings and audio-lyrics fusion (Objective
3 PI5) remain a separate stretch task.

## Current progress

Stylistic features and error analysis are complete. Audio feature extraction and
matched-subset model evaluation (Objective 3 PI1 and PI2) are complete.
Audio category interpretation (PI3) and review of modality-disagreement examples
(PI4) are **in progress**. The generated `audio/per_category_comparison.csv` and
`audio/modality_disagreements.csv` files are preliminary supporting outputs;
their presence does not mark these PIs complete. Pretrained audio embeddings
and late fusion (PI5) have not started and remain stretch work.

The progress summary is **12 done, 2 in progress, 1 not started**.
`python scripts/summarize_results.py --update-paper` regenerates the individual
progress report, status JSON, figures and the contributor's manuscript entries
with these statuses.

## Evaluation limits

All annotation tables show both 0.5 and validation-tuned thresholds. Threshold
tuning and reporting on the same validation songs make tuned F1 optimistic.
Retrieval metrics use raw scores and do not depend on thresholds. Macro-F1 is
primary; MAP, P@10 and ROC-AUC supplement it. Rare-tag AUC may be undefined on the
smaller audio subset; category means omit undefined values, and per-tag positive
counts are retained. Only the final frozen-model evaluation should consume the
test set.

Synthetic smoke tests, downloaded raw data, private lyric review files and large
score arrays are not paper results or public data releases. `reports/implementation_progress.md`
records the actual completed runs and the report-ready status of each PI.
