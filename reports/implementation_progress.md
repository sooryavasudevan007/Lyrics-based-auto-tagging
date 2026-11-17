# Implementation progress update (2026-11-16)

**Status: 12/15 PIs done; 2 in progress; 1 not started.**

| Objective | PI | Requirement | Status | Evidence (results/Harish/) |
|---|---|---|---|---|
| 1 | PI1 | Text statistics for all songs | done | stylistic/features_all.csv.gz |
| 1 | PI2 | Repetition and phonetic end-rhyme features | done | stylistic/features_all.csv.gz |
| 1 | PI3 | Stylistic-only LR with shared evaluation | done | stylistic/summary_val.csv |
| 1 | PI4 | TF-IDF + stylistic comparison by category | done | stylistic/combined_minus_tfidf_per_category.csv |
| 1 | PI5 | Feature importance by tag/category | done | stylistic/importance_per_category.csv |
| 2 | PI1 | Category-balanced error sample | done | errors/review_sample.csv |
| 2 | PI2 | Frequent missed/extra tag pairs | done | errors/confused_pairs.csv |
| 2 | PI3 | Qualitative failure-pattern analysis | done | errors/review_sample.csv |
| 2 | PI4 | Length/repetition vs. quality | done | errors/quality_correlations.csv |
| 2 | PI5 | Word contributions for selected tags | done | errors/selected_tag_terms.csv |
| 3 | PI1 | MFCC and spectral features for fixed 10k | done | audio/run_info.json |
| 3 | PI2 | Audio LR and lyrics on identical songs | done | audio/summary_val.csv |
| 3 | PI3 | Audio/lyrics category comparison | in progress | audio/per_category_comparison.csv |
| 3 | PI4 | Modality disagreements and examples | in progress | audio/modality_disagreements.csv |
| 3 | PI5 | Pretrained audio embeddings + late fusion | not started | Stretch task |

## Report-ready text (three sentences per objective)

### Objective 1

I extracted 15 text-statistic, repetition and phonetic rhyme features for all 83,633 frozen songs and fitted preprocessing only on training data. On validation, the stylistic-only model achieved macro-F1 0.106 and MAP 0.063; combining it with TF-IDF achieved 0.173 and 0.124, versus 0.165 and 0.122 for TF-IDF alone. Per-category comparisons and standardized feature-coefficient tables are complete; next I will coordinate model selection with the team's tuning work before the single final test evaluation.

### Objective 2

I analyzed tfidf_style_lr on validation, sampled 36 category-balanced error rows, and produced missed/extra tag-pair tables plus exact word-contribution explanations. Qualitative inspection found overlapping genres, attributes not established by lyrics, and plausible tags missing from social annotations; song length correlated with song-level F1 (Spearman rho 0.215), while repetition had a much smaller association (0.066). Next I will repeat the analysis for the team's final selected model and examine rare-tag threshold behavior without using the test set.

### Objective 3

I obtained and verified all 10,000 selected audio clips, retaining 6,857 training, 1,555 validation and 1,588 test songs, and extracted 53 MFCC/spectral summary features per clip. On the identical training/validation subset, audio LR achieved macro-F1 0.149 and MAP 0.103, versus 0.127 and 0.098 for the selected matched lyrics model (matched_tfidf_style_lr). Per-category interpretation and review of modality-disagreement examples are in progress; their generated tables are preliminary, and pretrained audio embeddings and audio-lyrics late fusion remain unstarted stretch work.

## Validation results

Annotation thresholds are tuned and evaluated on the same validation split, so F1 is optimistic. Retrieval metrics use the raw scores. No test metrics are reported.

| model | thresholds | macro_F1 | micro_F1 | MAP | P@10 | ROC-AUC |
|---|---|---|---|---|---|---|
| style_lr | fixed_0.5 | 0.0113 | 0.0284 | 0.0631 | 0.0777 | 0.6814 |
| style_lr | tuned_on_val | 0.1057 | 0.1417 | 0.0631 | 0.0777 | 0.6814 |
| tfidf_lr | fixed_0.5 | 0.0195 | 0.0659 | 0.1224 | 0.2741 | 0.7595 |
| tfidf_lr | tuned_on_val | 0.1645 | 0.2067 | 0.1224 | 0.2741 | 0.7595 |
| tfidf_style_lr | fixed_0.5 | 0.0242 | 0.0842 | 0.1236 | 0.2786 | 0.7607 |
| tfidf_style_lr | tuned_on_val | 0.1731 | 0.2405 | 0.1236 | 0.2786 | 0.7607 |

### Matched audio subset

Both modalities train on the same 6,857 songs and score the same 1,555 validation songs. These results should not be compared directly to models trained on all 57,348 training songs.

| model | thresholds | macro_F1 | micro_F1 | MAP | P@10 | ROC-AUC |
|---|---|---|---|---|---|---|
| audio_mfcc_lr | fixed_0.5 | 0.0249 | 0.0902 | 0.1029 | 0.1339 | 0.7312 |
| audio_mfcc_lr | tuned_on_val | 0.1494 | 0.2433 | 0.1029 | 0.1339 | 0.7312 |
| matched_tfidf_lr | fixed_0.5 | 0.0071 | 0.0241 | 0.1051 | 0.1375 | 0.7215 |
| matched_tfidf_lr | tuned_on_val | 0.1197 | 0.0968 | 0.1051 | 0.1375 | 0.7215 |
| matched_style_lr | fixed_0.5 | 0.0104 | 0.0278 | 0.0680 | 0.0839 | 0.6560 |
| matched_style_lr | tuned_on_val | 0.0995 | 0.1573 | 0.0680 | 0.0839 | 0.6560 |
| matched_tfidf_style_lr | fixed_0.5 | 0.0130 | 0.0511 | 0.0979 | 0.1429 | 0.7036 |
| matched_tfidf_style_lr | tuned_on_val | 0.1273 | 0.1901 | 0.0979 | 0.1429 | 0.7036 |

## Error-analysis findings

The 36-row qualitative sample has equal counts by category; it is exploratory and is not a representative estimate of error-pattern prevalence. Reviews were performed by the assistant using the full local lyrics and observed/predicted tags. Musical attributes were not independently checked against audio.

Era and nationality labels usually lack direct support in lyrics; performer voice, covers and instrumentation can depend on the recording. Genre errors often involve nearby overlapping labels. The combined-model review includes plausible love predictions without observed annotations, missed love/party labels despite relevant cues, and a Christian prediction from religious imagery in a metal-tagged song. The separate TF-IDF reference misses two Christian-tagged songs with overt religious cues. Missing social tags must not be treated as verified semantic negatives.

The baseline's 0.01 threshold is an important source of extra predictions for rare tags. For example, its Canadian prediction occurs on 9,740 validation songs although only 173 have that observed label, and mellow occurs on 4,374 songs with 114 observed positives. Error-pair counts therefore include each extra tag's marginal rate and conditional lift; common extra tags do not by themselves prove a specific semantic confusion. These counts describe the TF-IDF reference; the selected model's own rates are in its error tables.

| feature | metric | spearman_rho | artist_bootstrap_low | artist_bootstrap_high | valid_bootstraps | n_finite_songs |
|---|---|---|---|---|---|---|
| n_words | sample_F1 | 0.2150 | 0.1761 | 0.2549 | 200 | 12999 |
| n_words | hamming_loss | -0.1908 | -0.2245 | -0.1507 | 200 | 13002 |
| n_lines | sample_F1 | 0.2032 | 0.1626 | 0.2385 | 200 | 12999 |
| n_lines | hamming_loss | -0.1469 | -0.1787 | -0.1145 | 200 | 13002 |
| repeated_line_ratio | sample_F1 | 0.0655 | 0.0306 | 0.1025 | 200 | 12999 |
| repeated_line_ratio | hamming_loss | 0.0595 | 0.0287 | 0.0908 | 200 | 13002 |
| chorus_repetition | sample_F1 | 0.0547 | 0.0249 | 0.0860 | 200 | 12999 |
| chorus_repetition | hamming_loss | 0.0336 | 0.0067 | 0.0635 | 200 | 13002 |

## Audio access and provenance

**Yes: all 10,000 clips have been obtained and verified.** The public mirror covers every ID in the fixed manifest. Each clip was checked against archive CRC32 and uncompressed size, and its SHA256 was recorded. Original-author delivery was not requested or confirmed; acquisition used a third-party mirror pinned to an exact revision.

Source: <https://huggingface.co/datasets/Leon299/music4all/tree/a391160e3e17f351d5ab2d05439a7d3d7f0440eb>. The authors' official access route is <https://sites.google.com/view/contact4music4all>. See `results/Harish/data_provenance.json` and `results/Harish/audio/clip_checksums.csv` for the local acquisition record.

The November 13 access condition is satisfied for the matched MFCC baseline. MFCC summaries are a limited audio reference; these results do not establish what a stronger audio encoder could achieve. Objective 3 PI5 remains stretch.

## Reproduction

See [experiment_reproducibility.md](../docs/experiment_reproducibility.md) for installation, frozen-data restoration and commands. Raw lyrics, audio, private review text, serialized models and large score arrays are excluded from Git. Numeric features, metrics, review notes and provenance are included.
