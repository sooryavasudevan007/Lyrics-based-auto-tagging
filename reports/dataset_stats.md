# Dataset statistics

## Filtering funnel

| Step | Songs |
|---|---|
| All songs in Music4All | 84,103 |
| Language = en | 84,103 |
| Lyrics file present | 84,103 |
| >= 50 tokens after cleaning | 83,633 |
| At least one tag | 83,633 |

## Summary (README Section 2 table)

| Statistic | Value |
|---|---|
| Songs after language and length filters | 83,633 |
| Artists after filtering | 11,838 |
| Train / validation / test songs | 57,348 / 13,002 / 13,283 (68.6% / 15.5% / 15.9%) |
| Final vocabulary size *K* | 112 (top 100 + 12 added for mood/theme) |
| Tags per category | genre: 71, mood/emotion: 13, era: 10, instrumentation/production: 7, other: 7, theme: 4 |
| Median tags per song (in vocabulary) | 3 |
| Median lyric length (words) | 224 |

## Split checks

- Artist overlap between splits: {'train&val': 0, 'train&test': 0, 'val&test': 0} (must all be 0)
- Largest per-tag prevalence difference vs. train: 0.018 (median 0.002)
- Tags merged as spelling variants: 256
- Tags added to reach category minimums: 12 (sexy, sad, epic, chill, catchy, dreamy, atmospheric, romantic, happy, party, christian, summer)
- Tags dropped for < 10 val/test positives: 0
