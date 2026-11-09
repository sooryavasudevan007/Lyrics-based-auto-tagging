# Baselines on the validation split

Untuned models; `tuned_on_val` thresholds are chosen on the same split, so those annotation scores are optimistic (README 3.4). Retrieval metrics do not depend on thresholds.

| model | thresholds | micro_F1 | macro_F1 | hamming_loss | MAP | P@10 | R-precision | ROC-AUC | fit_s | predict_s |
|---|---|---|---|---|---|---|---|---|---|---|
| popularity | fixed_0.5 | 0.202 | 0.009 | 0.043 | 0.028 | 0.029 | 0.028 | 0.5 | 0.0 | 0.0 |
| random | fixed_0.5 | 0.079 | 0.028 | 0.05 | 0.028 | 0.029 | 0.026 | 0.504 | 0.0 | 0.0 |
| tfidf_lr | fixed_0.5 | 0.066 | 0.019 | 0.027 | 0.122 | 0.274 | 0.158 | 0.759 | 24.5 | 0.45 |
| tfidf_lr | tuned_on_val | 0.207 | 0.165 | 0.089 | 0.122 | 0.274 | 0.158 | 0.759 | 24.5 | 0.45 |
| tfidf_svm | fixed_0.5 | 0.078 | 0.04 | 0.027 | 0.113 | 0.262 | 0.147 | 0.716 | 85.1 | 2.33 |
| tfidf_svm | tuned_on_val | 0.186 | 0.161 | 0.099 | 0.113 | 0.262 | 0.147 | 0.716 | 85.1 | 2.33 |
| tfidf_nb | fixed_0.5 | 0.066 | 0.023 | 0.027 | 0.078 | 0.138 | 0.093 | 0.599 | 14.3 | 8.1 |
| tfidf_nb | tuned_on_val | 0.066 | 0.088 | 0.452 | 0.078 | 0.138 | 0.093 | 0.599 | 14.3 | 8.1 |
