"""Baseline model definitions shared by the baseline and analysis scripts."""
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

TFIDF = dict(ngram_range=(1, 2), sublinear_tf=True, min_df=5, max_df=0.5,
             max_features=200_000, dtype=np.float32)

# n_jobs=4 rather than -1: the calibrated SVM ran out of worker memory with all cores
N_JOBS = 4


def make_vectorizer():
    return TfidfVectorizer(**TFIDF)


def _calibrated(est, artists):
    cv = list(GroupKFold(n_splits=5).split(np.zeros(len(artists)), groups=artists))
    return CalibratedClassifierCV(est, method="sigmoid", cv=cv)


def make_model(name, artists=None):
    if name == "tfidf_lr":
        return OneVsRestClassifier(LogisticRegression(C=1.0, solver="liblinear"), n_jobs=N_JOBS)
    if name == "tfidf_svm":
        return OneVsRestClassifier(_calibrated(LinearSVC(C=1.0), artists), n_jobs=N_JOBS)
    if name == "tfidf_nb":
        return OneVsRestClassifier(_calibrated(MultinomialNB(alpha=0.1), artists), n_jobs=N_JOBS)
    raise ValueError(name)
