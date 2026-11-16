"""Classical baseline: tf-idf features + logistic regression (technique area B)."""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from .normalize import normalize, tokenize


def _full_normalize(text: str) -> str:
    return normalize(text, "full")


def build_pipeline(C: float = 4.0) -> Pipeline:
    features = FeatureUnion([
        # word unigrams + bigrams on our regex tokens
        ("word", TfidfVectorizer(preprocessor=_full_normalize, tokenizer=tokenize,
                                 token_pattern=None, ngram_range=(1, 2), min_df=2,
                                 sublinear_tf=True)),
        # character 2-5 grams inside word boundaries: tolerates suffixes and typos
        ("char", TfidfVectorizer(preprocessor=_full_normalize, analyzer="char_wb",
                                 ngram_range=(2, 5), min_df=3, sublinear_tf=True,
                                 max_features=200_000)),
    ])
    clf = LogisticRegression(C=C, class_weight="balanced", max_iter=3000)
    return Pipeline([("tfidf", features), ("clf", clf)])


class TfidfModel:
    name = "tf-idf + Logistic Regression"

    def __init__(self, pipeline: Pipeline, threshold: float = 0.5):
        self.pipeline = pipeline
        self.threshold = threshold

    def predict_proba(self, texts) -> np.ndarray:
        return self.pipeline.predict_proba(list(texts))[:, 1]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"pipeline": self.pipeline, "threshold": self.threshold}, path, compress=3)

    @classmethod
    def load(cls, path: Path) -> "TfidfModel":
        obj = joblib.load(path)
        return cls(obj["pipeline"], obj["threshold"])
