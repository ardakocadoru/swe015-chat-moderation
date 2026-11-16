"""One place that knows how to load every model of the ladder.

Used by the benchmark (scripts/06_benchmark.py) and by the application (app.py),
so that both always run exactly the same code path.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .bert_model import BertOnnxModel, BertTorchModel
from .config import (BERT_FP32_DIR, BERT_ONNX_DIR, LEXICON_PATH, ONNX_FP32_PATH,
                     ONNX_INT8_PATH, TFIDF_PATH, W2V_CNN_DIR, W2V_DIR)
from .lexicon import LexiconMatcher
from .tfidf_model import TfidfModel
from .w2v_cnn import W2VCNNModel
from .w2v_ffn import W2VFFNModel


@dataclass
class ModelSpec:
    key: str
    display: str
    ladder_step: str
    precision: str
    runtime: str
    files: list[Path]          # artefacts counted for the on-disk size
    short: str = ""            # compact name for the app's tables


SPECS = {
    "lexicon": ModelSpec("lexicon", "Rule: edit-distance lexicon", "Rule baseline (extra)",
                         "rules", "Python", [LEXICON_PATH], "Lexicon (rules)"),
    "tfidf": ModelSpec("tfidf", "tf-idf + Logistic Regression", "1 · Classical baseline",
                       "FP64", "scikit-learn", [TFIDF_PATH], "1 · tf-idf + LR"),
    "w2v_ffn": ModelSpec("w2v_ffn", "word2vec + Feedforward NN", "2a · Neural network (ours, extra)",
                         "FP32", "PyTorch", [W2V_DIR], "2a · w2v + FFN"),
    "w2v_cnn": ModelSpec("w2v_cnn", "word2vec + 1D CNN", "2 · Neural network (ours)",
                         "FP32", "PyTorch", [W2V_CNN_DIR], "2 · w2v + 1D CNN"),
    "bert_fp32": ModelSpec("bert_fp32", "BERTurk fine-tuned", "3 · Transformer",
                           "FP32", "PyTorch", [BERT_FP32_DIR], "3 · BERTurk FP32"),
    "bert_onnx_fp32": ModelSpec("bert_onnx_fp32", "BERTurk fine-tuned (ONNX)", "3b · Transformer, ONNX graph",
                                "FP32", "ONNX Runtime", [ONNX_FP32_PATH], "3b · BERTurk ONNX FP32"),
    "bert_int8": ModelSpec("bert_int8", "BERTurk fine-tuned, quantized", "4 · Quantized Transformer",
                           "INT8 (dynamic)", "ONNX Runtime", [ONNX_INT8_PATH], "4 · BERTurk INT8 (served)"),
}
APP_DEFAULT = "bert_int8"


def size_mb(spec: ModelSpec) -> float:
    total = 0
    for p in spec.files:
        if p.is_dir():
            total += sum(f.stat().st_size for f in p.rglob("*") if f.is_file()
                         and not f.name.startswith("tokenizer") and f.name != "meta.json")
        else:
            total += p.stat().st_size
    return total / 2**20


def load_model(key: str, num_threads: int | None = None):
    if key == "lexicon":
        m = LexiconMatcher.load(LEXICON_PATH)
        m.threshold = min(m.lexicon.values()) if m.lexicon else 0.5   # any match => offensive
    elif key == "tfidf":
        m = TfidfModel.load(TFIDF_PATH)
    elif key == "w2v_ffn":
        m = W2VFFNModel.load(W2V_DIR)
    elif key == "w2v_cnn":
        m = W2VCNNModel.load(W2V_CNN_DIR)
    elif key == "bert_fp32":
        m = BertTorchModel(BERT_FP32_DIR, num_threads=num_threads)
    elif key == "bert_onnx_fp32":
        m = BertOnnxModel(ONNX_FP32_PATH, BERT_ONNX_DIR, SPECS[key].display, num_threads)
    elif key == "bert_int8":
        m = BertOnnxModel(ONNX_INT8_PATH, BERT_ONNX_DIR, SPECS[key].display, num_threads)
    else:
        raise KeyError(key)
    m.name = SPECS[key].display
    return m


def predict_one(model, text: str) -> tuple[int, float]:
    """(label, P(offensive)) for one message using the model's tuned threshold."""
    p = float(np.asarray(model.predict_proba([text]))[0])
    return int(p >= model.threshold), p


def available(key: str) -> bool:
    return all(Path(p).exists() for p in SPECS[key].files)
