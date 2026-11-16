"""Transformer models (technique areas C and D).

* ``BertTorchModel``  - fine-tuned BERTurk, FP32, run with PyTorch on the CPU.
* ``BertOnnxModel``   - the same network exported to ONNX and run with ONNX
                        Runtime; used both for the FP32 graph and for the INT8
                        dynamically-quantized graph that the application serves.
Both share the tokenizer (WordPiece, 32k vocabulary) and the light normalisation.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .config import MAX_LEN
from .normalize import normalize


def _load_meta(directory: Path) -> dict:
    path = Path(directory) / "meta.json"
    return json.loads(path.read_text()) if path.exists() else {"threshold": 0.5}


class BertTorchModel:
    name = "BERTurk fine-tuned (FP32, PyTorch)"

    def __init__(self, directory: Path, num_threads: int | None = None):
        if num_threads:
            torch.set_num_threads(num_threads)
        self.tokenizer = AutoTokenizer.from_pretrained(directory)
        self.model = AutoModelForSequenceClassification.from_pretrained(directory).eval()
        self.threshold = _load_meta(directory)["threshold"]

    @torch.no_grad()
    def predict_proba(self, texts, batch_size: int = 64) -> np.ndarray:
        texts = [normalize(t, "light") for t in texts]
        out = []
        for i in range(0, len(texts), batch_size):
            enc = self.tokenizer(texts[i:i + batch_size], truncation=True, max_length=MAX_LEN,
                                 padding=True, return_tensors="pt")
            out.append(torch.softmax(self.model(**enc).logits, -1)[:, 1].numpy())
        return np.concatenate(out)


class BertOnnxModel:
    def __init__(self, onnx_path: Path, tokenizer_dir: Path, name: str,
                 num_threads: int | None = None):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if num_threads:
            opts.intra_op_num_threads = num_threads
            opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(onnx_path), opts, providers=["CPUExecutionProvider"])
        self.input_names = {i.name for i in self.session.get_inputs()}
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
        self.threshold = _load_meta(tokenizer_dir)["threshold"]
        self.name = name

    def predict_proba(self, texts, batch_size: int = 64) -> np.ndarray:
        texts = [normalize(t, "light") for t in texts]
        out = []
        for i in range(0, len(texts), batch_size):
            enc = self.tokenizer(texts[i:i + batch_size], truncation=True, max_length=MAX_LEN,
                                 padding=True, return_tensors="np")
            feeds = {k: v.astype(np.int64) for k, v in enc.items() if k in self.input_names}
            logits = self.session.run(["logits"], feeds)[0]
            e = np.exp(logits - logits.max(-1, keepdims=True))
            out.append((e / e.sum(-1, keepdims=True))[:, 1])
        return np.concatenate(out)
