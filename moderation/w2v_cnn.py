"""Neural model: 1D convolutional network over our own word2vec (technique area A).

* The embedding layer is initialised with the word2vec vectors trained by
  scripts/03_train_w2v_ffn.py on the training tweets and fine-tuned.
* Three 1D convolutions (kernel widths 2, 3, 4 words, 128 filters each, ReLU)
  slide over the sequence of word vectors; global max-pooling over time keeps
  the strongest n-gram detector response; dropout + a linear layer classify.
* Trained with mini-batch SGD + momentum and backpropagation.
Unlike the averaged-embedding FFN, the convolutions see word order.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .normalize import normalize, tokenize

MAX_WORDS = 60   # the longest training tweet has 51 tokens after normalisation


class TextCNN(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int = 100, n_filters: int = 128,
                 kernel_sizes=(2, 3, 4), dropout: float = 0.5):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv1d(emb_dim, n_filters, k, padding=k // 2) for k in kernel_sizes])
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(n_filters * len(kernel_sizes), 2)

    def forward(self, ids):                       # ids: (batch, words)
        e = self.emb(ids).transpose(1, 2)          # (batch, emb_dim, words)
        pooled = [torch.relu(conv(e)).max(dim=-1).values for conv in self.convs]   # max-pool over time
        return self.fc(self.drop(torch.cat(pooled, dim=1)))


def build_vocab(kv) -> tuple[dict[str, int], np.ndarray]:
    """Index 0 = padding / unknown; row i of the matrix = word2vec vector of word i."""
    vocab = {w: i + 1 for i, w in enumerate(kv.index_to_key)}
    matrix = np.vstack([np.zeros((1, kv.vector_size), np.float32), kv.vectors]).astype(np.float32)
    return vocab, matrix


def encode(token_lists, vocab: dict[str, int]) -> torch.Tensor:
    lengths = [min(max(1, sum(t in vocab for t in toks)), MAX_WORDS) for toks in token_lists]
    out = np.zeros((len(token_lists), max(max(lengths), 4)), dtype=np.int64)
    for i, toks in enumerate(token_lists):
        ids = [vocab[t] for t in toks if t in vocab][:MAX_WORDS]
        out[i, :len(ids)] = ids
    return torch.from_numpy(out)


class W2VCNNModel:
    name = "word2vec + 1D CNN"

    def __init__(self, vocab: dict[str, int], net: TextCNN, threshold: float = 0.5):
        self.vocab, self.net, self.threshold = vocab, net.eval(), threshold

    @staticmethod
    def tokens(texts):
        return [tokenize(normalize(t)) for t in texts]

    @torch.no_grad()
    def predict_proba(self, texts, batch_size: int = 256) -> np.ndarray:
        toks = self.tokens(texts)
        out = []
        for i in range(0, len(toks), batch_size):
            out.append(torch.softmax(self.net(encode(toks[i:i + batch_size], self.vocab)), -1)[:, 1].numpy())
        return np.concatenate(out)

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(self.net.state_dict(), directory / "cnn.pt")
        (directory / "vocab.json").write_text(json.dumps(self.vocab, ensure_ascii=False), encoding="utf-8")
        (directory / "meta.json").write_text(json.dumps({"threshold": self.threshold}))

    @classmethod
    def load(cls, directory: Path) -> "W2VCNNModel":
        vocab = json.loads((directory / "vocab.json").read_text(encoding="utf-8"))
        net = TextCNN(len(vocab) + 1)
        net.load_state_dict(torch.load(directory / "cnn.pt", map_location="cpu"))
        meta = json.loads((directory / "meta.json").read_text())
        return cls(vocab, net, meta["threshold"])
