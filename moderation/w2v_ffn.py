"""Neural model: our own word2vec + a feedforward network (technique area A).

* word2vec (skip-gram, gensim) is trained from scratch on the *training* tweets
  only - no pretrained vectors. gensim is needed for training only; the trained
  vectors are stored as a plain NumPy matrix + word list, so the app runs
  without gensim.
* A message is represented by [mean ; max] pooling of its word vectors.
* A 2-hidden-layer feedforward network (ReLU, dropout) is trained with
  mini-batch SGD + momentum; gradients come from backpropagation (autograd).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .normalize import normalize, tokenize

EMB_DIM = 100


class WordVectors:
    """Minimal read-only word -> vector table (same attribute names as gensim's KeyedVectors)."""

    def __init__(self, index_to_key: list[str], vectors: np.ndarray):
        self.index_to_key = list(index_to_key)
        self.key_to_index = {w: i for i, w in enumerate(self.index_to_key)}
        self.vectors = np.asarray(vectors, dtype=np.float32)
        self.vector_size = self.vectors.shape[1]

    def __getitem__(self, word: str) -> np.ndarray:
        return self.vectors[self.key_to_index[word]]

    def save(self, directory: Path) -> None:
        np.save(directory / "word2vec_vectors.npy", self.vectors)
        (directory / "word2vec_vocab.json").write_text(json.dumps(self.index_to_key, ensure_ascii=False),
                                                       encoding="utf-8")

    @classmethod
    def load(cls, directory: Path) -> "WordVectors":
        vocab = json.loads((directory / "word2vec_vocab.json").read_text(encoding="utf-8"))
        return cls(vocab, np.load(directory / "word2vec_vectors.npy"))


def train_word2vec(token_lists, seed: int = 42):
    """Skip-gram word2vec on the training tweets (returns gensim KeyedVectors)."""
    from gensim.models import Word2Vec  # training-time dependency only

    w2v = Word2Vec(sentences=token_lists, vector_size=EMB_DIM, window=5, min_count=2,
                   sg=1, negative=10, epochs=15, seed=seed, workers=1)
    return w2v.wv


def embed(token_lists, kv) -> np.ndarray:
    """[mean ; max] pooled word vectors; zeros if no token is in the vocabulary."""
    out = np.zeros((len(token_lists), 2 * kv.vector_size), dtype=np.float32)
    for i, toks in enumerate(token_lists):
        vecs = [kv[t] for t in toks if t in kv.key_to_index]
        if vecs:
            m = np.stack(vecs)
            out[i, :kv.vector_size] = m.mean(0)
            out[i, kv.vector_size:] = m.max(0)
    return out


class FeedForward(nn.Module):
    def __init__(self, in_dim: int = 2 * EMB_DIM, hidden=(512, 128), dropout: float = 0.5):
        super().__init__()
        # feature standardisation (mean / std of the training features), stored with the model
        self.register_buffer("mu", torch.zeros(in_dim))
        self.register_buffer("sd", torch.ones(in_dim))
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden[0]), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden[0], hidden[1]), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden[1], 2),
        )

    def forward(self, x):
        return self.net((x - self.mu) / self.sd)


class W2VFFNModel:
    name = "word2vec + Feedforward NN"

    def __init__(self, kv, net: FeedForward, threshold: float = 0.5):
        if not isinstance(kv, WordVectors):          # gensim KeyedVectors right after training
            kv = WordVectors(kv.index_to_key, kv.vectors)
        self.kv, self.net, self.threshold = kv, net.eval(), threshold

    @staticmethod
    def tokens(texts):
        return [tokenize(normalize(t)) for t in texts]

    @torch.no_grad()
    def predict_proba(self, texts) -> np.ndarray:
        x = torch.from_numpy(embed(self.tokens(texts), self.kv))
        return torch.softmax(self.net(x), dim=-1)[:, 1].numpy()

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.kv.save(directory)
        torch.save(self.net.state_dict(), directory / "ffn.pt")
        (directory / "meta.json").write_text(json.dumps({"threshold": self.threshold}))

    @classmethod
    def load(cls, directory: Path) -> "W2VFFNModel":
        net = FeedForward()
        net.load_state_dict(torch.load(directory / "ffn.pt", map_location="cpu"))
        meta = json.loads((directory / "meta.json").read_text())
        return cls(WordVectors.load(directory), net, meta["threshold"])
