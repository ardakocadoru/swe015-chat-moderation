"""Rule baseline: an edit-distance lexicon matcher (technique area B).

1. ``build_lexicon`` derives an offensive-word lexicon from the *training* split:
   a token enters the lexicon if it appears in at least ``min_df`` training
   tweets and a smoothed share ``p_off`` of those tweets is offensive.
2. ``LexiconMatcher`` flags a message when one of its (normalised) tokens is
   within a small minimum edit distance of a lexicon word.  This catches
   misspellings and obfuscations that an exact word list misses
   ('salakk', 'aptall', 'ap*al' vs 'aptal').
   A '*' in the message is a wildcard (substitution cost 0).

Names of people, parties and organisations, and words for ethnic, religious or national groups and
the state, are never put in the lexicon (``is_topic_word``). They often occur in offensive tweets, but
they name a topic, not an insult, and a moderation word list must not flag a message for mentioning
them.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np

from .normalize import normalize, tokenize


def edit_distance(a: str, b: str, max_dist: int | None = None) -> int:
    """Levenshtein minimum edit distance (insert = delete = substitute = 1).

    Classic dynamic programme over a (len(a)+1) x (len(b)+1) table, kept as two
    rows. '*' in ``a`` matches any character. If ``max_dist`` is given the
    computation stops early once every cell in a row exceeds it.
    """
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        cur = [i] + [0] * len(b)
        ca = a[i - 1]
        for j in range(1, len(b) + 1):
            sub = 0 if (ca == b[j - 1] or ca == "*") else 1
            cur[j] = min(prev[j] + 1,        # deletion
                         cur[j - 1] + 1,     # insertion
                         prev[j - 1] + sub)  # substitution / match
        if max_dist is not None and min(cur) > max_dist:
            return max_dist + 1
        prev = cur
    return prev[-1]


def allowed_distance(token: str) -> int:
    """Short words must match exactly; longer words may differ by 1 or 2 edits."""
    n = len(token)
    if n < 5:
        return 0
    if n < 9:
        return 1
    return 2


# topic words that never enter the lexicon (tokens are normalised: lower-case, ASCII-folded)
TOPIC_STEMS = (
    "erdogan", "tayyip", "rte", "trump", "putin", "obama", "ataturk", "kilicdaroglu", "bahceli", "imamoglu",
    "akp", "chp", "mhp", "hdp", "pkk", "feto", "isid", "parti", "secim", "meclis", "hukumet", "bakan",
    "baskan", "cumhur", "devlet", "millet", "ulke", "vatan", "sehit", "asker", "teror", "darbe", "hain",
    "turk", "kurt", "arap", "suriye", "ermeni", "yahudi", "rum", "amerika", "abd", "israil", "rus", "cin",
    "allah", "islam", "muslum", "haram", "kafir", "ateist", "gavur",
    "kemal", "oylar", "oyla", "diktator", "emperyal", "firavun", "kumpas", "devsirme", "solcu", "sagci",
    "komunist", "fasist", "libos", "yobaz", "dinci", "seriat", "laik", "alevi", "sunni", "turban", "basortu",
    "carsaf", "kahhar", "kahret", "mubah", "begenmiyorsan",
)


def is_topic_word(tok: str) -> bool:
    return tok.startswith(TOPIC_STEMS)


def build_lexicon(texts, labels, min_df: int = 5, min_p_off: float = 0.7,
                  prior: float = 2.0) -> dict[str, float]:
    """Return {word: smoothed P(offensive | word in tweet)} for frequent, strongly
    offensive tokens of length >= 3."""
    labels = np.asarray(labels)
    base_rate = labels.mean()
    df, off_df = Counter(), Counter()
    for text, y in zip(texts, labels):
        toks = set(tokenize(normalize(text)))
        df.update(toks)
        if y == 1:
            off_df.update(toks)
    lexicon = {}
    for tok, n in df.items():
        if n < min_df or len(tok) < 3 or is_topic_word(tok):
            continue
        p_off = (off_df[tok] + prior * base_rate) / (n + prior)  # additive smoothing toward the base rate
        if p_off >= min_p_off:
            lexicon[tok] = round(float(p_off), 4)
    return lexicon


class LexiconMatcher:
    name = "Rule: edit-distance lexicon"

    def __init__(self, lexicon: dict[str, float]):
        self.lexicon = lexicon
        self._by_len: dict[int, list[str]] = defaultdict(list)
        for w in lexicon:
            self._by_len[len(w)].append(w)
        self._match_token = lru_cache(maxsize=200_000)(self._match_token_uncached)

    def _match_token_uncached(self, tok: str) -> tuple[str, float] | None:
        if tok in self.lexicon:
            return tok, self.lexicon[tok]
        k = allowed_distance(tok)
        if k == 0 and "*" not in tok:
            return None
        k = max(k, tok.count("*"))  # a censored word may still need its edits
        best = None
        for length in range(len(tok) - k, len(tok) + k + 1):
            for w in self._by_len.get(length, ()):
                if w[0] != tok[0] and tok[0] != "*":
                    continue
                d = edit_distance(tok, w, max_dist=k)
                if d <= k and (best is None or self.lexicon[w] > best[1]):
                    best = (w, self.lexicon[w])
        return best

    def matches(self, text: str) -> list[tuple[str, str, float]]:
        """[(message_token, lexicon_word, p_off), ...] for every matched token."""
        out = []
        for tok in tokenize(normalize(text)):
            m = self._match_token(tok)
            if m:
                out.append((tok, m[0], m[1]))
        return out

    def predict_proba(self, texts) -> np.ndarray:
        """P(offensive): the strongest matched word's p_off, else 0."""
        return np.array([max((p for _, _, p in self.matches(t)), default=0.0) for t in texts])

    # --- persistence ---------------------------------------------------------
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.lexicon, ensure_ascii=False, indent=0), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "LexiconMatcher":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))
