"""Regex-based normalisation of noisy Turkish social-media text (technique area B).

Two levels are provided:

* ``normalize(text, level="full")`` - for the classical models (lexicon, tf-idf,
  word2vec): Turkish-aware lower-casing, leetspeak decoding, ASCII folding of
  Turkish characters, repeated-letter collapsing, joining of spaced-out letters.
* ``normalize(text, level="light")`` - for BERTurk, which is cased and has its
  own WordPiece tokenizer: only removes noise (mentions, URLs, '#', repeated
  letters, spaced-out letters, ALL-CAPS shouting) and keeps Turkish characters.

Every rule below was motivated by patterns observed in the OffensEval-TR
training tweets (see the README / slides for examples).
"""
from __future__ import annotations

import re
import unicodedata

LETTER = "a-zA-ZçğıöşüÇĞİÖŞÜâîûÂÎÛ"

# @USER mentions (the corpus anonymises user names as @USER) and URLs.
_MENTION = re.compile(r"@\w+")
_URL = re.compile(r"(https?://\S+|www\.\S+|\bURL\b)")
# '#İyiOyunlar2026' -> 'İyi Oyunlar 2026'  (CamelCase / digit boundaries).
_HASHTAG = re.compile(r"#(\w+)")
_CAMEL = re.compile(rf"(?<=[a-zçğıöşü])(?=[A-ZÇĞİÖŞÜ])|(?<=[{LETTER}])(?=\d)|(?<=\d)(?=[{LETTER}])")
# 'L Ü T F E N' / 's.a.l.a.k' / 'a-p-t-a-l' -> 'LÜTFEN' / 'salak' / 'aptal'
_SPACED = re.compile(rf"(?<![{LETTER}])(?:[{LETTER}][ .\-_]){{2,}}[{LETTER}](?![{LETTER}])")
# three or more identical letters -> one ('çoooook' -> 'çok', 'yaaa' -> 'ya')
_REPEAT = re.compile(rf"([{LETTER}])\1{{2,}}")
# leetspeak symbols that sit *between* two letters: 'gelm3ye' -> 'gelmeye'
_LEET_MAP = {"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s"}
_LEET = re.compile(rf"(?<=[{LETTER}])[013457@$]+(?=[{LETTER}])")
_ALLCAPS = re.compile(r"\b[A-ZÇĞİÖŞÜÂÎÛ]{2,}\b")
_WS = re.compile(r"\s+")

_TR_LOWER = str.maketrans({"I": "ı", "İ": "i"})
_ASCII_FOLD = str.maketrans("çğıöşüâîû", "cgiosuaiu")


def turkish_lower(text: str) -> str:
    """Lower-case with Turkish dotted/dotless i rules (I -> ı, İ -> i)."""
    return text.translate(_TR_LOWER).lower()


def _join_spaced(match: re.Match) -> str:
    return re.sub(r"[ .\-_]", "", match.group(0))


def normalize(text: str, level: str = "full") -> str:
    if level not in {"full", "light"}:
        raise ValueError(level)
    text = unicodedata.normalize("NFC", str(text))
    text = _URL.sub(" ", text)
    text = _MENTION.sub(" ", text)
    text = _HASHTAG.sub(lambda m: " " + _CAMEL.sub(" ", m.group(1)) + " ", text)
    text = _SPACED.sub(_join_spaced, text)
    text = _REPEAT.sub(r"\1", text)
    if level == "full":
        text = _LEET.sub(lambda m: "".join(_LEET_MAP[c] for c in m.group(0)), text)
        text = turkish_lower(text).translate(_ASCII_FOLD)
    else:
        # shouting ('LÜTFEN') is split into rare WordPieces by the cased BERTurk
        # vocabulary, so all-caps words are lower-cased (Turkish-aware).
        text = _ALLCAPS.sub(lambda m: turkish_lower(m.group(0)), text)
    return _WS.sub(" ", text).strip()


# Tokens for the classical models: words (keeping '*' so that censored words
# such as 'ap*al' stay one token) and emoji / symbols are dropped.
_TOKEN = re.compile(r"[\w*]+")


def tokenize(normalized_text: str) -> list[str]:
    return [t for t in _TOKEN.findall(normalized_text) if not t.isdigit()]
