"""Download OffensEval-TR 2020 and create fixed train / validation / test splits.

* test  = the official OffensEval-TR 2020 test set (3,528 tweets, gold labels
          from offenseval-tr-labela-v1.tsv). It is never used for training or
          model selection and is the single held-out set for every model.
* train / validation = a stratified 90 / 10 split of the official training
          file (31,756 tweets) with a fixed seed.
"""
from __future__ import annotations

import csv
import io
import urllib.request
import zipfile

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import (LABELS, OFFENSEVAL_URL, PROCESSED_DIR, RAW_DIR, SEED,
                     VAL_FRACTION)

_ROOT = RAW_DIR / "offenseval2020-turkish"
_TRAIN_TSV = _ROOT / "offenseval-tr-training-v1" / "offenseval-tr-training-v1.tsv"
_TEST_TSV = _ROOT / "offenseval-tr-testset-v1" / "offenseval-tr-testset-v1.tsv"
_TEST_LABELS = _ROOT / "offenseval-tr-testset-v1" / "offenseval-tr-labela-v1.tsv"


def download() -> None:
    """Download and unzip the corpus once (about 2 MB)."""
    if _TRAIN_TSV.exists():
        return
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {OFFENSEVAL_URL} ...")
    with urllib.request.urlopen(OFFENSEVAL_URL) as resp:
        payload = resp.read()
    zipfile.ZipFile(io.BytesIO(payload)).extractall(RAW_DIR)


def _read_tsv(path) -> pd.DataFrame:
    # Tweets contain quote characters, so CSV quoting must be disabled.
    return pd.read_csv(path, sep="\t", quoting=csv.QUOTE_NONE, dtype={"id": int},
                       keep_default_na=False, encoding="utf-8")


def prepare() -> dict[str, pd.DataFrame]:
    """Create data/processed/{train,val,test}.csv with columns id, text, label."""
    download()
    train_full = _read_tsv(_TRAIN_TSV).rename(columns={"tweet": "text"})
    train_full["label"] = train_full["subtask_a"].map({l: i for i, l in enumerate(LABELS)})
    train_full = train_full[["id", "text", "label"]]

    test = _read_tsv(_TEST_TSV).rename(columns={"tweet": "text"})
    gold = pd.read_csv(_TEST_LABELS, header=None, names=["id", "gold"], dtype={"id": int})
    test = test.merge(gold, on="id", how="inner", validate="one_to_one")
    test["label"] = test["gold"].map({l: i for i, l in enumerate(LABELS)})
    test = test[["id", "text", "label"]]

    assert train_full["label"].notna().all() and test["label"].notna().all()
    assert not set(train_full["id"]) & set(test["id"]), "train/test overlap"

    train, val = train_test_split(train_full, test_size=VAL_FRACTION,
                                  stratify=train_full["label"], random_state=SEED)
    splits = {"train": train.reset_index(drop=True),
              "val": val.reset_index(drop=True),
              "test": test.reset_index(drop=True)}
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for name, df in splits.items():
        df.to_csv(PROCESSED_DIR / f"{name}.csv", index=False)
    return splits


def load_split(name: str) -> pd.DataFrame:
    path = PROCESSED_DIR / f"{name}.csv"
    if not path.exists():
        prepare()
    return pd.read_csv(path, keep_default_na=False)
