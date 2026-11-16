"""Step 2 - classical baselines: edit-distance lexicon matcher and tf-idf + LR.

Hyper-parameters are chosen on the validation split only; the test split is
used only for the final evaluation in scripts/06_benchmark.py.
"""
import itertools
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from moderation.config import LEXICON_PATH, RESULTS_DIR, TFIDF_PATH  # noqa: E402
from moderation.data import load_split  # noqa: E402
from moderation.lexicon import LexiconMatcher, build_lexicon  # noqa: E402
from moderation.metrics import best_threshold, classification_metrics  # noqa: E402
from moderation.tfidf_model import TfidfModel, build_pipeline  # noqa: E402


def train_lexicon(train, val) -> dict:
    log = []
    best = None
    for min_df, min_p in itertools.product([3, 5, 10, 20], [0.5, 0.6, 0.7, 0.8, 0.9]):
        lex = build_lexicon(train.text, train.label, min_df=min_df, min_p_off=min_p)
        matcher = LexiconMatcher(lex)
        m = classification_metrics(val.label, matcher.predict_proba(val.text), threshold=1e-9)
        log.append({"min_df": min_df, "min_p_off": min_p, "size": len(lex), "val_macro_f1": m["macro_f1"]})
        print(f"lexicon min_df={min_df:<3} min_p_off={min_p:.1f} size={len(lex):<5} val macro-F1={m['macro_f1']:.4f}")
        if best is None or m["macro_f1"] > best[0]:
            best = (m["macro_f1"], matcher, min_df, min_p)
    _, matcher, min_df, min_p = best
    matcher.save(LEXICON_PATH)
    return {"chosen": {"min_df": min_df, "min_p_off": min_p, "size": len(matcher.lexicon),
                       "val_macro_f1": best[0]}, "grid": log}


def train_tfidf(train, val) -> dict:
    log = []
    best = None
    for C in [0.5, 1, 2, 4, 8, 16]:
        t0 = time.time()
        pipe = build_pipeline(C=C).fit(train.text, train.label)
        p = pipe.predict_proba(val.text)[:, 1]
        thr = best_threshold(val.label, p)
        m = classification_metrics(val.label, p, thr)
        log.append({"C": C, "threshold": thr, "val_macro_f1": m["macro_f1"]})
        print(f"tf-idf C={C:<4} thr={thr:.2f} val macro-F1={m['macro_f1']:.4f} ({time.time() - t0:.0f}s)")
        if best is None or m["macro_f1"] > best[0]:
            best = (m["macro_f1"], pipe, C, thr)
    _, pipe, C, thr = best
    TfidfModel(pipe, thr).save(TFIDF_PATH)
    return {"chosen": {"C": C, "threshold": thr, "val_macro_f1": best[0]}, "grid": log}


def main() -> None:
    train, val = load_split("train"), load_split("val")
    if "--lexicon-only" in sys.argv:         # rebuild only the lexicon, keep the saved tf-idf model and report
        report = json.loads((RESULTS_DIR / "train_classical.json").read_text())
        report["lexicon"] = train_lexicon(train, val)
    else:
        report = {"lexicon": train_lexicon(train, val), "tfidf": train_tfidf(train, val)}
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "train_classical.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v["chosen"] for k, v in report.items()}, indent=2))


if __name__ == "__main__":
    main()
