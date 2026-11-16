"""Step 1 - download OffensEval-TR 2020 and write the fixed splits + statistics."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from moderation.config import RESULTS_DIR  # noqa: E402
from moderation.data import prepare  # noqa: E402


def main() -> None:
    splits = prepare()
    stats = {}
    for name, df in splits.items():
        words = df["text"].str.split().str.len()
        stats[name] = {
            "n": int(len(df)),
            "offensive": int(df["label"].sum()),
            "offensive_pct": round(100 * df["label"].mean(), 2),
            "mean_words": round(float(words.mean()), 1),
            "p95_words": int(words.quantile(0.95)),
            "with_user_mention_pct": round(100 * df["text"].str.contains("@USER").mean(), 1),
            "with_repeated_letters_pct": round(100 * df["text"].str.contains(r"(\w)\1\1", regex=True).mean(), 1),
        }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "data_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
