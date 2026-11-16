"""Step 8 - error analysis on the test predictions written by 06_benchmark.py.

Writes results/error_analysis.md with
* false positives / false negatives of the served INT8 model,
* messages the INT8 BERTurk gets right but the tf-idf baseline gets wrong (and vice versa),
* messages where quantization changed the decision (FP32 vs INT8),
* the share of errors per normalisation phenomenon.

The tables quote a few test tweets as examples. Tweets with strong profanity, sexual, political or religious
content are skipped in these examples (SHOW_FILTER); every count still includes all 3,528 tweets.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from moderation.config import RESULTS_DIR  # noqa: E402
from moderation.data import load_split  # noqa: E402
from moderation.registry import SPECS, load_model  # noqa: E402

N_SHOW = 8
# display filter for the example tables only (it does not change any count)
SHOW_FILTER = re.compile(
    r"\bs[i1ı*.]+k|\bamk|\bam[ıi]n|orospu|oruspu|orspu|\boç\b|\bpi[çc]|\bg[öo]t|yarr|\bibne|pezevenk|kahpe|[şs]erefsiz|"
    r"anas[ıi]n[ıi]|geri ?ze|kancık|fahişe|\bfuck|"
    r"\bmeme|dekolte|\bsex|seks|porno|tecavüz|zevk|sapkın|namus|cinsel|olgun|erkeksi|"
    r"erdo[ğg]an|\brte\b|tayyip|binali|atatürk|mustafa kemal|\b(?:chp|akp|hdp|mhp|pkk|fet[öo]|ysk)\b|temmuz|\breis|"
    r"kılıçdaroğlu|bahçeli|imamoğlu|muhal|belediye|\beyt|\boy\b|rüşvet|hırsız|kürt|ermeni|suriyeli|\barap|yahudi|"
    r"putin|\brus\b|amerika|israil|\bçin|afrika|allah|peygamber|\bdin\b|islam|müslüman|sarık|\bşal|türkçe|terör|şehit|vatan|"
    r"başkan|\bbakan|meclis|darbe|seçim|\bparti|hükümet|devlet|millet|cumhur|siyanür|öldür|dövmek|"
    r"diren|dolar|ekonomi|enflasyon|\bzam\b|emekli|asker|polis|\bordu\b|savaş|hizaya|bayrak|\bhalk|türkiye|ülke|çocuklarımız|eşinizin|kızınızın|k\*le",
    re.IGNORECASE)


def main() -> None:
    df = pd.read_csv(RESULTS_DIR / "test_predictions.csv", keep_default_na=False)
    test = load_split("test")[["id", "text"]]              # the tweet text is not stored in results/
    df = df.merge(test, on="id", how="left", validate="one_to_one")
    assert df.text.notna().all() and len(df) == len(test)
    thr = {k: load_model(k).threshold for k in SPECS if f"p_{k}" in df}
    for k, t in thr.items():
        df[f"y_{k}"] = (df[f"p_{k}"] >= t).astype(int)

    def show(sub: pd.DataFrame, cols) -> list[str]:
        out = ["| tweet | gold | " + " | ".join(cols) + " |", "|---|---|" + "---|" * len(cols)]
        sub = sub[~sub.text.str.contains(SHOW_FILTER)]
        for _, r in sub.head(N_SHOW).iterrows():
            text = r["text"].replace("|", "/").replace("\n", " ")[:160]
            vals = " | ".join(f"{r[f'p_{c}']:.2f}" for c in cols)
            out.append(f"| {text} | {'OFF' if r['label'] else 'NOT'} | {vals} |")
        return out

    s = "bert_int8"
    fp = df[(df.label == 0) & (df[f"y_{s}"] == 1)].sort_values(f"p_{s}", ascending=False)
    fn = df[(df.label == 1) & (df[f"y_{s}"] == 0)].sort_values(f"p_{s}")
    bert_wins = df[(df[f"y_{s}"] == df.label) & (df["y_tfidf"] != df.label)]
    tfidf_wins = df[(df[f"y_{s}"] != df.label) & (df["y_tfidf"] == df.label)]
    quant_flip = df[df["y_bert_fp32"] != df[f"y_{s}"]] if "y_bert_fp32" in df else df.iloc[:0]
    cols = [c for c in ["lexicon", "tfidf", "w2v_cnn", "bert_fp32", s] if f"p_{c}" in df]

    lines = ["# Error analysis (official test set, n = 3,528)", "",
             f"Served model: INT8 BERTurk, threshold {thr[s]:.2f}. Values in the tables are P(offensive).", "",
             "Example tweets with strong profanity, sexual, political or religious content are not shown in the "
             "tables. All counts include every test tweet.", "",
             f"* False positives: **{len(fp)}**, false negatives: **{len(fn)}**",
             f"* INT8 BERTurk right & tf-idf wrong: **{len(bert_wins)}**; tf-idf right & INT8 BERTurk wrong: "
             f"**{len(tfidf_wins)}**",
             f"* Decisions changed by quantization (FP32 vs INT8): **{len(quant_flip)}** "
             f"({100 * len(quant_flip) / len(df):.2f} %)", ""]
    for title, sub in [("False positives of INT8 BERTurk (most confident first)", fp),
                       ("False negatives of INT8 BERTurk (least confident first)", fn),
                       ("INT8 BERTurk right, tf-idf wrong", bert_wins),
                       ("tf-idf right, INT8 BERTurk wrong", tfidf_wins),
                       ("Decisions changed by quantization", quant_flip)]:
        lines += [f"## {title}", ""] + show(sub, cols) + [""]

    # errors by phenomenon
    phenomena = {
        "contains @USER mention": df.text.str.contains("@USER"),
        "contains '*' censoring": df.text.str.contains(r"\w\*+\w", regex=True),
        "repeated letters (3+)": df.text.str.contains(r"(\w)\1\1", regex=True),
        "very short (<= 5 words)": df.text.str.split().str.len() <= 5,
    }
    lines += ["## Error rate of INT8 BERTurk by phenomenon", "", "| subset | n | error rate |", "|---|---|---|"]
    err = df[f"y_{s}"] != df.label
    lines.append(f"| all tweets | {len(df)} | {100 * err.mean():.1f} % |")
    for name, mask in phenomena.items():
        lines.append(f"| {name} | {int(mask.sum())} | {100 * err[mask].mean():.1f} % |")
    (RESULTS_DIR / "error_analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    summary = {"false_positives": len(fp), "false_negatives": len(fn), "bert_wins": len(bert_wins),
               "tfidf_wins": len(tfidf_wins), "quantization_flips": len(quant_flip)}
    (RESULTS_DIR / "error_analysis.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
