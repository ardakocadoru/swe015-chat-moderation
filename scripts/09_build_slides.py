"""Step 9 - build the presentation: figures (docs/fig/*.svg), docs/slides.html and docs/slides.pdf.

Every number and every chart is generated from results/*.json written by the
pipeline, so the slides cannot drift from the measurements. Name, student ID
and links come from docs/slides_config.json. The PDF is printed with Edge/Chrome.
"""
import html
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from moderation.config import BERT_FP32_DIR, RESULTS_DIR, ROOT  # noqa: E402
from moderation.data import load_split  # noqa: E402
from moderation.registry import SPECS  # noqa: E402

DOCS = ROOT / "docs"
FIG = DOCS / "fig"
TOTAL = 12
INK, MUTE, LINE, ACC, CLEAN, SOFT = "#16181d", "#5b616e", "#e3e5e8", "#e8452c", "#2f7d6d", "#f5f5f3"
LADDER = [("tfidf", "1", "tf-idf + LR"), ("w2v_cnn", "2", "word2vec + 1D CNN"), ("bert_fp32", "3", "BERTurk FP32"),
          ("bert_onnx_fp32", "3b", "BERTurk FP32, ONNX"), ("bert_int8", "4", "BERTurk INT8"),
          ("lexicon", "+", "Lexicon (rules)"), ("w2v_ffn", "+", "word2vec + FFN")]


def R(name):
    return json.loads((RESULTS_DIR / name).read_text(encoding="utf-8"))


def esc(s) -> str:
    return html.escape(str(s))


def link(url: str) -> str:
    """Clickable link; the visible text drops the scheme and any query string, long ones may wrap."""
    url = str(url).strip()
    if not url.startswith(("https://", "http://")):        # placeholder text, not a link
        return esc(url)
    shown = url.removeprefix("https://").removeprefix("http://").removeprefix("www.").split("?")[0]
    shown = shown.removesuffix("/view").rstrip("/")
    cls = ' class="long"' if len(shown) > 50 else ""
    return f'<a{cls} href="{esc(url)}">{esc(shown)}</a>'


def bert_params_m():
    """Parameter count of our fine-tuned checkpoint (counted, not copied from a model card)."""
    try:
        from safetensors import safe_open
        n = 0
        with safe_open(str(BERT_FP32_DIR / "model.safetensors"), "np") as f:
            for k in f.keys():
                size = 1
                for d in f.get_slice(k).get_shape():
                    size *= d
                n += size
        return n / 1e6
    except Exception:
        return None


# --------------------------------------------------------------------------- figures
def make_figures(M, st, q):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "Segoe UI", "font.size": 12, "svg.fonttype": "none",
        "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#c9ccd1",
        "axes.labelcolor": MUTE, "xtick.color": MUTE, "ytick.color": INK, "axes.titlesize": 13,
        "figure.facecolor": "white", "axes.facecolor": "white",
    })

    def color(k):
        if k == "bert_int8":
            return ACC
        return "#b8bcc4" if k in ("lexicon", "w2v_ffn") else "#3a3f48"

    keys = [k for k, _, _ in LADDER]
    labels = [f"{s}   {n}" for _, s, n in LADDER]
    y = list(range(len(keys)))[::-1]

    # macro-F1
    fig, ax = plt.subplots(figsize=(6.0, 2.9))
    vals = [M[k]["macro_f1"] for k in keys]
    ax.barh(y, vals, color=[color(k) for k in keys], height=0.62)
    for yi, v, k in zip(y, vals, keys):
        ax.text(v + 0.004, yi, f"{v:.3f}", va="center", fontsize=12, color=ACC if k == "bert_int8" else INK,
                fontweight="bold" if k == "bert_int8" else "normal")
    ax.set_yticks(y, labels)
    ax.set_xlim(0.5, 0.87)
    ax.set_xlabel("Macro-F1 on the test set")
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    fig.savefig(FIG / "f1.svg")
    plt.close(fig)

    # latency
    fig, ax = plt.subplots(figsize=(6.0, 2.9))
    means = [M[k]["latency"]["1_thread"]["mean_ms"] for k in keys]
    p95 = [M[k]["latency"]["1_thread"]["p95_ms"] for k in keys]
    ax.barh(y, means, color=[color(k) for k in keys], height=0.62)
    ax.plot(p95, y, linestyle="none", marker="|", markersize=15, markeredgewidth=2, color=INK)
    for yi, v, p, k in zip(y, means, p95, keys):
        ax.text(max(v, p) * 1.15, yi, f"{v:.1f} ms", va="center", fontsize=12, color=ACC if k == "bert_int8" else INK,
                fontweight="bold" if k == "bert_int8" else "normal")
    ax.axvline(50, color=ACC, linestyle="--", linewidth=1.4)
    ax.text(50 * 1.06, y[0] + 0.55, "50 ms budget", color=ACC, fontsize=11)
    ax.set_xscale("log")
    ax.set_xlim(0.1, 900)
    ax.set_yticks(y, labels)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Mean ms per message, 1 thread, log scale;  tick = p95")
    fig.tight_layout()
    fig.savefig(FIG / "latency.svg")
    plt.close(fig)

    # quantization variants
    v = q["variants"]
    cats = [("FP32\nPyTorch", M["bert_fp32"]["size_mb"], q["val_macro_f1_fp32"], "#3a3f48"),
            ("FP32\nONNX", q["fp32_onnx_size_mb"], q["val_macro_f1_fp32"], "#3a3f48"),
            ("INT8\nMatMul", v["matmul"]["size_mb"], v["matmul"]["val_macro_f1"], "#b8bcc4"),
            ("INT8\nper-chan.", v["matmul_per_channel"]["size_mb"], v["matmul_per_channel"]["val_macro_f1"], "#ffffff"),
            ("INT8\npre-proc.", v["pre+matmul"]["size_mb"], v["pre+matmul"]["val_macro_f1"], "#b8bcc4"),
            ("INT8\n+ embed.", v["pre+matmul+embeddings"]["size_mb"], v["pre+matmul+embeddings"]["val_macro_f1"], ACC)]
    fig, ax = plt.subplots(figsize=(6.2, 4.1))
    x = list(range(len(cats)))
    bars = ax.bar(x, [c[1] for c in cats], color=[c[3] for c in cats], width=0.62,
                  edgecolor=[INK if c[3] == "#ffffff" else c[3] for c in cats], linewidth=1.2)
    bars[3].set_hatch("///")
    for xi, c in zip(x, cats):
        broken = c[2] < 0.6
        ax.text(xi, c[1] + 10, f"{c[1]:.0f} MB", ha="center", fontsize=12, color=INK, fontweight="bold")
        ax.text(xi, c[1] + 40, f"F1 {c[2]:.3f}", ha="center", fontsize=10.5, color=ACC if broken else MUTE)
    ax.set_xticks(x, [c[0] for c in cats], fontsize=10.5)
    ax.set_ylim(0, 500)
    ax.set_ylabel("Model file (MB)")
    ax.tick_params(axis="x", length=0)
    fig.tight_layout()
    fig.savefig(FIG / "quant.svg")
    plt.close(fig)

    # data splits
    fig, ax = plt.subplots(figsize=(5.6, 1.9))
    rows = [("test", st["test"]), ("val", st["val"]), ("train", st["train"])]
    for i, (name, s) in enumerate(rows):
        off = s["offensive"]
        ax.barh(i, s["n"] - off, color="#c9ccd1", height=0.6)
        ax.barh(i, off, left=s["n"] - off, color=ACC, height=0.6)
        ax.text(s["n"] + 400, i, f"{s['n']:,}  ({s['offensive_pct']:.0f}% OFF)", va="center", fontsize=11, color=INK)
    ax.set_yticks(range(3), [r[0] for r in rows])
    ax.set_xlim(0, 42000)
    ax.tick_params(axis="y", length=0)
    ax.spines["bottom"].set_visible(False)
    ax.set_xticks([])
    fig.tight_layout()
    fig.savefig(FIG / "splits.svg")
    plt.close(fig)


# --------------------------------------------------------------------------- slides
SECTIONS = ["Introduction", "Development", "Conclusion"]


def page(n, section, title, body, cfg):
    nav = "".join(f'<span class="{"on" if s == section else ""}">{s}</span>' for s in SECTIONS)
    return f"""
<section class="slide">
  <div class="top"><div class="nav">{nav}</div><div class="pno">{n:02d} / {TOTAL}</div></div>
  <h1>{title}</h1>
  <div class="body">{body}</div>
  <div class="foot"><span>{esc(cfg['student_name'])}, {esc(cfg['student_id'])}</span><span>SWE015 Introduction to LLMs, homework</span></div>
</section>"""


def build() -> str:
    cfg = json.loads((DOCS / "slides_config.json").read_text(encoding="utf-8"))
    res = R("results.json")
    M = {m["key"]: m for m in res["models"]}
    st, q, ea = R("data_stats.json"), R("quantization_check.json"), R("error_analysis.json")
    tb, tc, tcnn, tffn = R("train_bert.json"), R("train_classical.json"), R("train_w2v_cnn.json"), R("train_w2v_ffn.json")
    hw = res["hardware"]
    budget = res["latency_protocol"]["budget_ms_mean_1_thread"]
    preds = pd.read_csv(RESULTS_DIR / "test_predictions.csv", keep_default_na=False)
    preds = preds.merge(load_split("test")[["id", "text"]], on="id", validate="one_to_one")  # text is not in results/
    make_figures(M, st, q)
    params = bert_params_m()
    bcfg = json.loads((BERT_FP32_DIR / "config.json").read_text())

    def lat(k, t="1_thread", s="mean_ms"):
        return M[k]["latency"][t][s]

    i8, f32, tf, cnn, ffn, lx = (M[k] for k in ["bert_int8", "bert_fp32", "tfidf", "w2v_cnn", "w2v_ffn", "lexicon"])
    n_total = st["train"]["n"] + st["val"]["n"] + st["test"]["n"]
    chosen = q["variants"][q["chosen_variant"]]
    S = []

    # 01 title -------------------------------------------------------------------------
    S.append(f"""
<section class="slide cover">
  <div class="cover-l">
    <div class="kick">SWE015 Introduction to Large Language Models<br>Homework: Building an Intelligent Application</div>
    <h1>Real-time<br>moderation of<br>Turkish game chat</h1>
    <p class="lead">A ladder of four models, from an edit-distance lexicon to an INT8-quantized BERTurk
    that flags offensive messages on a single CPU core.</p>
    <table class="who">
      <tr><td>Student</td><td><b>{esc(cfg['student_name'])}</b></td></tr>
      <tr><td>Student ID</td><td>{esc(cfg['student_id'])}</td></tr>
      <tr><td>Use case</td><td>4, Real-Time Chat Moderation</td></tr>
      <tr><td>Repository</td><td>{link(cfg['repo_url'])}</td></tr>
      <tr><td>Demo video</td><td>{link(cfg['video_url'])}</td></tr>
      <tr><td>Model weights</td><td>{link(cfg['hf_url'])}</td></tr>
    </table>
  </div>
  <div class="cover-r">
    <img src="app_screenshot_chat.png" alt="app">
    <div class="kpis">
      <div><b>{i8['macro_f1']:.3f}</b><span>macro-F1 on the<br>official test set</span></div>
      <div><b>{lat('bert_int8'):.1f}<small> ms</small></b><span>per message,<br>one CPU thread</span></div>
      <div><b>{i8['size_mb']:.0f}<small> MB</small></b><span>model size,<br>down from {f32['size_mb']:.0f} MB</span></div>
    </div>
  </div>
</section>""")

    # 02 introduction ------------------------------------------------------------------
    S.append(page(2, "Introduction", "Offensive messages must be hidden before others see them", f"""
<div class="g2 intro">
  <div>
    <p class="big-p">Most online games have an open chat, and players insult each other in it. The moderation model
    checks each message before it is shown, so its run time adds to the delay of every message. The chat server has no GPU.</p>
    <p>For <b>Use Case 4, Real-Time Chat Moderation</b>, I built an application that labels each Turkish message as clean or
    offensive, hides the offensive ones and logs how long every decision takes.</p>
    <div class="sub">Who it is for</div>
    <div class="users"><div><b>Players</b><span>see fewer insults</span></div>
      <div><b>Moderators</b><span>get fewer reports to review</span></div>
      <div><b>Platform</b><span>runs it on CPU servers, no GPU</span></div></div>
  </div>
  <div class="targets">
    <div class="sub">What success looks like</div>
    <div class="tg"><div class="no">01</div><div><b>Macro-F1</b><span>on the official OffensEval-TR test set. Only
      {st['test']['offensive_pct']:.0f}% of the tweets are offensive, so plain accuracy would be misleading.</span></div></div>
    <div class="tg"><div class="no">02</div><div><b>&le; {budget:.0f} ms</b><span>mean time per message on one CPU thread,
      including normalisation and tokenisation.</span></div></div>
    <div class="tg"><div class="no">03</div><div><b>INT8 in a live app</b><span>the quantized Transformer runs inside a chat
      interface that accepts new messages.</span></div></div>
  </div>
</div>""", cfg))

    # 03 data --------------------------------------------------------------------------
    S.append(page(3, "Development", f"{n_total:,} Turkish tweets: one in five is offensive, and the text is noisy", f"""
<div class="g2">
  <div>
    <p><b>OffensEval-TR 2020</b> (Çöltekin, LREC 2020), tweets labelled NOT or OFF, CC-BY 2.0.</p>
    <img class="fig" src="fig/splits.svg" alt="splits">
    <p class="note">Train and validation are a stratified 90/10 split of the official training file (seed 42). The
    official test set is the held-out set: hyper-parameters, early stopping, thresholds and the quantization
    variant were all chosen on validation. The test set is used only for the final evaluation, the same for every model.</p>
  </div>
  <div>
    <h3>What the regex normaliser fixes</h3>
    <table class="ba">
      <tr><td>@USER @USER çok iyi</td><td>çok iyi</td><td>mentions in {st['train']['with_user_mention_pct']:.0f}% of tweets</td></tr>
      <tr><td>çoooook güzeeel</td><td>çok güzel</td><td>repeated letters, {st['train']['with_repeated_letters_pct']:.0f}%</td></tr>
      <tr><td>L Ü T F E N</td><td>lütfen</td><td>spaced-out letters</td></tr>
      <tr><td>s4l4k</td><td>salak</td><td>leetspeak</td></tr>
      <tr><td>#İyiOyunlar</td><td>iyi oyunlar</td><td>hashtags</td></tr>
      <tr><td>IŞIK, güzel</td><td>isik, guzel</td><td>Turkish casing, ASCII</td></tr>
    </table>
    <p class="note">The classical models get the full normalisation. BERTurk is cased and has its own WordPiece
    vocabulary, so it only gets the noise removal (<code>normalize.py</code>).</p>
  </div>
</div>""", cfg))

    # 04 architecture ------------------------------------------------------------------
    svg = (DOCS / "architecture.svg").read_text(encoding="utf-8").replace(
        'viewBox="0 0 1600 840" width="1600" height="840"', 'viewBox="10 10 1580 820"')
    S.append(page(4, "Development", "One code path from the chat box to the masked message",
                  f'<div class="arch">{svg}</div>', cfg))

    # 05 technique map -----------------------------------------------------------------
    def tech(items):
        return "".join(f'<div class="ti"><b>{a}</b><span>{b}</span><code>{c}</code></div>' for a, b, c in items)
    S.append(page(5, "Development", "Technique map: which component uses which technique", f"""
<div class="tmap">
  <div class="col"><div class="L">A</div><div class="Ln">Neural networks</div>{tech([
      ("Feedforward network, SGD, backpropagation, dropout", "word2vec + FFN classifier", "w2v_ffn.py"),
      ("1D convolution and max-pooling", "word2vec + 1D CNN, ladder step 2", "w2v_cnn.py")])}</div>
  <div class="col"><div class="L">B</div><div class="Ln">Classical NLP</div>{tech([
      ("Regular expressions, tokenization, normalization", "pre-processing in front of every model", "normalize.py"),
      ("Minimum edit distance", "rule baseline that tolerates typos and censored letters", "lexicon.py"),
      ("tf-idf", "classical baseline with logistic regression, ladder step 1", "tfidf_model.py"),
      ("word2vec", "own skip-gram embeddings from the training tweets", "w2v_ffn.py")])}</div>
  <div class="col"><div class="L">C</div><div class="Ln">Transformers</div>{tech([
      ("Fine-tuning BERT with a task head", "BERTurk classifier, ladder step 3", "04_finetune_bert.py"),
      ("WordPiece subword tokenizer, attention", f"{bcfg['vocab_size'] // 1000}k-token Turkish vocabulary, {bcfg['num_hidden_layers']} attention layers", "bert_model.py")])}</div>
  <div class="col hot"><div class="L">D</div><div class="Ln">Efficient models</div>{tech([
      ("Dynamic INT8 quantization", "the model served in the app, ladder step 4", "05_export_quantize.py"),
      ("ONNX and ONNX Runtime", "graph export and CPU inference", "bert_model.py"),
      ("Benchmarking size, latency, accuracy", "one protocol for all seven models", "06_benchmark.py")])}</div>
</div>""", cfg))

    # 06 ladder ------------------------------------------------------------------------
    lxc, tfc = tc["lexicon"]["chosen"], tc["tfidf"]["chosen"]
    bert_det = (f"{params:.0f}M parameters, {bcfg['num_hidden_layers']} layers + linear head; AdamW 2e-5, 3 epochs, "
                f"{tb['train_minutes']} min on an RTX 4070") if params else "AdamW 2e-5, 3 epochs"
    steps = [
        ("1", "tf-idf + logistic regression", f"word 1-2 and character 2-5 grams, class-balanced; C={tfc['C']} and threshold tuned", tfc["val_macro_f1"]),
        ("2", "word2vec + 1D CNN", f"own 100-d skip-gram vectors, filters over 2, 3, 4 words, max-pooling; SGD, best epoch {tcnn['best_epoch']}", tcnn["val_macro_f1"]),
        ("3", "BERTurk, fine-tuned", bert_det, q["val_macro_f1_fp32"]),
        ("4", "BERTurk, INT8", "same network exported to ONNX, weights quantized to 8-bit integers; no retraining", chosen["val_macro_f1"]),
    ]
    cards = ""
    for i, (no, name, det, f1) in enumerate(steps):
        w = max(0.0, (f1 - 0.6) / 0.25) * 100
        cards += (f'<div class="step s{i}{" hot" if no == "4" else ""}"><div class="sn">{no}</div><div class="sb"><b>{name}</b>'
                  f'<span>{det}</span><div class="bar"><i style="width:{w:.0f}%"></i><em>val F1 {f1:.3f}</em></div></div></div>')
    S.append(page(6, "Development", "Four models of increasing capacity, tuned on validation only", f"""
<div class="ladder">{cards}
<div class="extras">Also trained for comparison: <b>lexicon + edit distance</b> ({lxc['size']} words mined from the training
data, names and group words excluded, val F1 {lxc['val_macro_f1']:.3f}) and <b>word2vec + feedforward network</b> (mean and max pooling, 512 and 128 units,
val F1 {tffn['val_macro_f1']:.3f}). Every decision threshold maximises validation macro-F1.</div></div>""", cfg))

    # 07 quantization ------------------------------------------------------------------
    dF1 = q["val_macro_f1_fp32"] - chosen["val_macro_f1"]
    S.append(page(7, "Development", f"INT8 quantization shrinks BERTurk from {f32['size_mb']:.0f} MB to {i8['size_mb']:.0f} MB", f"""
<div class="g2 q">
  <div>
    <ol class="steps">
      <li><b>Export.</b> <code>torch.onnx.export</code> with dynamic batch and sequence axes. The ONNX graph reproduces
        PyTorch to within {q['max_abs_diff_torch_vs_onnx_fp32']:.6f} in probability.</li>
      <li><b>Quantize.</b> Weights become 8-bit integers with a scale S and a zero point Z; activations are quantized at
        run time, so no calibration data is needed.
        <div class="formula">q = round( x / S ) + Z</div></li>
      <li><b>Select.</b> Keep the smallest of four settings that loses at most 0.01 validation macro-F1. The chosen one
        costs {dF1:.3f} ({q['val_macro_f1_fp32']:.3f} to {chosen['val_macro_f1']:.3f}).</li>
    </ol>
    <p class="note">Per-channel scales collapsed the model to one class (val F1 {q['variants']['matmul_per_channel']['val_macro_f1']:.3f}),
    so per-tensor scales are used. Quantizing the {bcfg['vocab_size'] // 1000}k &times; {bcfg['hidden_size']} embedding matrix as well saved another
    {q['variants']['pre+matmul']['size_mb'] - chosen['size_mb']:.0f} MB.</p>
  </div>
  <div><img class="fig" src="fig/quant.svg" alt="quantization variants"><p class="note">F1 above each bar is the validation macro-F1. Hatched: per-channel scales.</p></div>
</div>""", cfg))

    # 08 results table -----------------------------------------------------------------
    body = ""
    for k, step, _ in LADDER:
        m = M[k]
        size = f"{m['size_mb']:.3f}" if m["size_mb"] < 1 else f"{m['size_mb']:.1f}"
        cls = "hot" if k == "bert_int8" else ("ext" if step == "+" else "")
        body += (f'<tr class="{cls}"><td>{step}</td><td>{esc(m["model"])}</td><td>{esc(SPECS[k].precision)}, {esc(m["runtime"])}</td>'
                 f'<td class="n">{m["macro_f1"]:.3f}</td><td class="n">{size}</td><td class="n">{lat(k):.1f}</td>'
                 f'<td class="n">{lat(k, s="p95_ms"):.1f}</td><td class="n">{lat(k, "4_thread"):.1f}</td>'
                 f'<td>{"yes" if m["meets_budget"] else "no"}</td></tr>')
    S.append(page(8, "Development", f"Results on the official test set, the same {st['test']['n']:,} tweets for all", f"""
<table class="res">
  <tr><th>Step</th><th>Model</th><th>Precision, runtime</th><th class="n">Macro-F1</th><th class="n">Size (MB)</th>
      <th class="n">Latency, 1 thread<br>mean (ms)</th><th class="n">p95 (ms)</th><th class="n">4 threads<br>mean (ms)</th><th>&le; {budget:.0f} ms</th></tr>
  {body}
</table>
<p class="note">Latency: one message at a time, end to end (normalisation, tokenisation, model), mean of
{res['latency_protocol']['runs']} different test messages after {res['latency_protocol']['warmup']} warm-up messages, freshly loaded model.
Size: model files on disk. Hardware: {esc(hw['cpu'])}, {hw['ram_gb']:.0f} GB RAM, {esc(hw['os'])}, CPU only;
PyTorch {esc(hw['torch'].split('+')[0])}, ONNX Runtime {esc(hw['onnxruntime'])}. Rows marked + were trained in addition to the ladder.</p>""", cfg))

    # 09 comparison --------------------------------------------------------------------
    S.append(page(9, "Conclusion", "INT8 BERTurk has the best F1 and stays under 50 ms", f"""
<div class="g2 charts"><img class="fig" src="fig/f1.svg" alt="f1"><img class="fig" src="fig/latency.svg" alt="latency"></div>
<table class="wl">
  <tr><th>Model</th><th>Where it wins</th><th>Where it loses</th></tr>
  <tr><td>Lexicon (rules)</td><td>under 1 ms ({lat('lexicon'):.1f} ms), tiny, every decision explainable</td>
      <td>finds only {100 * lx['off_recall']:.0f}% of offensive tweets; misses insults without a listed word</td></tr>
  <tr><td>tf-idf + LR</td><td>best cheap model: {tf['macro_f1']:.3f} in {lat('tfidf'):.1f} ms; char n-grams catch typos and suffixes</td>
      <td>no context; {100 * (f32['macro_f1'] - tf['macro_f1']):.1f} F1 points below BERTurk</td></tr>
  <tr><td>word2vec + CNN / FFN</td><td>under 1 ms; the CNN sees word order ({cnn['macro_f1']:.3f} vs {ffn['macro_f1']:.3f})</td>
      <td>embeddings learned from 28k tweets are weak, so both stay below tf-idf</td></tr>
  <tr><td>BERTurk FP32</td><td>macro-F1 {f32['macro_f1']:.3f}, above every model without pre-training</td>
      <td>{f32['size_mb']:.0f} MB; {lat('bert_fp32'):.0f} ms in PyTorch misses the budget, ONNX ({lat('bert_onnx_fp32'):.0f} ms, p95 {lat('bert_onnx_fp32', s='p95_ms'):.0f}) {'just misses it' if lat('bert_onnx_fp32') > budget else 'only just meets it'}</td></tr>
  <tr class="hot"><td>BERTurk INT8</td><td>same F1 ({i8['macro_f1']:.3f}), {f32['size_mb'] / i8['size_mb']:.1f}&times; smaller, {lat('bert_fp32') / lat('bert_int8'):.1f}&times; faster ({lat('bert_int8'):.1f} ms): served in the app</td>
      <td>p95 {lat('bert_int8', s='p95_ms'):.0f} ms on one thread; still misses veiled insults</td></tr>
</table>
</div>""", cfg))

    # 10 error analysis ----------------------------------------------------------------
    def row(i):
        return preds[preds.id == i].iloc[0]

    fn, fp, nz = row(cfg["failure_fn_id"]), row(cfg["failure_fp_id"]), row(cfg["label_noise_id"])
    (tn, fpc), (fnc, tp) = i8["confusion_matrix"]
    S.append(page(10, "Conclusion", "Where it fails: veiled insults, look-alikes, noisy labels", f"""
<div class="g2 err">
  <div>
    <div class="ex"><div class="eh"><span class="tag off">missed</span> gold OFF, P(offensive) = {fn['p_bert_int8']:.2f}</div>
      <q>{esc(fn['text'])}</q><div class="en">{esc(cfg['failure_fn_note'].format(p_tfidf=fn['p_tfidf']))}</div></div>
    <div class="ex"><div class="eh"><span class="tag fa">false alarm</span> gold NOT, P(offensive) = {fp['p_bert_int8']:.2f}</div>
      <q>{esc(fp['text'])}</q><div class="en">{esc(cfg['failure_fp_note'])}</div></div>
    <div class="ex"><div class="eh"><span class="tag nz">label?</span> gold {'OFF' if nz['label'] else 'NOT'}, P(offensive) = {nz['p_bert_int8']:.2f}</div>
      <q>{esc(nz['text'])}</q><div class="en">{esc(cfg['label_noise_note'])}</div></div>
  </div>
  <div>
    <h3>INT8 BERTurk on the test set</h3>
    <table class="cm">
      <tr><th></th><th>predicted clean</th><th>predicted offensive</th></tr>
      <tr><th>gold clean</th><td class="good">{tn:,}</td><td class="bad">{fpc:,}</td></tr>
      <tr><th>gold offensive</th><td class="bad">{fnc:,}</td><td class="good">{tp:,}</td></tr>
    </table>
    <p class="note">Recall on offensive tweets {100 * i8['off_recall']:.0f}%, precision {100 * i8['off_precision']:.0f}%. Missed insults
    are more frequent than false alarms, so some abuse still gets through. Quantization changed only
    {ea['quantization_flips']} of {st['test']['n']:,} decisions compared to FP32.</p>
  </div>
</div>""", cfg))

    # 11 limitations -------------------------------------------------------------------
    S.append(page(11, "Conclusion", "Limitations, and what I would do next", f"""
<div class="g2 lim">
  <div>
    <h3>Limitations</h3>
    <ul>
      <li><b>Domain.</b> The data are tweets, not game chat; game slang and one-word messages are under-represented.</li>
      <li><b>Hard cases.</b> Veiled insults, sarcasm and sub-word look-alikes are still misclassified.</li>
      <li><b>Tail latency.</b> On one thread the INT8 p95 is {lat('bert_int8', s='p95_ms'):.0f} ms, so long messages can exceed
          {budget:.0f} ms. With 4 threads the p95 drops to {lat('bert_int8', '4_thread', 'p95_ms'):.0f} ms.</li>
      <li><b>Labels.</b> Only clean or offensive, one annotation per tweet, with some label noise.</li>
    </ul>
  </div>
  <div>
    <h3>Next steps</h3>
    <ul>
      <li><b>Distillation.</b> Train a smaller student (e.g. DistilBERTurk) from the fine-tuned model, then quantize it,
          to bring the p95 under {budget:.0f} ms.</li>
      <li><b>Cascade.</b> Let tf-idf ({lat('tfidf'):.1f} ms) decide the easy messages and call BERTurk only when it is unsure.</li>
      <li><b>In-domain data.</b> Collect and label real game-chat messages, starting with the ones the model is least sure about.</li>
    </ul>
  </div>
</div>
<div class="verdict">The goal is met: the quantized Transformer is the most accurate model ({i8['macro_f1']:.3f} macro-F1) and
answers in {lat('bert_int8'):.1f} ms on one CPU thread, inside the {budget:.0f} ms budget.
<span>Still open: tail latency and veiled insults (see next steps).</span></div>""", cfg))

    # 12 AI usage ----------------------------------------------------------------------
    S.append(page(12, "Conclusion", "AI usage statement", """
<div class="g2 ai">
  <div>
    <h3>Tool</h3>
    <p><b>Claude Code</b> (Anthropic), an AI coding assistant, used for:</p>
    <ul>
      <li>going through the homework requirements with me and suggesting the plan;</li>
      <li>writing and debugging the Python code: data pipeline, models, training and benchmark scripts, Gradio app;</li>
      <li>running the scripts on my computer;</li>
      <li>the README and first drafts of the slide texts, the architecture diagram and the video script.</li>
    </ul>
    <h3>My part</h3>
    <ul>
      <li>chose the use case and approved the design decisions (latency budget, model ladder, quantization rule);</li>
      <li>all experiments ran on my own computer; every number here comes from these runs (<code>results/*.json</code>),
          not from papers, model cards or AI answers;</li>
      <li>read through the code, tested the app and recorded the demo video.</li>
    </ul>
  </div>
  <div>
    <h3>Sources</h3>
    <ul class="src">
      <li>Ç. Çöltekin. A Corpus of Turkish Offensive Language on Social Media. LREC 2020 (OffensEval-TR 2020, CC-BY 2.0).</li>
      <li>S. Schweter. BERTurk, dbmdz/bert-base-turkish-cased, Hugging Face Hub (MIT licence).</li>
      <li>L. Tunstall, L. von Werra, T. Wolf. Natural Language Processing with Transformers, ch. 8.</li>
      <li>D. Jurafsky, J. H. Martin. Speech and Language Processing, 3rd ed. draft.</li>
      <li>T. Mikolov et al. Distributed Representations of Words and Phrases. NeurIPS 2013 (word2vec).</li>
      <li>Y. Kim. Convolutional Neural Networks for Sentence Classification. EMNLP 2014 (1D CNN).</li>
      <li>PyTorch, Hugging Face Transformers, ONNX Runtime, gensim, scikit-learn, Gradio.</li>
    </ul>
  </div>
</div>""", cfg))

    return TEMPLATE.replace("{{SLIDES}}", "\n".join(S))


TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Real-time moderation of Turkish game chat - SWE015</title>
<style>
@page { size: 1280px 720px; margin: 0; }
:root { --ink:#16181d; --mute:#5b616e; --line:#e3e5e8; --acc:#e8452c; --clean:#2f7d6d; --soft:#f5f5f3; }
* { box-sizing: border-box; }
html, body { margin: 0; background: #d9d9d9; }
body { font-family: "Segoe UI", Arial, sans-serif; color: var(--ink); -webkit-print-color-adjust: exact; print-color-adjust: exact; }
.slide { width: 1280px; height: 720px; background: #fff; position: relative; overflow: hidden; padding: 30px 64px 46px;
         margin: 0 auto 16px; break-after: page; page-break-after: always; display: flex; flex-direction: column; }
@media print { .slide { margin: 0; } html, body { background: #fff; } }
.top { display: flex; justify-content: space-between; align-items: flex-start; }
.nav { display: flex; gap: 6px; }
.nav span { font-size: 13px; color: #a3a7af; padding-top: 7px; width: 132px; border-top: 3px solid var(--line); }
.nav span.on { color: var(--ink); border-top-color: var(--acc); font-weight: 600; }
.pno { font-family: Bahnschrift, "Segoe UI", sans-serif; font-size: 15px; color: var(--mute); padding-top: 6px; }
h1 { font-family: Bahnschrift, "Segoe UI", sans-serif; font-weight: 600; font-size: 37px; line-height: 1.12;
     letter-spacing: -0.005em; margin: 20px 0 0; max-width: 1100px; }
.body { flex: 1; margin-top: 24px; min-height: 0; }
h3 { font-family: Bahnschrift, "Segoe UI", sans-serif; font-weight: 600; font-size: 21px; margin: 0 0 10px; }
p, li { font-size: 17.5px; line-height: 1.42; margin: 0 0 10px; }
ul, ol { margin: 0; padding-left: 22px; }
li { margin-bottom: 9px; }
b { font-weight: 600; }
code { font-family: Consolas, monospace; font-size: .86em; color: #3a3f48; }
.note { font-size: 14.5px; color: var(--mute); line-height: 1.45; }
.g2 { display: grid; grid-template-columns: 1fr 1fr; gap: 48px; }
.g3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 32px; }
img.fig { width: 100%; display: block; }
.foot { position: absolute; left: 64px; right: 64px; bottom: 18px; display: flex; justify-content: space-between;
        font-size: 12.5px; color: #9a9ea6; }

/* cover */
.cover { flex-direction: row; padding: 0; }
.cover-l { width: 600px; padding: 64px 20px 50px 72px; display: flex; flex-direction: column; }
.cover-l .kick { font-size: 15px; color: var(--mute); line-height: 1.45; border-left: 4px solid var(--acc); padding-left: 14px; }
.cover-l h1 { font-size: 56px; line-height: 1.04; margin: 30px 0 16px; letter-spacing: -0.015em; }
.cover-l .lead { font-size: 19px; color: #3a3f48; line-height: 1.45; margin-bottom: 24px; }
.who { border-collapse: collapse; font-size: 16px; margin-top: auto; }
.who td { padding: 4px 14px 4px 0; vertical-align: top; }
.who a.long { white-space: normal; overflow-wrap: anywhere; }
.who td:first-child { color: var(--mute); width: 108px; white-space: nowrap; }
.who a { white-space: nowrap; font-size: 15px; color: var(--ink); text-decoration: underline; text-decoration-color: var(--acc); text-underline-offset: 3px; }
.cover-r { flex: 1; background: var(--soft); padding: 52px 56px 40px 48px; display: flex; flex-direction: column; justify-content: center; }
.cover-r img { width: 100%; border-radius: 6px; box-shadow: 0 18px 40px rgba(22,24,29,.16); border: 1px solid #e0e0e0; }
.kpis { display: grid; grid-template-columns: repeat(3, 1fr); gap: 18px; margin-top: 30px; }
.kpis div { border-top: 3px solid var(--ink); padding-top: 10px; }
.kpis div:first-child { border-top-color: var(--acc); }
.kpis b { display: block; font-family: Bahnschrift, sans-serif; font-size: 40px; font-weight: 600; line-height: 1; }
.kpis small { font-size: 20px; font-weight: 400; }
.kpis span { display: block; font-size: 13.5px; color: var(--mute); margin-top: 6px; line-height: 1.35; }

/* intro */
.intro .big-p { font-size: 21px; line-height: 1.45; }
.sub { font-size: 13px; font-weight: 600; letter-spacing: .08em; text-transform: uppercase; color: var(--mute); margin-top: 22px; }
.targets .sub { margin: 0 0 -6px; }
.users { display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; margin-top: 8px; }
.users div { background: var(--soft); padding: 12px 14px; }
.users b { display: block; font-size: 16px; } .users span { font-size: 14px; color: var(--mute); }
.targets { display: flex; flex-direction: column; gap: 18px; }
.tg { display: flex; gap: 18px; border-top: 1px solid var(--line); padding-top: 16px; }
.tg .no { font-family: Bahnschrift, sans-serif; font-size: 18px; color: var(--acc); font-weight: 600; padding-top: 6px; width: 28px; flex: none; }
.tg b { display: block; font-family: Bahnschrift, sans-serif; font-size: 34px; font-weight: 600; line-height: 1.1; margin-bottom: 4px; }
.tg span { font-size: 16px; color: var(--mute); line-height: 1.4; }

/* data */
.ba { border-collapse: collapse; width: 100%; margin-bottom: 12px; }
.ba td { padding: 8px 8px; border-bottom: 1px solid var(--line); font-size: 15.5px; }
.ba td:nth-child(1) { font-family: Consolas, monospace; font-size: 15px; color: var(--acc); }
.ba td:nth-child(2) { font-family: Consolas, monospace; font-size: 15px; }
.ba td:nth-child(2)::before { content: "\\2192\\00a0\\00a0"; color: #a3a7af; }
.ba td:nth-child(3) { color: var(--mute); font-size: 14px; }

/* architecture */
.arch { display: flex; justify-content: center; }
.arch svg { height: 532px; width: auto; max-width: 100%; }

/* technique map */
.tmap { display: grid; grid-template-columns: repeat(4, 1fr); gap: 18px; }
.tmap .col { border-top: 3px solid var(--ink); padding-top: 10px; }
.tmap .col.hot { border-top-color: var(--acc); }
.tmap .L { font-family: Bahnschrift, sans-serif; font-size: 44px; font-weight: 600; line-height: 1; }
.tmap .hot .L { color: var(--acc); }
.tmap .Ln { font-size: 15px; color: var(--mute); margin: 2px 0 12px; }
.ti { background: var(--soft); padding: 9px 12px; margin-bottom: 9px; }
.ti b { display: block; font-size: 15px; line-height: 1.3; }
.ti span { display: block; font-size: 14px; color: #3a3f48; margin: 3px 0 4px; line-height: 1.35; }
.ti code { font-size: 12.5px; color: var(--mute); }

/* ladder */
.ladder { position: relative; height: 100%; }
.step { position: absolute; width: 540px; display: flex; gap: 16px; background: #fff; border: 1px solid var(--line); padding: 14px 18px;
        box-shadow: 0 6px 16px rgba(22,24,29,.06); }
.step.s0 { left: 0; top: 330px; } .step.s1 { left: 196px; top: 220px; } .step.s2 { left: 392px; top: 110px; } .step.s3 { left: 588px; top: 0; }
.step.hot { border: 2px solid var(--acc); }
.sn { font-family: Bahnschrift, sans-serif; font-size: 46px; font-weight: 600; line-height: .95; width: 34px; color: #c3c6cc; flex: none; }
.step.hot .sn { color: var(--acc); }
.sb { flex: 1; }
.sb b { display: block; font-size: 18px; } .sb span { display: block; font-size: 13.5px; color: var(--mute); line-height: 1.35; margin: 2px 0 7px; }
.bar { position: relative; height: 20px; background: var(--soft); width: calc(100% - 104px); }
.bar i { position: absolute; left: 0; top: 0; bottom: 0; background: #3a3f48; }
.step.hot .bar i { background: var(--acc); }
.bar em { position: absolute; left: calc(100% + 10px); white-space: nowrap; top: 0; font-style: normal; font-size: 13px; line-height: 20px; color: var(--ink); font-weight: 600; }
.extras { position: absolute; right: 0; top: 330px; width: 400px; font-size: 14.5px; color: var(--mute); line-height: 1.45;
          border-left: 3px solid var(--line); padding-left: 14px; }

/* quantization */
.steps li { margin-bottom: 14px; }
.formula { font-family: Cambria, Georgia, serif; font-style: italic; font-size: 26px; margin: 10px 0 2px; padding: 8px 16px;
           background: var(--soft); display: block; width: fit-content; }

/* results */
.res { width: 100%; border-collapse: collapse; margin-bottom: 14px; }
.res th { font-size: 14px; color: var(--mute); font-weight: 600; text-align: left; padding: 6px 10px; border-bottom: 2px solid var(--ink); vertical-align: bottom; }
.res td { font-size: 17px; padding: 10px 10px; border-bottom: 1px solid var(--line); }
.res .n { text-align: right; font-variant-numeric: tabular-nums; }
.res td.n { font-family: Bahnschrift, "Segoe UI", sans-serif; font-size: 18px; }
.res tr.hot td { background: #fdeeeb; font-weight: 600; }
.res tr.hot td:first-child { box-shadow: inset 4px 0 0 var(--acc); }
.res tr.ext td { color: #8a8f98; }

/* comparison */
.charts { gap: 24px; }
.finds { margin-top: 8px; }
.finds div { border-top: 3px solid var(--ink); padding-top: 8px; }
.finds div:nth-child(2) { border-top-color: var(--acc); }
.finds b { display: block; font-family: Bahnschrift, sans-serif; font-size: 22px; font-weight: 600; margin-bottom: 3px; }
.finds span { font-size: 14.5px; color: var(--mute); line-height: 1.4; }

.wl { width: 100%; border-collapse: collapse; margin-top: 6px; }
.wl th { font-size: 13.5px; color: var(--mute); font-weight: 600; text-align: left; padding: 4px 10px; border-bottom: 2px solid var(--ink); }
.wl td { font-size: 14.5px; padding: 6px 10px; border-bottom: 1px solid var(--line); line-height: 1.3; }
.wl td:first-child { font-weight: 600; white-space: nowrap; }
.wl tr.hot td { background: #fdeeeb; }
.wl tr.hot td:first-child { box-shadow: inset 4px 0 0 var(--acc); }
/* errors */
.ex { border-left: 3px solid var(--ink); padding: 4px 0 4px 14px; margin-bottom: 16px; }
.eh { font-size: 13.5px; color: var(--mute); margin-bottom: 3px; }
.tag { font-size: 12px; font-weight: 600; padding: 2px 8px; margin-right: 6px; color: #fff; }
.tag.off { background: var(--acc); } .tag.fa { background: #3a3f48; } .tag.nz { background: #a3a7af; }
.ex q { display: block; font-size: 17px; quotes: "\\201C" "\\201D"; }
.en { font-size: 13.5px; color: var(--mute); margin-top: 4px; line-height: 1.4; }
.cm { border-collapse: collapse; margin: 6px 0 18px; }
.cm th { font-size: 14px; color: var(--mute); font-weight: 600; padding: 8px 14px; text-align: left; }
.cm td { font-family: Bahnschrift, sans-serif; font-size: 34px; font-weight: 600; width: 170px; height: 92px; text-align: center; }
.cm td.good { background: #e9f3f0; color: var(--clean); } .cm td.bad { background: #fdeeeb; color: var(--acc); }

/* limitations / ai */
.lim li { font-size: 18px; } .ai li { font-size: 17px; }
.verdict { margin-top: 26px; border-left: 5px solid var(--acc); background: var(--soft); padding: 18px 24px; font-family: Bahnschrift, sans-serif; font-size: 25px; line-height: 1.3; }
.verdict span { display: block; font-family: "Segoe UI", sans-serif; font-size: 15px; color: var(--mute); margin-top: 6px; }
.ai li { margin-bottom: 7px; }
.src li { font-size: 14.5px; color: #3a3f48; }
</style></head><body>
{{SLIDES}}
</body></html>"""


def find_browser():
    for c in [r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
              r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
              r"C:\Program Files\Google\Chrome\Application\chrome.exe"]:
        if Path(c).exists():
            return c
    return shutil.which("msedge") or shutil.which("google-chrome") or shutil.which("chromium")


def main() -> None:
    out_html = DOCS / "slides.html"
    out_html.write_text(build(), encoding="utf-8")
    print(f"wrote {out_html}")
    browser = find_browser()
    if not browser:
        print("No Edge/Chrome found - open slides.html in a browser and print it to PDF.")
        return
    out_pdf = DOCS / "slides.pdf"
    subprocess.run([browser, "--headless", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={out_pdf}", out_html.as_uri()], check=True, timeout=120)
    print(f"wrote {out_pdf}")


if __name__ == "__main__":
    main()
