"""Real-time Turkish chat moderation - Gradio application.

Run:  python app.py      (opens http://127.0.0.1:7860)

* Every message typed into the chat is classified live (clean / offensive)
  with a confidence score; offensive messages are masked.
* The served model is the INT8-quantized BERTurk (ONNX Runtime, CPU, 1 thread).
  The dropdown switches to any other model of the ladder.
* The side panel logs the end-to-end response time of every message and
  compares all models on the latest message.
"""
from __future__ import annotations

import os
import time
from datetime import datetime

import gradio as gr
import torch

from moderation.config import HF_REPO_ID
from moderation.normalize import normalize
from moderation.registry import APP_DEFAULT, SPECS, available, load_model, predict_one

APP_THREADS = 1                  # same setting as the latency budget in the benchmark
LATENCY_BUDGET_MS = 50.0
torch.set_num_threads(APP_THREADS)

# --- make sure the weights are present -------------------------------------
WANT_ALL = os.environ.get("ALL_MODELS") == "1"   # also fetch the two FP32 BERTurk variants
if not available(APP_DEFAULT) or (WANT_ALL and not all(available(k) for k in SPECS)):
    if HF_REPO_ID is None:
        raise SystemExit("Model files missing. Run the training scripts (see README) "
                         "or set HF_REPO_ID in moderation/config.py.")
    from moderation.hub import download_models
    print(f"Downloading model weights from the Hugging Face Hub ({HF_REPO_ID}) ...")
    download_models(all_models=WANT_ALL)

MODEL_KEYS = [k for k in SPECS if available(k)]
CHOICES = [(f"{SPECS[k].short}   [{SPECS[k].precision}, {SPECS[k].runtime}]", k)
           for k in MODEL_KEYS]
_cache: dict = {}


def get_model(key: str):
    if key not in _cache:
        _cache[key] = load_model(key, num_threads=APP_THREADS)
        _cache[key].predict_proba(["ısınma mesajı"])          # warm-up
    return _cache[key]


for _k in MODEL_KEYS:            # load + warm up every model at start-up (no delay on the first message)
    get_model(_k)
print(f"Loaded {len(MODEL_KEYS)} models; served model: {SPECS[APP_DEFAULT].display}", flush=True)


def classify(text: str, key: str) -> dict:
    model = get_model(key)
    t0 = time.perf_counter()
    label, p_off = predict_one(model, text)
    ms = (time.perf_counter() - t0) * 1000
    return {"label": label, "p_off": p_off, "conf": p_off if label else 1 - p_off, "ms": ms,
            "model": SPECS[key].display, "short": SPECS[key].short, "precision": SPECS[key].precision}


def render_chat(messages: list[dict], moderator_view: bool) -> list[dict]:
    chat = []
    for m in messages:
        r = m["result"]
        badge = (f"`{'OFFENSIVE' if r['label'] else 'clean'} | confidence {r['conf']:.2f} | "
                 f"{r['ms']:.1f} ms | {r['short']}`")
        if r["label"] and not moderator_view:
            body = "*[message hidden by moderation]*"
        elif r["label"]:
            body = f"~~{m['text']}~~"
        else:
            body = m["text"]
        chat.append({"role": "user", "content": f"{body}\n\n{badge}"})
    return chat


def log_rows(messages: list[dict]) -> list[list]:
    rows = []
    for i, m in enumerate(reversed(messages)):
        r = m["result"]
        rows.append([len(messages) - i, m["time"], r["short"],
                     "offensive" if r["label"] else "clean", round(r["conf"], 3), round(r["ms"], 2)])
    return rows


def stats_md(messages: list[dict]) -> str:
    if not messages:
        return f"No messages yet. Latency budget: **{LATENCY_BUDGET_MS:.0f} ms** per message (1 CPU thread)."
    lat = [m["result"]["ms"] for m in messages]
    flagged = sum(m["result"]["label"] for m in messages)
    mean = sum(lat) / len(lat)
    ok = "within" if mean <= LATENCY_BUDGET_MS else "over"
    return (f"{len(messages)} messages, {flagged} flagged. Mean latency **{mean:.1f} ms** "
            f"({ok} the {LATENCY_BUDGET_MS:.0f} ms budget), last message {lat[-1]:.1f} ms.")


def compare_all(text: str) -> list[list]:
    rows = []
    for key in MODEL_KEYS:
        r = classify(text, key)
        rows.append([SPECS[key].short, SPECS[key].precision,
                     "offensive" if r["label"] else "clean", round(r["p_off"], 3), round(r["ms"], 2)])
    return rows


def on_send(text, key, moderator_view, do_compare, messages):
    text = (text or "").strip()
    if not text:
        return gr.skip(), "", gr.skip(), gr.skip(), gr.skip(), gr.skip(), messages
    result = classify(text, key)
    messages = messages + [{"text": text, "result": result, "time": datetime.now().strftime("%H:%M:%S")}]
    comparison = compare_all(text) if do_compare else []
    norm = f"`full:` {normalize(text, 'full')}  \n`light (BERTurk):` {normalize(text, 'light')}"
    return (render_chat(messages, moderator_view), "", log_rows(messages), stats_md(messages),
            comparison, norm, messages)


def on_clear():
    return [], [], stats_md([]), [], "", []


CSS = """
#title {text-align:center; margin-bottom:0}
#subtitle {text-align:center; color:#666; margin-top:4px}
"""

with gr.Blocks(title="Turkish Chat Moderation") as demo:
    messages_state = gr.State([])
    gr.Markdown("# Turkish Chat Moderation", elem_id="title")
    gr.Markdown("SWE015 homework, use case 4. Served model: BERTurk fine-tuned on OffensEval-TR, "
                "INT8 quantized, ONNX Runtime on CPU (1 thread).", elem_id="subtitle")
    with gr.Row():
        with gr.Column(scale=1):
            chat = gr.Chatbot(label="Game chat", height=480, feedback_options=None, group_consecutive_messages=False,
                              placeholder="Type a Turkish chat message below.")
            with gr.Row():
                box = gr.Textbox(placeholder="Mesajını yaz ve Enter'a bas", show_label=False, scale=5,
                                 autofocus=True)
                send = gr.Button("Send", variant="primary", scale=1)
            gr.Examples(
                examples=["Maç çok güzeldi, tebrikler herkese",
                          "s.a.l.a.k mısın sen, oyna düzgün",
                          "L Ü T F E N  susar mısın artık salaaaak",
                          "Bu oyunu kim tasarladıysa beyni yok herhalde"],
                inputs=box, label="Examples")
            norm_md = gr.Markdown(label="Normalised input")
        with gr.Column(scale=1):
            model_dd = gr.Dropdown(choices=CHOICES, value=APP_DEFAULT, label="Moderation model")
            with gr.Row():
                moderator = gr.Checkbox(label="Moderator view (show masked text)", value=False)
                compare = gr.Checkbox(label="Compare all models", value=True)
            stats = gr.Markdown(stats_md([]))
            log = gr.Dataframe(headers=["#", "time", "model", "label", "conf.", "ms"],
                               column_widths=["6%", "14%", "30%", "22%", "13%", "15%"],
                               label="Response-time log (newest first)", interactive=False, wrap=True,
                               max_height=260)
            comp = gr.Dataframe(headers=["model", "precision", "label", "P(off)", "ms"],
                                column_widths=["34%", "18%", "20%", "14%", "14%"],
                                label="All models on the last message", interactive=False, wrap=True)
            clear = gr.Button("Clear chat")

    inputs = [box, model_dd, moderator, compare, messages_state]
    outputs = [chat, box, log, stats, comp, norm_md, messages_state]
    box.submit(on_send, inputs, outputs)
    send.click(on_send, inputs, outputs)
    moderator.change(render_chat, [messages_state, moderator], chat)
    clear.click(on_clear, None, [chat, log, stats, comp, norm_md, messages_state])

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Default(font=["Segoe UI", "Arial", "sans-serif"],
                                        font_mono=["Consolas", "monospace"]), css=CSS)
