"""Step 6 - evaluate every model on the SAME held-out test set (official
OffensEval-TR 2020 test, 3,528 tweets) and benchmark size and CPU latency.

Latency protocol (identical for every model):
  * batch size 1 (one chat message at a time), end-to-end: normalisation +
    tokenisation + model + softmax/threshold;
  * N_RUNS = 300 different test messages (>= 100 required), after N_WARMUP
    warm-up calls on 20 other messages; each latency run uses a freshly loaded
    model (no cache from the accuracy pass); mean, median, p95 in milliseconds;
  * CPU only; measured with 1 thread (main setting = one worker per core) and
    with 4 threads.
Outputs: results/results.json, results/results.md, results/test_predictions.csv
and results/tradeoff.png.
"""
import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from moderation.config import RESULTS_DIR, SEED  # noqa: E402
from moderation.data import load_split  # noqa: E402
from moderation.metrics import classification_metrics  # noqa: E402
from moderation.registry import SPECS, load_model, size_mb  # noqa: E402

N_RUNS, N_WARMUP = 300, 20
THREAD_SETTINGS = [1, 4]
LATENCY_BUDGET_MS = 50.0          # mean per message, 1 CPU thread


def hardware() -> dict:
    try:
        import cpuinfo
        cpu = cpuinfo.get_cpu_info().get("brand_raw", platform.processor())
    except Exception:  # pragma: no cover
        cpu = platform.processor()
    import onnxruntime
    import psutil
    return {"cpu": cpu, "physical_cores": psutil.cpu_count(logical=False),
            "ram_gb": round(psutil.virtual_memory().total / 2**30, 1),
            "os": f"{platform.system()} {platform.release()}", "python": platform.python_version(),
            "torch": torch.__version__, "onnxruntime": onnxruntime.__version__,
            "device": "CPU only (GPU not used for inference)",
            "date": datetime.now().isoformat(timespec="minutes")}


def time_model(model, warmup, messages) -> dict:
    for m in warmup:
        model.predict_proba([m])
    times = []
    for m in messages:
        t0 = time.perf_counter()
        p = model.predict_proba([m])
        _ = p[0] >= model.threshold
        times.append((time.perf_counter() - t0) * 1000)
    t = np.array(times)
    return {"mean_ms": float(t.mean()), "median_ms": float(np.median(t)),
            "p95_ms": float(np.percentile(t, 95)), "runs": len(t)}


def main(only: list[str] | None = None) -> None:
    """Benchmark every model, or with ``only`` re-measure just those models and keep the other rows."""
    test = load_split("test")
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(test), N_WARMUP + N_RUNS, replace=False)
    warmup_msgs = list(test.text.iloc[idx[:N_WARMUP]])
    bench_msgs = list(test.text.iloc[idx[N_WARMUP:]])     # disjoint from the warm-up messages
    hw = hardware()
    print(json.dumps(hw, indent=2))

    # tweet ids only, no text: the tweets stay in data/ (downloaded by 01_prepare_data.py)
    rows, preds = [], {"id": test.id, "label": test.label}
    for key, spec in SPECS.items():
        if only and key not in only:
            continue
        print(f"\n== {spec.display}")
        # 1) accuracy pass over the whole test set (batched; thread count does not change outputs)
        torch.set_num_threads(8)
        model = load_model(key, num_threads=8)
        p = np.asarray(model.predict_proba(list(test.text)))
        metrics = classification_metrics(test.label, p, model.threshold)
        preds[f"p_{key}"] = p
        del model
        # 2) latency pass: a FRESH instance per setting, so no cache from the accuracy pass
        latency = {}
        for th in THREAD_SETTINGS:
            torch.set_num_threads(th)
            m = load_model(key, num_threads=th)
            latency[f"{th}_thread"] = time_model(m, warmup_msgs, bench_msgs)
            del m
            print(f"  {th} thread(s): mean {latency[f'{th}_thread']['mean_ms']:.2f} ms, "
                  f"p95 {latency[f'{th}_thread']['p95_ms']:.2f} ms")
        row = {"key": key, "model": spec.display, "ladder_step": spec.ladder_step,
               "precision": spec.precision, "runtime": spec.runtime,
               "size_mb": size_mb(spec), **metrics, "latency": latency,
               "meets_budget": latency["1_thread"]["mean_ms"] <= LATENCY_BUDGET_MS}
        print(f"  macro-F1 {metrics['macro_f1']:.4f}  size {row['size_mb']:.1f} MB")
        rows.append(row)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if only:                                 # merge the re-measured models into the saved results
        saved = json.loads((RESULTS_DIR / "results.json").read_text())
        new = {r["key"]: r for r in rows}
        saved["models"] = [new.get(r["key"], r) for r in saved["models"]]
        saved.setdefault("reruns", []).append({"models": sorted(new), "date": hw["date"]})
        old = pd.read_csv(RESULTS_DIR / "test_predictions.csv", keep_default_na=False)
        assert (old.id.values == test.id.values).all()
        for k in new:
            old[f"p_{k}"] = preds[f"p_{k}"]
        (RESULTS_DIR / "results.json").write_text(json.dumps(saved, indent=2))
        old.to_csv(RESULTS_DIR / "test_predictions.csv", index=False)
        write_markdown(saved)
        plot(saved["models"])
        print((RESULTS_DIR / "results.md").read_text(encoding="utf-8"))
        return
    out = {"hardware": hw, "test_set": "OffensEval-TR 2020 official test (n=3528)",
           "latency_protocol": {"runs": N_RUNS, "warmup": N_WARMUP, "batch_size": 1,
                                "threads": THREAD_SETTINGS, "budget_ms_mean_1_thread": LATENCY_BUDGET_MS},
           "models": rows}
    (RESULTS_DIR / "results.json").write_text(json.dumps(out, indent=2))
    pd.DataFrame(preds).to_csv(RESULTS_DIR / "test_predictions.csv", index=False)
    write_markdown(out)
    plot(rows)
    print((RESULTS_DIR / "results.md").read_text(encoding="utf-8"))


def fmt_size(mb: float) -> str:
    return f"{mb:.3f}" if mb < 1 else f"{mb:.1f}"


def write_markdown(out: dict) -> None:
    hw = out["hardware"]
    lines = [
        "# Results - OffensEval-TR 2020 official test set (n = 3,528)", "",
        f"Hardware: {hw['cpu']}, {hw['physical_cores']} cores, {hw['ram_gb']} GB RAM, {hw['os']}; "
        f"{hw['device']}. PyTorch {hw['torch']}, ONNX Runtime {hw['onnxruntime']}. Measured {hw['date']}"
        + "".join(f"; {', '.join(r['models'])} re-measured {r['date']}" for r in out.get("reruns", [])) + ".", "",
        f"Latency: batch size 1, end-to-end (normalisation + tokenisation + model), mean of "
        f"{out['latency_protocol']['runs']} runs after {out['latency_protocol']['warmup']} warm-up calls. "
        f"Budget: mean <= {LATENCY_BUDGET_MS:.0f} ms per message on 1 CPU thread.", "",
        "| Ladder step | Model | Precision / runtime | Macro-F1 | Offensive F1 | Size (MB) "
        "| Latency 1 thread mean (ms) | p95 (ms) | Latency 4 threads mean (ms) | Budget |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in out["models"]:
        l1, l4 = r["latency"]["1_thread"], r["latency"]["4_thread"]
        lines.append(
            f"| {r['ladder_step']} | {r['model']} | {SPECS[r['key']].precision} / {r['runtime']} | {r['macro_f1']:.3f} | "
            f"{r['off_f1']:.3f} | {fmt_size(r['size_mb'])} | {l1['mean_ms']:.2f} | {l1['p95_ms']:.2f} | "
            f"{l4['mean_ms']:.2f} | {'yes' if r['meets_budget'] else 'no'} |")
    (RESULTS_DIR / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


LABEL_OFFSETS = {  # (dx, dy) in points, chosen so that labels do not overlap
    "lexicon": (10, -2), "tfidf": (10, -4), "w2v_ffn": (10, -4), "w2v_cnn": (10, -4),
    "bert_fp32": (-12, -38), "bert_onnx_fp32": (-30, 18), "bert_int8": (-150, -6),
}
COLORS = {"lexicon": "#64748b", "tfidf": "#0891b2", "w2v_ffn": "#818cf8", "w2v_cnn": "#4338ca",
          "bert_fp32": "#f59e0b", "bert_onnx_fp32": "#ea580c", "bert_int8": "#dc2626"}


def plot(rows) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 5.4), dpi=170)
    for r in rows:
        lat = r["latency"]["1_thread"]["mean_ms"]
        ax.scatter(lat, r["macro_f1"], s=30 + 1.6 * r["size_mb"], alpha=0.75,
                   color=COLORS.get(r["key"], "grey"), edgecolor="white", linewidth=1, zorder=3)
        dx, dy = LABEL_OFFSETS.get(r["key"], (8, -4))
        ax.annotate(f"{r['model']}\n{fmt_size(r['size_mb'])} MB · {lat:.1f} ms", (lat, r["macro_f1"]),
                    textcoords="offset points", xytext=(dx, dy), fontsize=7.5, zorder=4)
    ax.axvline(LATENCY_BUDGET_MS, ls="--", c="#dc2626", lw=1)
    ax.text(LATENCY_BUDGET_MS * 1.04, 0.675, f"budget\n{LATENCY_BUDGET_MS:.0f} ms", fontsize=8, color="#dc2626")
    ax.set_xscale("log")
    ax.set_xlim(0.1, 250)
    ax.set_ylim(min(r["macro_f1"] for r in rows) - 0.02, max(r["macro_f1"] for r in rows) + 0.025)
    ax.grid(alpha=0.25)
    ax.set_xlabel("Mean latency per message, 1 CPU thread (ms, log scale)")
    ax.set_ylabel("Macro-F1 on the test set")
    ax.set_title("Accuracy vs. latency (bubble area ~ model size)", fontsize=11)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "tradeoff.png")


if __name__ == "__main__":
    if "--only" in sys.argv:                 # e.g. --only lexicon  (re-measure one model after retraining it)
        main(only=sys.argv[sys.argv.index("--only") + 1].split(","))
    elif "--report-only" in sys.argv:          # re-draw table / figure from results.json, no re-measuring
        saved = json.loads((RESULTS_DIR / "results.json").read_text())
        write_markdown(saved)
        plot(saved["models"])
    else:
        main()
