"""Step 5 - export the fine-tuned BERTurk to ONNX (FP32) and quantize it to INT8
with ONNX Runtime dynamic quantization (technique area D).

Dynamic quantization: the weights of the chosen operators are stored as INT8
with a scale S and zero point Z (affine map  q = round(x / S) + Z); activations
are quantized on the fly at inference time, so no calibration data is needed.

Several quantization variants are compared on the VALIDATION split. The served
model is the smallest variant whose validation macro-F1 is within
MAX_F1_DROP of the FP32 model. All variants are logged to
results/quantization_check.json.
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from onnxruntime.quantization.shape_inference import quant_pre_process
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from moderation.bert_model import BertOnnxModel, BertTorchModel  # noqa: E402
from moderation.config import (BERT_FP32_DIR, BERT_ONNX_DIR, ONNX_FP32_PATH,  # noqa: E402
                               ONNX_INT8_PATH, RESULTS_DIR)
from moderation.data import load_split  # noqa: E402
from moderation.metrics import classification_metrics  # noqa: E402

MAX_F1_DROP = 0.01
PREPROCESSED = BERT_ONNX_DIR / "model_fp32_preprocessed.onnx"

# name -> (source graph, operators quantized to INT8, per-channel scales)
VARIANTS = {
    "matmul":                ("raw", ["MatMul", "Gemm"], False),
    "matmul_per_channel":    ("raw", ["MatMul", "Gemm"], True),
    "pre+matmul":            ("pre", ["MatMul", "Gemm"], False),
    "pre+matmul+embeddings": ("pre", ["MatMul", "Gemm", "Gather"], False),
}


def export_onnx() -> None:
    model = AutoModelForSequenceClassification.from_pretrained(BERT_FP32_DIR).eval()
    tok = AutoTokenizer.from_pretrained(BERT_FP32_DIR)
    enc = tok(["örnek bir mesaj", "ikinci örnek mesaj biraz daha uzun"], padding=True, return_tensors="pt")
    args = (enc["input_ids"], enc["attention_mask"], enc["token_type_ids"])
    BERT_ONNX_DIR.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model, args, str(ONNX_FP32_PATH),
        input_names=["input_ids", "attention_mask", "token_type_ids"],
        output_names=["logits"],
        dynamic_axes={n: {0: "batch", 1: "sequence"} for n in ["input_ids", "attention_mask", "token_type_ids"]}
        | {"logits": {0: "batch"}},
        opset_version=17, dynamo=False,
    )
    # tokenizer + threshold travel with the ONNX graphs so the app needs only this folder
    for f in BERT_FP32_DIR.iterdir():
        if f.suffix != ".safetensors" and f.name != "config.json":
            shutil.copy(f, BERT_ONNX_DIR / f.name)


def main() -> None:
    export_onnx()
    # graph clean-up / constant folding recommended by ORT before quantization
    quant_pre_process(str(ONNX_FP32_PATH), str(PREPROCESSED), skip_symbolic_shape=True)

    val = load_split("val")
    thr = json.loads((BERT_FP32_DIR / "meta.json").read_text())["threshold"]
    p_torch = BertTorchModel(BERT_FP32_DIR).predict_proba(val.text)
    p_fp32 = BertOnnxModel(ONNX_FP32_PATH, BERT_ONNX_DIR, "onnx fp32").predict_proba(val.text)
    f1_fp32 = classification_metrics(val.label, p_fp32, thr)["macro_f1"]

    variants = {}
    for name, (src, ops, per_channel) in VARIANTS.items():
        out = BERT_ONNX_DIR / f"variant_{name.replace('+', '_')}.onnx"
        quantize_dynamic(str(PREPROCESSED if src == "pre" else ONNX_FP32_PATH), str(out),
                         weight_type=QuantType.QInt8, op_types_to_quantize=ops, per_channel=per_channel)
        p = BertOnnxModel(out, BERT_ONNX_DIR, name).predict_proba(val.text)
        variants[name] = {"path": out, "ops": ops, "per_channel": per_channel,
                          "size_mb": out.stat().st_size / 2**20,
                          "val_macro_f1": classification_metrics(val.label, p, thr)["macro_f1"],
                          "label_agreement_with_fp32": float(((p >= thr) == (p_fp32 >= thr)).mean())}
        print(f"{name:24s} {variants[name]['size_mb']:6.1f} MB  val macro-F1 {variants[name]['val_macro_f1']:.4f}")

    ok = {k: v for k, v in variants.items() if f1_fp32 - v["val_macro_f1"] <= MAX_F1_DROP}
    chosen = min(ok, key=lambda k: ok[k]["size_mb"])
    shutil.copy(variants[chosen]["path"], ONNX_INT8_PATH)
    for v in variants.values():
        v["path"].unlink()
    PREPROCESSED.unlink()

    report = {
        "max_abs_diff_torch_vs_onnx_fp32": float(np.abs(p_torch - p_fp32).max()),
        "val_macro_f1_fp32": f1_fp32,
        "fp32_onnx_size_mb": ONNX_FP32_PATH.stat().st_size / 2**20,
        "selection_rule": f"smallest variant with val macro-F1 drop <= {MAX_F1_DROP}",
        "chosen_variant": chosen,
        "variants": {k: {kk: vv for kk, vv in v.items() if kk != "path"} for k, v in variants.items()},
    }
    (RESULTS_DIR / "quantization_check.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ["max_abs_diff_torch_vs_onnx_fp32", "val_macro_f1_fp32",
                                             "chosen_variant"]}, indent=2))


if __name__ == "__main__":
    main()
