"""Fetch trained weights from the Hugging Face Hub (they are not stored in git).

By default only the artefacts the application needs are downloaded (~135 MB):
the quantized BERTurk (INT8 ONNX), the tokenizer, and the three small models.
``--all`` also downloads the two FP32 BERTurk variants (~850 MB) used by the
benchmark and the side-by-side comparison.
"""
from __future__ import annotations

from .config import HF_REPO_ID, MODELS_DIR

APP_PATTERNS = [
    "lexicon/*", "tfidf/*", "w2v_ffn/*", "w2v_cnn/*",
    "berturk_onnx/model_int8.onnx", "berturk_onnx/tokenizer*", "berturk_onnx/meta.json",
]
ALL_PATTERNS = APP_PATTERNS + ["berturk_onnx/model_fp32.onnx", "berturk_fp32/*"]


def download_models(all_models: bool = False, repo_id: str | None = None) -> None:
    from huggingface_hub import snapshot_download

    repo_id = repo_id or HF_REPO_ID
    if not repo_id:
        raise RuntimeError("HF_REPO_ID is not set in moderation/config.py")
    snapshot_download(repo_id=repo_id, local_dir=MODELS_DIR,
                      allow_patterns=ALL_PATTERNS if all_models else APP_PATTERNS)
