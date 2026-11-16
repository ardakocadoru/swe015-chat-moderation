"""Central paths and constants shared by the training scripts and the app."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

# Model artefact locations (created by the scripts in scripts/).
LEXICON_PATH = MODELS_DIR / "lexicon" / "lexicon.json"
TFIDF_PATH = MODELS_DIR / "tfidf" / "tfidf_logreg.joblib"
W2V_DIR = MODELS_DIR / "w2v_ffn"
W2V_CNN_DIR = MODELS_DIR / "w2v_cnn"
BERT_FP32_DIR = MODELS_DIR / "berturk_fp32"
BERT_ONNX_DIR = MODELS_DIR / "berturk_onnx"
ONNX_FP32_PATH = BERT_ONNX_DIR / "model_fp32.onnx"
ONNX_INT8_PATH = BERT_ONNX_DIR / "model_int8.onnx"

# Dataset: OffensEval-TR 2020 (Coltekin, 2020), CC-BY 2.0.
OFFENSEVAL_URL = "https://coltekin.github.io/offensive-turkish/offenseval2020-turkish.zip"
LABELS = ["NOT", "OFF"]          # 0 = clean, 1 = offensive
LABEL_NAMES_TR = {0: "clean", 1: "offensive"}

BASE_BERT = "dbmdz/bert-base-turkish-cased"   # BERTurk
MAX_LEN = 96                                   # covers 99.8 % of training tweets (BERTurk tokens)
SEED = 42
VAL_FRACTION = 0.10

# Hugging Face Hub repo that hosts the trained weights: https://huggingface.co/ardakocadoru/swe015-chat-moderation
HF_REPO_ID = "ardakocadoru/swe015-chat-moderation"
