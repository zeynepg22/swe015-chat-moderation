from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

RAW_CSV = ROOT / "data" / "raw" / "train.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
LEXICON_TXT = ROOT / "data" / "lexicon.txt"

MODELS_DIR = ROOT / "models"
RULE_DIR = MODELS_DIR / "rule"
TFIDF_DIR = MODELS_DIR / "tfidf"
W2V_DIR = MODELS_DIR / "w2v_ffn"
BERT_DIR = MODELS_DIR / "bert"
ONNX_DIR = MODELS_DIR / "bert_onnx"

RESULTS_DIR = ROOT / "results"

SEED = 42
MAX_CHARS = 300
N_SAMPLES = 40_000
BERT_NAME = "bert-base-uncased"
MAX_LEN = 64
THRESHOLD = 0.5
LATENCY_BUDGET_MS = 50.0
LABELS = ["clean", "offensive"]

HF_REPO = "zeynep22/swe015-chat-moderation"
