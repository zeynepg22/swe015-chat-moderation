import json

import numpy as np

from .config import (LABELS, MAX_LEN, ONNX_DIR, RULE_DIR, TFIDF_DIR, THRESHOLD, W2V_DIR)
from .lexicon import LexiconMatcher
from .normalize import normalize


def dir_size_mb(*paths) -> float:
    total = 0
    for p in paths:
        total += sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) if p.is_dir() else p.stat().st_size
    return total / 1e6


class BaseModel:
    name = "base"
    short = "base"
    precision = "-"

    def proba(self, text: str) -> float:
        raise NotImplementedError

    def predict(self, text: str) -> tuple[str, float]:
        p = self.proba(text)
        return (LABELS[1], p) if p >= THRESHOLD else (LABELS[0], 1.0 - p)


class RuleModel(BaseModel):
    name, short, precision = "Rule: edit-distance lexicon", "Rule lexicon", "rules"

    def __init__(self):
        self.matcher = LexiconMatcher()
        from .config import LEXICON_TXT
        self.size_mb = dir_size_mb(LEXICON_TXT, RULE_DIR)

    def proba(self, text):
        return 1.0 if self.matcher.find(text) else 0.0


class TfidfModel(BaseModel):
    name, short, precision = "Classical: tf-idf + logistic regression", "tf-idf + LR", "FP64"

    def __init__(self):
        import joblib
        self.pipe = joblib.load(TFIDF_DIR / "tfidf_lr.joblib")
        self.size_mb = dir_size_mb(TFIDF_DIR)

    def proba(self, text):
        return float(self.pipe.predict_proba([normalize(text)])[0, 1])


class W2VModel(BaseModel):
    name, short, precision = "Neural: word2vec + feedforward network", "word2vec + FFN", "FP32"

    def __init__(self):
        import torch
        from .train_w2v_nn import FeedForward, average_vector
        self.torch, self.average_vector = torch, average_vector
        self.vocab = json.loads((W2V_DIR / "vocab.json").read_text(encoding="utf-8"))
        self.vectors = np.load(W2V_DIR / "vectors.npy")
        self.net = FeedForward(self.vectors.shape[1])
        self.net.load_state_dict(torch.load(W2V_DIR / "ffn.pt", map_location="cpu"))
        self.net.eval()
        self.size_mb = dir_size_mb(W2V_DIR)

    def proba(self, text):
        x = self.average_vector(normalize(text).split(), self.vocab, self.vectors)
        with self.torch.no_grad():
            logits = self.net(self.torch.from_numpy(x).unsqueeze(0))
        return float(self.torch.softmax(logits, -1)[0, 1])


class OnnxBertModel(BaseModel):

    def __init__(self, filename: str, name: str, precision: str):
        import onnxruntime as ort
        from transformers import AutoTokenizer
        self.name, self.precision = name, precision
        self.short = f"BERT {precision}"
        path = ONNX_DIR / filename
        self.session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        self.tokenizer = AutoTokenizer.from_pretrained(ONNX_DIR)
        self.size_mb = dir_size_mb(path)

    def proba(self, text):
        enc = self.tokenizer(normalize(text), truncation=True, max_length=MAX_LEN, return_tensors="np")
        feed = {"input_ids": enc["input_ids"].astype(np.int64),
                "attention_mask": enc["attention_mask"].astype(np.int64)}
        logits = self.session.run(None, feed)[0][0]
        e = np.exp(logits - logits.max())
        return float(e[1] / e.sum())


REGISTRY = {
    "rule": RuleModel,
    "tfidf": TfidfModel,
    "w2v": W2VModel,
    "bert_fp32": lambda: OnnxBertModel("model_fp32.onnx", "Transformer: BERT (FP32, ONNX Runtime)", "FP32"),
    "bert_int8": lambda: OnnxBertModel("model_int8.onnx", "Quantized Transformer: BERT (INT8, ONNX Runtime)", "INT8"),
}


def load_available() -> dict:
    models = {}
    for key, ctor in REGISTRY.items():
        try:
            models[key] = ctor()
        except Exception as e:
            print(f"[skip] {key}: {type(e).__name__}: {e}")
    return models
