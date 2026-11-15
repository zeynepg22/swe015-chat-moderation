import shutil

import numpy as np
import onnxruntime as ort
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .config import BERT_DIR, ONNX_DIR


class LogitsOnly(torch.nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits


def main():
    if not (BERT_DIR / "config.json").exists():
        raise SystemExit(f"{BERT_DIR} not found - fine-tune first: python -m src.train_bert")
    ONNX_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(BERT_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(BERT_DIR).eval()

    fp32 = ONNX_DIR / "model_fp32.onnx"
    int8 = ONNX_DIR / "model_int8.onnx"
    sample = tokenizer(["you are a great player", "gg"], padding=True, return_tensors="pt")

    torch.onnx.export(
        LogitsOnly(model), (sample["input_ids"], sample["attention_mask"]), str(fp32),
        input_names=["input_ids", "attention_mask"], output_names=["logits"],
        dynamic_axes={"input_ids": {0: "batch", 1: "seq"},
                      "attention_mask": {0: "batch", 1: "seq"},
                      "logits": {0: "batch"}},
        opset_version=17, dynamo=False,
    )

    quantize_dynamic(str(fp32), str(int8), weight_type=QuantType.QInt8)

    tokenizer.save_pretrained(ONNX_DIR)
    shutil.copy(BERT_DIR / "config.json", ONNX_DIR / "config.json")

    feed = {k: v.numpy().astype(np.int64) for k, v in sample.items() if k != "token_type_ids"}
    out = {}
    for path in (fp32, int8):
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        out[path.name] = sess.run(None, feed)[0]
        print(f"{path.name:18} {path.stat().st_size / 1e6:8.1f} MB  logits[0] = {out[path.name][0].round(3)}")
    print("max |FP32 - INT8| logit difference:", float(np.abs(out[fp32.name] - out[int8.name]).max()))


if __name__ == "__main__":
    main()
