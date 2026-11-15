import argparse
import json
import time

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          get_linear_schedule_with_warmup)

from .config import BERT_DIR, BERT_NAME, LABELS, MAX_LEN, SEED
from .data import load_split
from .normalize import normalize


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def make_loader(texts, labels, tokenizer, max_len, batch_size, shuffle):
    data = list(zip([normalize(t) for t in texts], labels))

    def collate(batch):
        enc = tokenizer([b[0] for b in batch], truncation=True, max_length=max_len,
                        padding=True, return_tensors="pt")
        enc["labels"] = torch.tensor([b[1] for b in batch])
        return enc

    return DataLoader(data, batch_size=batch_size, shuffle=shuffle, collate_fn=collate)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    out = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items() if k != "labels"}
        out.append(model(**batch).logits.argmax(-1).cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=BERT_NAME)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-len", type=int, default=MAX_LEN)
    ap.add_argument("--max-train", type=int, default=0, help="use only the first N training messages (0 = all)")
    args = ap.parse_args()
    torch.manual_seed(SEED)
    device = pick_device()
    print("device:", device)

    x_train, y_train = load_split("train")
    x_val, y_val = load_split("val")
    if args.max_train:
        x_train, y_train = x_train[:args.max_train], y_train[:args.max_train]

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=2,
        id2label=dict(enumerate(LABELS)), label2id={l: i for i, l in enumerate(LABELS)},
    ).to(device)

    train_loader = make_loader(x_train, y_train, tokenizer, args.max_len, args.batch_size, True)
    val_loader = make_loader(x_val, y_val, tokenizer, args.max_len, 64, False)

    n_pos = max(sum(y_train), 1)
    weights = torch.tensor([1.0, (len(y_train) - n_pos) / n_pos], device=device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * steps), steps)

    best_f1, log, t0 = -1.0, [], time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for step, batch in enumerate(train_loader, 1):
            labels = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            loss = loss_fn(model(**batch).logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total += loss.item()
            if step % 100 == 0:
                print(f"epoch {epoch} step {step}/{len(train_loader)} loss {total / step:.4f}")
        f1 = f1_score(y_val, predict(model, val_loader, device), average="macro")
        log.append({"epoch": epoch, "train_loss": total / len(train_loader), "val_macro_f1": f1})
        print(f"epoch {epoch} | train loss {total / len(train_loader):.4f} | val macro-F1 {f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            BERT_DIR.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(BERT_DIR)
            tokenizer.save_pretrained(BERT_DIR)

    info = {"base_model": args.model, "device": device, "epochs": args.epochs, "lr": args.lr,
            "batch_size": args.batch_size, "max_len": args.max_len, "train_size": len(x_train),
            "best_val_macro_f1": best_f1, "minutes": (time.time() - t0) / 60, "log": log}
    (BERT_DIR / "train_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"best val macro-F1 {best_f1:.4f} | saved to {BERT_DIR}")


if __name__ == "__main__":
    main()
