import argparse
import json

import numpy as np
import torch
import torch.nn as nn
from gensim.models import Word2Vec
from sklearn.metrics import f1_score

from .config import SEED, W2V_DIR
from .data import load_split
from .normalize import tokenize

DIM = 100


class FeedForward(nn.Module):

    def __init__(self, dim: int = DIM, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        return self.net(x)


def average_vector(tokens, vocab, vectors):
    idx = [vocab[t] for t in tokens if t in vocab]
    if not idx:
        return np.zeros(vectors.shape[1], dtype=np.float32)
    return vectors[idx].mean(axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=0.05)
    args = ap.parse_args()
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    x_train, y_train = load_split("train")
    x_val, y_val = load_split("val")
    tok_train = [tokenize(t) for t in x_train]
    tok_val = [tokenize(t) for t in x_val]

    w2v = Word2Vec(sentences=tok_train, vector_size=DIM, window=5, min_count=2,
                   sg=1, negative=5, epochs=10, seed=SEED, workers=1)
    vocab = {w: i for i, w in enumerate(w2v.wv.index_to_key)}
    vectors = w2v.wv.vectors.astype(np.float32)
    print(f"word2vec vocabulary: {len(vocab)} words, dim {DIM}")

    xtr = torch.tensor(np.stack([average_vector(t, vocab, vectors) for t in tok_train]))
    xva = torch.tensor(np.stack([average_vector(t, vocab, vectors) for t in tok_val]))
    ytr = torch.tensor(y_train)

    model = FeedForward()
    n_pos = max(int(ytr.sum()), 1)
    weights = torch.tensor([1.0, (len(ytr) - n_pos) / n_pos])
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    best_f1, best_state = -1.0, None
    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(len(xtr))
        total = 0.0
        for i in range(0, len(xtr), 64):
            b = perm[i:i + 64]
            optimizer.zero_grad()
            loss = loss_fn(model(xtr[b]), ytr[b])
            loss.backward()
            optimizer.step()
            total += loss.item() * len(b)
        model.eval()
        with torch.no_grad():
            pred = model(xva).argmax(1).numpy()
        f1 = f1_score(y_val, pred, average="macro")
        print(f"epoch {epoch:2d} | train loss {total / len(xtr):.4f} | val macro-F1 {f1:.4f}")
        if f1 > best_f1:
            best_f1, best_state = f1, {k: v.clone() for k, v in model.state_dict().items()}

    W2V_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, W2V_DIR / "ffn.pt")
    np.save(W2V_DIR / "vectors.npy", vectors)
    (W2V_DIR / "vocab.json").write_text(json.dumps(vocab), encoding="utf-8")
    print(f"best val macro-F1 {best_f1:.4f} | saved to {W2V_DIR}")


if __name__ == "__main__":
    main()
