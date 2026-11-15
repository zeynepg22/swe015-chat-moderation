import argparse

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import MAX_CHARS, N_SAMPLES, PROCESSED_DIR, RAW_CSV, SEED

TOXIC_COLS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(RAW_CSV))
    ap.add_argument("--n", type=int, default=N_SAMPLES, help="total number of messages to keep")
    args = ap.parse_args()

    df = pd.read_csv(args.raw, encoding="utf-8")
    df["label"] = (df[TOXIC_COLS].max(axis=1) > 0).astype(int)
    df["text"] = df["comment_text"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    df = df[["text", "label"]].drop_duplicates("text")
    df = df[df["text"].str.len().between(3, MAX_CHARS)]
    print(f"chat-length comments: {len(df)} (offensive ratio {df.label.mean():.3f})")

    if len(df) > args.n:
        df, _ = train_test_split(df, train_size=args.n, stratify=df["label"], random_state=SEED)

    train, rest = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=SEED)
    val, test = train_test_split(rest, test_size=0.5, stratify=rest["label"], random_state=SEED)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for name, part in [("train", train), ("val", val), ("test", test)]:
        part.to_csv(PROCESSED_DIR / f"{name}.csv", index=False, encoding="utf-8")
        print(f"{name:5} {len(part):6} messages | offensive {part.label.sum():5} ({part.label.mean():.3f})")


if __name__ == "__main__":
    main()
