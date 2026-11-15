import pandas as pd

from .config import PROCESSED_DIR


def load_split(name: str) -> tuple[list[str], list[int]]:
    path = PROCESSED_DIR / f"{name}.csv"
    if not path.exists():
        raise SystemExit(f"{path} not found - run: python -m src.prepare_data")
    df = pd.read_csv(path, encoding="utf-8")
    return df["text"].astype(str).tolist(), df["label"].astype(int).tolist()
