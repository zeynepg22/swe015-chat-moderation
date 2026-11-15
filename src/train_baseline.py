from collections import Counter

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline

from .config import RULE_DIR, SEED, TFIDF_DIR
from .data import load_split
from .lexicon import LexiconMatcher
from .normalize import normalize


def main():
    x_train, y_train = load_split("train")
    x_val, y_val = load_split("val")
    norm_train = [normalize(t) for t in x_train]
    norm_val = [normalize(t) for t in x_val]

    counts = Counter(tok for t, y in zip(norm_train, y_train) if y == 0 for tok in t.split())
    safe = sorted(w for w, c in counts.items() if c >= 3)
    RULE_DIR.mkdir(parents=True, exist_ok=True)
    (RULE_DIR / "safe_words.txt").write_text("\n".join(safe), encoding="utf-8")
    matcher = LexiconMatcher(safe_words=safe)
    rule_pred = [1 if matcher.find(t) else 0 for t in x_val]
    print(f"rule baseline   | safe words {len(safe):6} | val macro-F1 {f1_score(y_val, rule_pred, average='macro'):.4f}")

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=False, token_pattern=r"\S+", ngram_range=(1, 2),
                                  min_df=2, sublinear_tf=True)),
        ("lr", LogisticRegression(C=4.0, max_iter=1000, class_weight="balanced", random_state=SEED)),
    ])
    pipe.fit(norm_train, y_train)
    val_pred = pipe.predict(norm_val)
    print(f"tf-idf + LR     | features {len(pipe['tfidf'].vocabulary_):8} | val macro-F1 {f1_score(y_val, val_pred, average='macro'):.4f}")

    TFIDF_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, TFIDF_DIR / "tfidf_lr.joblib")
    print("saved", TFIDF_DIR / "tfidf_lr.joblib")


if __name__ == "__main__":
    main()
