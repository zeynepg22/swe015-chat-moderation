import re

from .config import LEXICON_TXT, RULE_DIR
from .normalize import normalize

_DOUBLE = re.compile(r"([a-z])\1+")


def edit_distance(a: str, b: str, sub_cost: int = 1, wildcard: str = "*") -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            same = ca == cb or ca == wildcard or cb == wildcard
            cur.append(min(prev[j] + 1,
                           cur[j - 1] + 1,
                           prev[j - 1] + (0 if same else sub_cost)))
        prev = cur
    return prev[-1]


def load_words(path) -> set[str]:
    with open(path, encoding="utf-8") as f:
        return {w.strip().lower() for w in f if w.strip() and not w.startswith("#")}


class LexiconMatcher:
    def __init__(self, words=None, safe_words=None):
        self.words = set(words) if words is not None else load_words(LEXICON_TXT)
        if safe_words is None:
            safe_path = RULE_DIR / "safe_words.txt"
            safe_words = load_words(safe_path) if safe_path.exists() else set()
        self.safe = set(safe_words) - self.words
        self.ordered = sorted(self.words)

    def match(self, token: str):
        tok = token.strip("'")
        if not any(c.isalpha() for c in tok):
            return None
        candidates = {tok, _DOUBLE.sub(r"\1", tok)}
        for c in candidates:
            if c in self.words:
                return c
        if tok in self.safe:
            return None
        masked = "*" in tok
        for c in candidates:
            for w in self.ordered:
                if abs(len(c) - len(w)) > 1:
                    continue
                if masked:
                    if len(c) == len(w) and edit_distance(c, w) == 0:
                        return w
                elif len(w) >= 4 and len(c) >= 3 and edit_distance(c, w) <= 1:
                    return w
        return None

    def find(self, text: str) -> list[str]:
        return [m for m in (self.match(t) for t in normalize(text).split()) if m]

    def mask(self, text: str) -> tuple[str, int]:
        out, hits = [], 0
        for raw in str(text).split():
            if any(self.match(t) for t in normalize(raw).split()):
                out.append("*" * len(raw))
                hits += 1
            else:
                out.append(raw)
        return " ".join(out), hits
