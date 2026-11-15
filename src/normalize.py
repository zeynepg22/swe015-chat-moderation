import re

_URL = re.compile(r"https?://\S+|www\.\S+")
_TURKISH = str.maketrans({"ç": "c", "ğ": "g", "ı": "i", "ö": "o", "ş": "s", "ü": "u", "â": "a", "î": "i", "û": "u",
                          "Ç": "c", "Ğ": "g", "İ": "i", "Ö": "o", "Ş": "s", "Ü": "u"})
_LEET_RUN = re.compile(r"(?<=[a-z])[013457]+(?=[a-z])")
_LEET_MAP = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t"})
_SYMBOL = re.compile(r"(?<=[a-z])[@$]+|[@$]+(?=[a-z])")
_SYMBOL_MAP = str.maketrans({"@": "a", "$": "s"})
_BANG = re.compile(r"(?<=[a-z])!(?=[a-z])")
_SPACED = re.compile(r"\b(?:[a-z][ .\-_]){3,}[a-z]\b")
_REPEAT = re.compile(r"([a-z])\1{2,}")
_JUNK = re.compile(r"[^a-z0-9*'\s<>]")
_SPACE = re.compile(r"\s+")


def normalize(text: str) -> str:
    t = str(text).translate(_TURKISH).lower()
    t = _URL.sub(" <url> ", t)
    t = _LEET_RUN.sub(lambda m: m.group(0).translate(_LEET_MAP), t)
    t = _SYMBOL.sub(lambda m: m.group(0).translate(_SYMBOL_MAP), t)
    t = _BANG.sub("i", t)
    t = _SPACED.sub(lambda m: re.sub(r"[ .\-_]", "", m.group(0)), t)
    t = _REPEAT.sub(r"\1\1", t)
    t = _JUNK.sub(" ", t)
    return _SPACE.sub(" ", t).strip()


def tokenize(text: str) -> list[str]:
    return normalize(text).split()


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    for s in ["You are an ID10T!!!", "shiiiiit this n00b", "f u c k off", "a$$ and sh!t",
              "see https://example.com now", "what the f*ck", "ŞEREFSİZ herif", "çok iyi oynadın"]:
        print(f"{s!r:35} -> {normalize(s)!r}")
