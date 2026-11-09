import re
from collections import Counter

_WORD = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)*")


def top_words(text, k):
    if not isinstance(text, str):
        raise TypeError("text must be a str")
    if isinstance(k, bool) or not isinstance(k, int):
        raise TypeError("k must be an int")
    if k < 0:
        raise ValueError("k must not be negative")
    counts = Counter(w.lower() for w in _WORD.findall(text))
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return ordered[:k]
