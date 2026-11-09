import re
from collections import Counter


def top_words(text, k):
    if not isinstance(text, str):
        raise TypeError("text must be a str")
    if isinstance(k, bool) or not isinstance(k, int):
        raise TypeError("k must be an int")
    if k < 0:
        raise ValueError("k must not be negative")
    counts = Counter(re.findall(r"\w+", text.lower()))
    return counts.most_common(k)
