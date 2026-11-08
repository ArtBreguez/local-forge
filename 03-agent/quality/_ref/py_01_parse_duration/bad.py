import re


def parse_duration(text):
    if not isinstance(text, str):
        raise TypeError("text must be a str")
    parts = re.findall(r"([0-9]+)([dhms])", text)
    if not parts:
        raise ValueError(text)
    mult = {"d": 86400, "h": 3600, "m": 60, "s": 1}
    return sum(int(n) * mult[u] for n, u in parts)
