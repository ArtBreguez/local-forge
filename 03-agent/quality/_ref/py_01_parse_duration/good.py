import re

_PATTERN = re.compile(r"(?:([0-9]+)d)?(?:([0-9]+)h)?(?:([0-9]+)m)?(?:([0-9]+)s)?")


def parse_duration(text):
    if not isinstance(text, str):
        raise TypeError("text must be a str")
    match = _PATTERN.fullmatch(text)
    if not text or match is None:
        raise ValueError(f"invalid duration: {text!r}")
    d, h, m, s = (int(g) if g else 0 for g in match.groups())
    return d * 86400 + h * 3600 + m * 60 + s
