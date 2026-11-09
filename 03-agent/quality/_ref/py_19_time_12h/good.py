import re

_PATTERN = re.compile(r"([0-9]{1,2}):([0-9]{2}) ?([AaPp][Mm])")


def to_24h(text):
    if not isinstance(text, str):
        raise TypeError("text must be a str")
    m = _PATTERN.fullmatch(text.strip())
    if m is None:
        raise ValueError(f"invalid time: {text!r}")
    hour, minute, suffix = int(m.group(1)), int(m.group(2)), m.group(3).lower()
    if not 1 <= hour <= 12 or minute > 59:
        raise ValueError(f"invalid time: {text!r}")
    hour %= 12
    if suffix == "pm":
        hour += 12
    return f"{hour:02d}:{minute:02d}"
