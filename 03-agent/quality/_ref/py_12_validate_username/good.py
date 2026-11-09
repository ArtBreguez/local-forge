import re

_PATTERN = re.compile(r"[a-z](?:[a-z0-9]|[_.](?![_.]))*[a-z0-9]")


def validate_username(name):
    if not isinstance(name, str):
        raise TypeError("name must be a str")
    s = name.strip().lower()
    if not 3 <= len(s) <= 20:
        raise ValueError("length must be 3..20")
    if _PATTERN.fullmatch(s) is None:
        raise ValueError(f"invalid username: {s!r}")
    return s
