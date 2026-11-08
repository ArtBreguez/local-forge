import re


def rle_encode(s):
    if not isinstance(s, str):
        raise TypeError("s must be a str")
    if any(ch.isdigit() for ch in s):
        raise ValueError("input must not contain digits")
    out = []
    i = 0
    while i < len(s):
        j = i
        while j < len(s) and s[j] == s[i]:
            j += 1
        out.append(f"{j - i}{s[i]}")
        i = j
    return "".join(out)


_GROUP = re.compile(r"([0-9]+)([^0-9])", re.DOTALL)


def rle_decode(s):
    if not isinstance(s, str):
        raise TypeError("s must be a str")
    out = []
    pos = 0
    while pos < len(s):
        m = _GROUP.match(s, pos)
        if m is None:
            raise ValueError(f"malformed run-length string at {pos}")
        count = int(m.group(1))
        if count < 1:
            raise ValueError("count must be at least 1")
        out.append(m.group(2) * count)
        pos = m.end()
    return "".join(out)
