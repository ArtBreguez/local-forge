import re

_ROMAN = re.compile(r"M{0,3}(?:CM|CD|D?C{0,3})(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{0,3})")
_VALUES = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}


def roman_to_int(s):
    if not isinstance(s, str):
        raise TypeError("s must be a str")
    if not s or _ROMAN.fullmatch(s) is None:
        raise ValueError(f"invalid roman numeral: {s!r}")
    total = 0
    for i, ch in enumerate(s):
        value = _VALUES[ch]
        if i + 1 < len(s) and _VALUES[s[i + 1]] > value:
            total -= value
        else:
            total += value
    return total
