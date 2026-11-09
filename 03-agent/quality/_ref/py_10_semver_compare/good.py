import re

_ID = r"(?:0|[1-9][0-9]*|[0-9]*[A-Za-z-][0-9A-Za-z-]*)"
_BUILD_ID = r"[0-9A-Za-z-]+"
_VERSION = re.compile(
    r"(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)"
    rf"(?:-({_ID}(?:\.{_ID})*))?"
    rf"(?:\+{_BUILD_ID}(?:\.{_BUILD_ID})*)?"
)


def _parse(text):
    if not isinstance(text, str):
        raise ValueError("version must be a string")
    m = _VERSION.fullmatch(text)
    if m is None:
        raise ValueError(f"invalid version: {text!r}")
    core = tuple(int(m.group(i)) for i in (1, 2, 3))
    pre = m.group(4).split(".") if m.group(4) else None
    return core, pre


def _cmp(x, y):
    return (x > y) - (x < y)


def compare_versions(a, b):
    core_a, pre_a = _parse(a)
    core_b, pre_b = _parse(b)
    if core_a != core_b:
        return _cmp(core_a, core_b)
    if pre_a is None and pre_b is None:
        return 0
    if pre_a is None:
        return 1
    if pre_b is None:
        return -1
    for x, y in zip(pre_a, pre_b):
        if x == y:
            continue
        x_num, y_num = x.isdigit(), y.isdigit()
        if x_num and y_num:
            return _cmp(int(x), int(y))
        if x_num:
            return -1
        if y_num:
            return 1
        return _cmp(x, y)
    return _cmp(len(pre_a), len(pre_b))
