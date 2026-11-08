import re

_VERSION = re.compile(r"(\d+)\.(\d+)\.(\d+)(?:-([0-9A-Za-z.-]+))?(?:\+[0-9A-Za-z.-]+)?")


def _parse(text):
    m = _VERSION.fullmatch(text)
    if m is None:
        raise ValueError(f"invalid version: {text!r}")
    return tuple(int(m.group(i)) for i in (1, 2, 3)), m.group(4)


def compare_versions(a, b):
    core_a, pre_a = _parse(a)
    core_b, pre_b = _parse(b)
    if core_a != core_b:
        return -1 if core_a < core_b else 1
    if pre_a == pre_b:
        return 0
    if pre_a is None:
        return 1
    if pre_b is None:
        return -1
    return -1 if pre_a < pre_b else 1
