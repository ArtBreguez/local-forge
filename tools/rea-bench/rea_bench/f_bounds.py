"""Bounds and off-by-one traps that compilers rewrite (range checks become `(unsigned)(v - lo) <= width`, loop counts become divisions).

Variants: an inclusive range written five different ways; an unsigned window `(unsigned)v + K < W`; a table lookup whose index
guard is `>=`, `>`, `< L + 1`, a wrapped subtraction or a two-sided test; the number of iterations a stepped loop runs for a
given n; a clamp; and the number of byte values a character-class test accepts. The truth is computed in Python from the
parameters, and the harness scans the whole domain of the C function and prints the first/last accepted value (or the count, or
the outputs at the edges) so an off-by-one in the source text fails the test.
"""
from __future__ import annotations

from .core import Task, c_bytes, dec, near, with_decoys

FAMILY = "bounds"


REWRITE = " The compiler may have rewritten the tests (a subtraction and a width, a bound one off); give the values the source intends."


def both(lo, hi):
    return {"score": "tokens", "accept": [dec(lo), dec(hi)]}


def span(lo, hi):
    """Facts for a lowest/highest answer: the low bound (or one off), and the high bound in any of the forms a range test takes after
    optimisation (hi, one off, the width hi - lo)."""
    return [near(lo), near(hi) + near(hi - lo)]


def lit(v):
    return f"({v})" if v < 0 else str(v)


def v_range_chain(rng):
    lo = rng.randint(-500, 4900)
    hi = lo + rng.randint(5, 600)
    form = rng.randrange(5)
    expr = [f"return v >= {lo} && v <= {hi};",
            f"return v > {lo - 1} && v < {hi + 1};",
            f"if (v < {lo}) return 0;\n    if (v > {hi}) return 0;\n    return 1;",
            f"return !(v < {lo} || v > {hi});",
            f"return (unsigned)v - (unsigned)({lo}) <= (unsigned)({hi - lo});"][form]
    body = f"int proc(int v) {{\n    {expr}\n}}"
    main = ('int main(void) {\n    int first = 0, last = 0, cnt = 0;\n    for (int v = -1000; v <= 6000; v++)\n'
            '        if (proc(v)) { if (!cnt) first = v; last = v; cnt++; }\n    printf("%d %d %d\\n", first, last, cnt);\n    return 0;\n}')
    return Task(kind=FAMILY, variant="range_chain", body=body, truth=both(lo, hi), params={"lo": lo, "hi": hi, "form": form},
                question="For which integers v does proc return 1? Give the lowest and the highest such value (inclusive, decimal)." + REWRITE,
                wrong=f"{lo + 1} and {hi}", check=(main, f"{lo} {hi} {hi - lo + 1}\n"), facts=span(lo, hi))


def v_window(rng):
    lo = rng.randint(-300, 300)
    w = rng.randint(4, 200)
    sub = f"(unsigned)v + {-lo}u" if lo < 0 else f"(unsigned)v - {lo}u"
    body = f"int proc(int v) {{\n    return {sub} < {w}u;\n}}"
    hi = lo + w - 1
    main = ('int main(void) {\n    int first = 0, last = 0, cnt = 0;\n    for (int v = -1000; v <= 1000; v++)\n'
            '        if (proc(v)) { if (!cnt) first = v; last = v; cnt++; }\n    printf("%d %d %d\\n", first, last, cnt);\n    return 0;\n}')
    return Task(kind=FAMILY, variant="window", body=body, truth=both(lo, hi), params={"lo": lo, "hi": hi, "w": w},
                question="For which integers v does proc return 1? Give the lowest and the highest such value (inclusive, decimal)." + REWRITE,
                wrong=f"{lo} and {hi + 1}", check=(main, f"{lo} {hi} {w}\n"), facts=[near(lo), near(w) + near(hi)])


def v_index_limit(rng):
    n = rng.choice([16, 24, 32, 48, 64])
    form = rng.randrange(6)
    limit = rng.randint(6, n - 4)                    # > a, so the two bounds of form 5 differ (lo == hi would accept the wrong answer)
    a = rng.randint(1, 5)
    b = rng.randint(3, n - 2 - a)
    guard, lo, hi = [
        (f"if (idx >= {limit}u) return -1;", 0, limit - 1),
        (f"if (idx > {limit}u) return -1;", 0, limit),
        (f"if (idx < {limit + 1}u) return T[idx];\n    return -1;", 0, limit),
        (f"if (idx - {a}u > {b}u) return -1;", a, a + b),
        (f"if (idx - {a}u >= {b}u) return -1;", a, a + b - 1),
        (f"if (idx > {limit}u || idx < {a}u) return -1;", a, limit),
    ][form]
    table = bytes(rng.randrange(256) for _ in range(n))
    tail = "" if form == 2 else "\n    return T[idx];"
    body = f"""int proc(unsigned idx) {{
    static const uint8_t T[{n}] = {{{c_bytes(table)}}};
    {guard}{tail}
}}"""
    main = (f'int main(void) {{\n    int first = -1, last = -1;\n    for (unsigned i = 0; i < {n + 20}; i++)\n'
            '        if (proc(i) != -1) { if (first < 0) first = (int)i; last = (int)i; }\n    printf("%d %d\\n", first, last);\n    return 0;\n}')
    return Task(kind=FAMILY, variant="index_limit", body=body, truth=both(lo, hi), params={"lo": lo, "hi": hi, "form": form, "n": n},
                question="proc looks a byte up in a table and returns -1 for an index it rejects. Give the smallest and the largest index for which it returns a table value (decimal)." + REWRITE,
                wrong=f"{lo} and {hi + 1}", check=(main, f"{lo} {hi}\n"), facts=([near(lo) + near(hi - lo)] if lo else []) + [near(hi) + near(hi - lo)])


def count_ref(form, a, s, n):
    if form == "up_lt":
        return len(range(a, n, s))
    if form == "up_le":
        return len(range(a, n + 1, s))
    if form == "down_gt":
        return len(range(n, a, -s))
    return len(range(n, a - 1, -s))


def v_loop_iterations(rng):
    form = rng.choice(["up_lt", "up_le", "down_gt", "down_ge"])
    a = rng.randint(0, 9)
    s = rng.randint(2, 9)
    loop = {"up_lt": f"for (int i = {a}; i < n; i += {s}) c++;", "up_le": f"for (int i = {a}; i <= n; i += {s}) c++;",
            "down_gt": f"for (int i = n; i > {a}; i -= {s}) c++;", "down_ge": f"for (int i = n; i >= {a}; i -= {s}) c++;"}[form]
    body = with_decoys(f"int proc(int n) {{\n    int c = 0;\n    {loop}\n    return c;\n}}", rng, rng.choice([1, 1, 2]), {a, s})
    n0 = a + rng.randint(5 * s, 60 * s)
    probes = [n0, 0, a, a + 1, n0 + 7]
    main = "int main(void) {\n" + "".join(f'    printf("%d\\n", proc({p}));\n' for p in probes) + "    return 0;\n}"
    want = count_ref(form, a, s, n0)
    return Task(kind=FAMILY, variant="loop_iterations", body=body, truth={"score": "tokens", "accept": [dec(want)]},
                params={"form": form, "a": a, "s": s, "n0": n0}, question=f"What does proc return when it is called with {n0} (decimal)?",
                wrong=str(want + 1), check=(main, "".join(f"{count_ref(form, a, s, p)}\n" for p in probes)))


def v_clamp(rng):
    lo = rng.randint(-500, 4000)
    hi = lo + rng.randint(5, 900)
    form = rng.randrange(3)
    body = ["int proc(int v) {\n    return v < LO ? LO : (v > HI ? HI : v);\n}",
            "int proc(int v) {\n    if (v > HI) v = HI;\n    if (v < LO) v = LO;\n    return v;\n}",
            "#define MAX(a, b) ((a) > (b) ? (a) : (b))\n#define MIN(a, b) ((a) < (b) ? (a) : (b))\nint proc(int v) {\n    return MIN(MAX(v, LO), HI);\n}"][form]
    body = body.replace("LO", lit(lo)).replace("HI", lit(hi))
    probes = [-1000, lo - 1, lo, (lo + hi) // 2, hi, hi + 1, 6000]
    main = "int main(void) {\n" + "".join(f'    printf("%d\\n", proc({lit(p)}));\n' for p in probes) + "    return 0;\n}"
    clamp = lambda v: min(max(v, lo), hi)
    return Task(kind=FAMILY, variant="clamp", body=body, truth=both(lo, hi), params={"lo": lo, "hi": hi, "form": form},
                question="proc limits its argument to an interval. Give the lower and the upper bound of that interval (inclusive, decimal).",
                wrong=f"{lo} and {hi - 1}", check=(main, "".join(f"{clamp(p)}\n" for p in probes)), facts=[near(lo), near(hi)])


def v_char_class(rng):
    ranges = []
    for _ in range(rng.randint(2, 3)):
        a = rng.randint(0x21, 0x70)
        ranges.append((a, a + rng.randint(1, 14)))
    single = rng.choice([None, 0x5F, 0x2D, 0x2E, 0x3A])

    def ch(v):
        c = chr(v)
        return f"'{c}'" if c.isalnum() else f"0x{v:02x}"

    def term(r):
        a, b = r
        return [f"(c >= {ch(a)} && c <= {ch(b)})", f"((unsigned)(c - {ch(a)}) <= {b - a}u)", f"(c > {a - 1} && c < {b + 1})",
                f"((unsigned)(c - {ch(a)}) < {b - a + 1}u)"][rng.randrange(4)]

    terms = [term(r) for r in ranges] + ([f"c == 0x{single:02x}"] if single else [])
    body = f"int proc(unsigned char c) {{\n    return {' || '.join(terms)};\n}}"
    accepted = set()
    for a, b in ranges:
        accepted |= set(range(a, b + 1))
    if single:
        accepted.add(single)
    main = 'int main(void) {\n    int cnt = 0;\n    for (int c = 0; c < 256; c++) cnt += proc((unsigned char)c) ? 1 : 0;\n    printf("%d\\n", cnt);\n    return 0;\n}'
    return Task(kind=FAMILY, variant="char_class", body=body, truth={"score": "tokens", "accept": [dec(len(accepted))]},
                params={"ranges": ranges, "single": single},
                question="How many different byte values (0 to 255) make proc return 1 (decimal)?" + REWRITE.replace("give the values the source intends", "count what the source accepts"),
                wrong=str(len(accepted) + 1), check=(main, f"{len(accepted)}\n"),
                facts=[g for a, b in ranges for g in (near(a), near(b) + near(b - a))] + ([[single]] if single else []))


VARIANTS = {"range_chain": v_range_chain, "window": v_window, "index_limit": v_index_limit, "loop_iterations": v_loop_iterations,
            "clamp": v_clamp, "char_class": v_char_class}
