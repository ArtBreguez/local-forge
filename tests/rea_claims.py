"""The checkers of the claims that the truth templates make (round 5, rewritten in round 6 to read the sentence).

tests/rea_sentences.py holds, for every sentence of every template, the PATTERN that parses the rendered sentence into named groups, and the names of the
claims that sentence makes. A claim is a function of a `Ctx` (the generated task, the record that was rendered, the groups PARSED FROM THE SENTENCE `c.s`, the
key of the template) that returns None when it holds and raises `ClaimFailed(why)` when it does not. Every number, name, byte order and direction word a claim
uses comes from `c.s`, that is from the text that is stored, and is compared with the task: its C compiled with gcc and RUN (a Python reference, a scan of the
domain, `offsetof`/`sizeof`, a fake libc call that records what proc asks for), or its C source parsed. Nothing is taken from the parameter that rendered the
sentence. What a sentence does not state (the other constants of a CRC, the packet layout) is read from the task record or the C.

Probes that depend only on the task are cached with `c.once(name, fn)` (shared by the sentences of a task); probes that depend on a value of the sentence go
through `c.run`, which caches by program text. A group that no claim reads fails the sentence (`Caps.unread`).
"""
from __future__ import annotations

import contextlib
import itertools
import os
import random
import re
import signal
import struct
import subprocess
import sys
import tempfile

from .helpers import ROOT

sys.path.insert(0, str(ROOT / "tools" / "rea-bench"))

from rea_bench import core, f_cipher, f_crc, f_encoding, f_pktsum  # noqa: E402

M32 = 0xFFFFFFFF
CLAIMS: dict = {}


class ClaimFailed(AssertionError):
    pass


def claim(fn):
    CLAIMS[fn.__name__] = fn
    return fn


def need(ok, why="") -> None:
    if not ok:
        raise ClaimFailed(why)


@contextlib.contextmanager
def time_limit(seconds: float):
    """Round 7, m4: a claim must FAIL, not hang, on a sentence that is false. A number of a swapped sentence (the rounds of a loop that became the multiplier, 1,103,515,245
    iterations) made `lcg_reference` run for minutes; now a claim that does not return within `seconds` is a failed claim with that reason (SIGALRM: the main thread of
    the process, which is where `exercise` runs, also in the workers of the claims test)."""
    def alarm(_signum, _frame):
        raise ClaimFailed(f"the claim did not return within {seconds:g} s: a number of the sentence makes it loop")
    try:
        old = signal.signal(signal.SIGALRM, alarm)
    except ValueError:                                               # not the main thread: no limit
        yield
        return
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


def small(value: int, lo: int, hi: int, what: str) -> int:
    """A number of the sentence that a claim loops on or uses as an exponent must be in the range the generators draw from (round 7, m4): otherwise the sentence is false."""
    need(lo <= value <= hi, f"{what} {value} is not in {lo} .. {hi}, the range of this kind of task: the sentence cannot be true")
    return value


class Ctx:
    def __init__(self, task, rec: dict, caps, key: str):
        self.t, self.rec, self.s, self.key = task, rec, caps, key
        self.cache: dict = {}

    def once(self, name, fn):
        """A probe that depends on the task only (never on a group of the sentence)."""
        if name not in self.cache:
            self.cache[name] = fn(self)
        return self.cache[name]

    @property
    def body(self) -> str:
        """The C of the task without its decoy statements."""
        return core.strip_decoys(self.t.body)

    @property
    def answer(self):
        """The accepted answer of the rendered question (a string for a string task, a list of groups for a tokens task)."""
        return self.rec["accept"]

    def run(self, main: str, pre: str = "", opt: str = "-O0") -> str:
        """Compile the task's source with `main` (and the preamble `pre`) with gcc and run it: its stdout."""
        key = ("run", main, pre, opt)
        if key not in self.cache:
            self.cache[key] = gcc_run(self.t.source(main, pre), opt)
        return self.cache[key]


def gcc_run(src: str, opt: str = "-O0") -> str:
    with tempfile.TemporaryDirectory() as d:
        exe = os.path.join(d, "x")
        r = subprocess.run(["gcc", opt, "-w", "-o", exe, "-x", "c", "-"], input=src, capture_output=True, text=True, timeout=60)
        if r.returncode:
            raise ClaimFailed("does not compile: " + r.stderr.strip()[:200])
        r = subprocess.run([exe], capture_output=True, text=True, timeout=60)
    if r.returncode:
        raise ClaimFailed(f"the program exits with {r.returncode}: {r.stderr.strip()[:120]}")
    return r.stdout


def num(s: str) -> int:
    s = s.strip()
    return int(s, 16) if s.lower().startswith(("0x", "-0x")) else int(s)


def ints(out: str) -> list:
    return [int(x) for x in out.split()]


def c_data(bs: bytes) -> str:
    return "{" + ", ".join(f"0x{b:02x}" for b in bs) + "}"


def c_str(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def c_bytes_str(b: bytes) -> str:
    """A C string literal of arbitrary bytes (no NUL), octal escapes so that nothing is read greedily."""
    return '"' + "".join(chr(x) if 0x20 <= x < 0x7f and chr(x) not in '"\\' else f"\\{x:03o}" for x in b) + '"'


LOW_WORDS = {"below", "smaller", "lower", "lowest", "smallest"}
HIGH_WORDS = {"above", "larger", "upper", "highest", "largest"}


def is_low(word: str) -> bool:
    w = word.lower()
    need(w in LOW_WORDS or w in HIGH_WORDS, f"{word!r} is no direction word")
    return w in LOW_WORDS


def bound_of(word: str, lo, hi):
    """The value that a direction word names when the range runs from lo to hi: lowest/smallest/lower -> lo, highest/largest/upper -> hi."""
    return lo if is_low(word) else hi


# ====================================================================================================================================
# the sweep of an int domain: -2**17 .. 2**17 one by one, then 2**32 in strides of 8191, then the extremes
# ====================================================================================================================================

def region_stats(c: Ctx, lo: int, hi: int) -> dict:
    """proc(int) over the sweep: (min, max, count) of its results on v < lo, v > lo, v < hi, v > hi, lo <= v <= hi, and on every v."""
    out = ints(c.run(f"""struct RS {{ long long mn, mx, n; }};
static struct RS rs[6];
static void upd(int k, long long r) {{ if (!rs[k].n || r < rs[k].mn) rs[k].mn = r; if (!rs[k].n || r > rs[k].mx) rs[k].mx = r; rs[k].n++; }}
static void visit(long long v) {{
    long long r = proc((int)v);
    upd(5, r);
    if (v < {lo}LL) upd(0, r);
    if (v > {lo}LL) upd(1, r);
    if (v < {hi}LL) upd(2, r);
    if (v > {hi}LL) upd(3, r);
    if (v >= {lo}LL && v <= {hi}LL) upd(4, r);
}}
int main(void) {{
    for (long long v = -131072; v <= 131072; v++) visit(v);
    for (long long v = -2147483648LL; v <= 2147483647LL; v += 8191) if (v < -131072 || v > 131072) visit(v);
    visit(-2147483647LL - 1); visit(2147483647LL); visit(-2147483647LL); visit(2147483646LL);
    for (int k = 0; k < 6; k++) printf("%lld %lld %lld\\n", rs[k].mn, rs[k].mx, rs[k].n);
    return 0;
}}"""))
    names = ("lt_lo", "gt_lo", "lt_hi", "gt_hi", "in", "all")
    return {n: tuple(out[3 * i:3 * i + 3]) for i, n in enumerate(names)}


def const_on(stats: dict, region: str, value: int, what: str) -> None:
    mn, mx, n = stats[region]
    need(n > 0, f"no integer was probed in the region {region}")
    need((mn, mx) == (value, value), f"proc returns {mn} .. {mx} on the integers of the region {region}, the sentence says {value} ({what})")


# ====================================================================================================================================
# structs
# ====================================================================================================================================

def _stride_probe(c: Ctx):
    b = c.t.body
    st = re.search(r"proc\(const struct (\w+) \*a, size_t n\)", b).group(1)
    fld = re.search(r"s \+= \(uint64_t\)a\[i\]\.(\w+);", b).group(1)
    out = c.run(f"""int main(void) {{
    struct {st} a[4];
    memset(a, 0, sizeof a);
    for (int i = 0; i < 4; i++) a[i].{fld} = (__typeof__(a[0].{fld}))((i + 1) * 16);
    printf("%zu %llu %llu\\n", sizeof(struct {st}), (unsigned long long)proc(a, 4), (unsigned long long)proc(a, 2));
    return 0;
}}""")
    return ints(out)


@claim
def stride_records(c: Ctx):
    size, four, two = c.once("stride", _stride_probe)
    n = c.s["n"]
    need(size == n, f"sizeof(record) is {size}, the sentence says {n}")
    need((four, two) == (16 + 32 + 48 + 64, 16 + 32), f"proc does not read consecutive records: sums {four}, {two}")


@claim
def field_read_offset(c: Ctx):
    b = c.t.body
    m = re.search(r"proc\(const struct (\w+) \*p\)", b)
    g, fld, k = re.search(r"if \(p->(\w+) == 0\) return 0;\n    return \(uint64_t\)p->(\w+) \+ (\d+);", b).groups()
    need(g != fld, "the flag field is the returned field")
    out = ints(c.run(f"""int main(void) {{
    struct {m.group(1)} s;
    memset(&s, 0, sizeof s);
    s.{fld} = (__typeof__(s.{fld}))0x2a;
    printf("%zu\\n", offsetof(struct {m.group(1)}, {fld}));
    s.{g} = 0;
    printf("%llu\\n", (unsigned long long)proc(&s));
    unsigned long long bad = 0, tried = 0;
    for (size_t bit = 0; bit < sizeof s.{g} * 8; bit++) {{
        s.{g} = (__typeof__(s.{g}))(1ULL << bit);
        tried++;
        if (proc(&s) != 0x2a + {k}) bad++;
    }}
    s.{g} = (__typeof__(s.{g}))-1;
    tried++;
    if (proc(&s) != 0x2a + {k}) bad++;
    if (sizeof s.{g} == 1) for (unsigned v = 1; v < 256; v++) {{
        s.{g} = (__typeof__(s.{g}))v;
        tried++;
        if (proc(&s) != 0x2a + {k}) bad++;
    }}
    printf("%llu %llu\\n", bad, tried);
    return 0;
}}"""))
    off, zero, bad, tried = out
    need(off == c.s["off"], f"the returned field is at offset {off:#x}, the sentence says {c.s['off']:#x}")
    if c.s["flag"] == "zero":
        need(zero == 0x2a + int(k), f"proc returns {zero} for a zero flag, the sentence says the field plus a constant")
    else:
        need(bad == 0 and tried >= 9, f"proc does not return the field + {k} for {bad} of {tried} non-zero flag values (every single bit, all ones, every value of a byte flag)")
        need(zero == 0, f"proc returns {zero} for a zero flag")


def _init_probe(c: Ctx, magic: int):
    b = c.t.body
    st = re.search(r"void proc\(struct (\w+) \*p\)", b).group(1)
    stores = re.findall(r"p->(\w+) = (0x[0-9a-f]+)(?:u|ull)?;", b)
    hit = [f for f, lit in stores if int(lit, 16) == magic]
    need(len(hit) == 1, f"{len(hit)} stores in the C write the constant {magic:#x} of the sentence")
    f = hit[0]
    runs = []
    for poison in (0xAA, 0x55):
        runs.append(ints(c.run(f"""int main(void) {{
    struct {st} s;
    memset(&s, {poison}, sizeof s);
    proc(&s);
    size_t off = offsetof(struct {st}, {f}), len = sizeof s.{f}, other = 0;
    for (size_t i = 0; i < sizeof s; i++)
        if ((i < off || i >= off + len) && ((unsigned char *)&s)[i] != {poison}) other++;
    printf("%zu %llu %zu\\n", off, (unsigned long long)s.{f}, other);
    return 0;
}}""")))
    return runs


@claim
def init_magic_offset(c: Ctx):
    magic = c.s["magic"]
    for off, val, _other in _init_probe(c, magic):
        need(off == c.s["off"], f"the field that receives {magic:#x} is at {off:#x}, the sentence says {c.s['off']:#x}")
        need(val == magic, f"proc stores {val:#x}, not {magic:#x}")


@claim
def init_other_fields(c: Ctx):
    need(max(r[2] for r in _init_probe(c, c.s["magic"])) > 0, "proc writes no other field than the one that receives the constant")


def _alloc_probe(c: Ctx):
    st = re.search(r"struct (\w+) \*proc\(uint32_t id\)", c.t.body).group(1)
    pre = """static size_t el[16], ct[16];
static int calls;
static void *my_malloc(size_t n) { if (calls < 16) { el[calls] = n; ct[calls] = 1; } calls++; return __builtin_calloc(1, n); }
static void *my_calloc(size_t c, size_t n) { if (calls < 16) { el[calls] = n; ct[calls] = c; } calls++; return __builtin_calloc(c, n); }
#define malloc my_malloc
#define calloc my_calloc
"""
    return ints(c.run(f'int main(void) {{ proc(7); printf("%zu %d", sizeof(struct {st}), calls); for (int i = 0; i < calls && i < 16; i++) printf(" %zu %zu", el[i], ct[i]); '
                      'printf("\\n"); return 0; }', pre))


@claim
def alloc_structure_size(c: Ctx):
    out = c.once("alloc", _alloc_probe)
    size, calls, rest = out[0], out[1], out[2:]
    n = c.s["n"]
    need(size == n, f"sizeof(structure) is {size}, the sentence says {n}")
    need(calls >= 1 and len(rest) == 2 * min(calls, 16), "proc makes no allocation")
    need(all(rest[i] == size and rest[i + 1] >= 1 for i in range(0, len(rest), 2)), f"proc allocates elements of {rest[::2]} bytes (counts {rest[1::2]}), the structure is {size}")


@claim
def nested_field_offset(c: Ctx):
    b = c.t.body
    st = re.search(r"proc\(const struct (\w+) \*o\)", b).group(1)
    fld = re.search(r"return \(uint64_t\)o->sub\.(\w+);", b).group(1)
    off, got = ints(c.run(f"""int main(void) {{
    struct {st} o;
    memset(&o, 0, sizeof o);
    o.sub.{fld} = (__typeof__(o.sub.{fld}))0x2a;
    printf("%zu %llu\\n", offsetof(struct {st}, sub.{fld}), (unsigned long long)proc(&o));
    return 0;
}}"""))
    need(off == c.s["off"], f"the nested field is at {off:#x} of the outer structure, the sentence says {c.s['off']:#x}")
    need(got == 0x2a, "proc does not return the nested field")


@claim
def linked_pointer_offset(c: Ctx):
    b = c.t.body
    st = re.search(r"proc\(const struct (\w+) \*n\)", b).group(1)
    vf, nf = re.search(r"s \+= \(uint64_t\)n->(\w+);\n        n = n->(\w+);", b).groups()
    off, got = ints(c.run(f"""int main(void) {{
    struct {st} a[3];
    memset(a, 0, sizeof a);
    for (int i = 0; i < 3; i++) a[i].{vf} = (__typeof__(a[0].{vf}))((i + 1) * 16);
    a[0].{nf} = &a[1];
    a[1].{nf} = &a[2];
    a[2].{nf} = NULL;
    printf("%zu %llu\\n", offsetof(struct {st}, {nf}), (unsigned long long)proc(&a[0]));
    return 0;
}}"""))
    need(off == c.s["off"], f"the pointer to the next node is at {off:#x}, the sentence says {c.s['off']:#x}")
    need(got == 16 + 32 + 48, f"proc visits the nodes {got:#x}: it does not follow the pointer to the end of the list")


# ====================================================================================================================================
# fsm
# ====================================================================================================================================

def parse_machine(b: str):
    mt = re.search(r'static const char alpha\[\] = "(\w+)";\s+static const uint8_t T\[(\d+)\]\[(\d+)\] = \{\n(.*?)\n    \};', b, re.S)
    if mt:
        rows = re.findall(r"\{([\d, ]+)\}", mt.group(4))
        return mt.group(1), [[int(x) for x in row.split(",")] for row in rows]
    cases = re.findall(r"case (\d+): (.*?) break;", b)
    need(cases, "no machine in the C")
    delta, alpha = [], None
    for st, chain in cases:
        ch = re.findall(r"c == '(\w)'\) st = (\d+);", chain)
        alpha = alpha or "".join(x for x, _ in ch)
        need("".join(x for x, _ in ch) == alpha and int(st) == len(delta), "the cases of the switch are not the machine of the generator")
        delta.append([int(x) for _, x in ch])
    return alpha, delta


def _sim(alpha, delta, s):
    st = 0
    for ch in s:
        st = delta[st][alpha.index(ch)]
    return st


def _bfs(delta, start=0, paths=False):
    dist, front, path = {start: 0}, [start], {start: ""}
    while front:
        nxt = []
        for s in front:
            for i, t in enumerate(delta[s]):
                if t not in dist:
                    dist[t] = dist[s] + 1
                    path[t] = path[s] + str(i)
                    nxt.append(t)
        front = nxt
    return (dist, path) if paths else dist


def _fsm_probe(c: Ctx):
    """The machine parsed from the C, proven equal to the compiled proc: proc returns the state it ends in, so agreement on the empty string, on a path to every
    reachable state followed by each letter, and on random strings is agreement on every string (induction on the length)."""
    alpha, delta = parse_machine(c.body)
    dist, path = _bfs(delta, 0, paths=True)
    rng = random.Random(c.t.id)
    strings = [""] + ["".join(rng.choice(alpha) for _ in range(rng.randint(1, 9))) for _ in range(40)]
    for s in sorted(path):
        word = "".join(alpha[int(i)] for i in path[s])
        strings += [word + ch for ch in alpha]
    if c.key.endswith("_input"):
        strings.append(c.answer[0][0])
    out = c.run("int main(void) {\n" + "".join(f'    printf("%d\\n", proc({c_str(s)}));\n' for s in strings) + "    return 0;\n}")
    got = ints(out)
    need(got == [_sim(alpha, delta, s) for s in strings], "the machine parsed from the C is not what the compiled proc computes")
    return alpha, delta, strings, got


@claim
def fsm_input_final(c: Ctx):
    alpha, delta, strings, got = c.once("fsm", _fsm_probe)
    s = strings[-1]
    need(got[-1] == c.s["final"], f"proc returns {got[-1]} for the input, the sentence says state {c.s['final']}")
    need(c.s["d"] == len(s) and set(s) <= set(alpha), f"the input has {len(s)} letters, the sentence says {c.s['d']}")
    if "z" in c.s:
        need(got[0] == c.s.get("z"), f"proc starts in state {got[0]} (its value for the empty input), the sentence says {c.s['z']}")


@claim
def fsm_input_string(c: Ctx):
    """A headline that gives the input string: the string is made of letters of the machine's alphabet and the compiled proc returns the state that the question asks
    for (the machine is proven equal to proc by `_fsm_probe`; the string itself is run too)."""
    alpha, delta, strings, got = c.once("fsm", _fsm_probe)
    s, final = c.s["s"], c.s["final"]
    need(s and set(s) <= set(alpha), f"the input {s!r} has a letter that the machine does not read (its alphabet is {alpha!r})")
    out = ints(c.run('int main(void) { printf("%d\\n", proc(' + c_str(s) + ')); return 0; }'))
    need(out == [final], f"proc returns {out} for the input {s!r}, the question asks for the state {final}")
    need(_sim(alpha, delta, s) == final, f"the machine of the C ends in state {_sim(alpha, delta, s)} on the input {s!r}, not {final}")


@claim
def fsm_input_shortest(c: Ctx):
    alpha, delta, strings, got = c.once("fsm", _fsm_probe)
    dist = _bfs(delta, got[0])
    need(dist.get(c.s["final"]) == c.s["d"], f"the shortest way to state {c.s['final']} is {dist.get(c.s['final'])} letters, the sentence says {c.s['d']}")
    need(c.s["d"] == len(strings[-1]), "the input has another length than the sentence says")


@claim
def fsm_states_count(c: Ctx):
    alpha, delta, strings, got = c.once("fsm", _fsm_probe)
    z, n = c.s["z"], c.s["n"]
    last = c.s.get("last", z + n - 1)
    need(len(delta) == n, f"the machine has {len(delta)} states, the sentence says {n}")
    need(got[0] == z, f"proc starts in state {got[0]}, the sentence says {z}")
    need(sorted(_bfs(delta, got[0])) == list(range(z, last + 1)), f"the states reachable from the start are not numbered {z} .. {last}")


@claim
def fsm_states_reachable(c: Ctx):
    alpha, delta, strings, got = c.once("fsm", _fsm_probe)
    need(got[0] == c.s["z"], f"proc starts in state {got[0]}, the sentence says {c.s['z']}")
    reach = _bfs(delta, got[0])
    need(len(reach) == c.s["n"], f"{len(reach)} of the {c.s['n']} states are reachable from the start")


PAIRS = [(0x1234abcd, 7), (0xdeadbeef, 0x12345679), (0x0f0f0f0f, 3)]


def _opcode_probe(c: Ctx):
    """proc(op, a, b) for op 0..63, and for a few far values of op; returns (rows, far rows)."""
    far = [64, 1000, 65535, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFE, 0xFFFFFFFF]
    ops = list(range(64)) + far
    out = c.run("int main(void) {\n    static const uint32_t ops[] = {" + ", ".join(f"{o}u" for o in ops) + "};\n    for (size_t i = 0; i < sizeof ops / sizeof ops[0]; i++) {\n"
                + "".join(f'        printf("%u ", proc(ops[i], 0x{a:x}u, {b}u));\n' for a, b in PAIRS) + '        printf("\\n");\n    }\n    return 0;\n}')
    rows = [ints(line) for line in out.strip().split("\n")]
    need(len(rows) == len(ops), "the scan did not run")
    return rows[:64], rows[64:]


def _opcode_sweep(c: Ctx, vec: list, equal: bool) -> int:
    """How many op >= 64 (every one up to 65535, then in strides of 8191, and the extremes) give the result triple `vec` (equal) or something else (not equal)."""
    cond = " && ".join(f"proc(op, 0x{a:x}u, {b}u) == {v}u" for (a, b), v in zip(PAIRS, vec))
    test = f"({cond})" if equal else f"!({cond})"
    return ints(c.run(f"""int main(void) {{
    unsigned long long hits = 0, op;
    for (op = 64; op <= 0xFFFFFFFFull; op += (op < 65536 ? 1 : 8191)) if ({test}) hits++;
    printf("%llu\\n", hits);
    return 0;
}}"""))[0]


def _eval_expr(expr: str, a: int, b: int) -> int:
    need(re.fullmatch(r"[ab0-9 ^&|+*()<>-]+", expr) is not None, f"{expr!r} is not an expression over a and b")
    need("**" not in expr and all(int(n) <= 64 for n in re.findall(r"<<\s*(\d+)", expr)), f"{expr!r}: an exponent or a shift that is not small would make eval run for ever")
    return eval(expr, {"__builtins__": {}}, {"a": a, "b": b}) & M32


@claim
def opcode_returns_expr(c: Ctx):
    rows, far = c.once("opcodes", _opcode_probe)
    n, expr = c.s["n"], c.s["expr"].strip()
    want = [_eval_expr(expr, a, b) for a, b in PAIRS]
    need(0 <= n < 64 and rows[n] == want, f"proc({n}, a, b) returns {rows[n] if 0 <= n < 64 else '?'}, not {expr}: {want}")


@claim
def opcode_unique(c: Ctx):
    rows, far = c.once("opcodes", _opcode_probe)
    n, expr = c.s["n"], c.s["expr"].strip()
    want = [_eval_expr(expr, a, b) for a, b in PAIRS]
    ops = [op for op, r in enumerate(rows) if r == want] + [f for f, r in zip((64, 1000, 65535, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFE, 0xFFFFFFFF), far) if r == want]
    need(ops == [n], f"the opcodes that return {expr} are {ops}")
    need(_opcode_sweep(c, want, True) == 0, f"an opcode above 63 returns {expr}")


def _handled(c: Ctx):
    rows, far = c.once("opcodes", _opcode_probe)
    labels = {int(x) for x in re.findall(r"case (\d+):", c.body)}
    d0 = far[-1]
    handled = {op for op, r in enumerate(rows) if r != d0}
    need(handled == labels, f"the opcodes that do not return the default are {sorted(handled)}, the cases of the C are {sorted(labels)}")
    return rows, far, labels, d0


@claim
def opcode_count_handled(c: Ctx):
    rows, far, labels, d0 = _handled(c)
    need(len(labels) == c.s["n"], f"{len(labels)} opcodes are handled, the sentence says {c.s['n']}")
    need(len({tuple(rows[op]) for op in labels}) == len(labels), "two handled opcodes return the same results: they have no result of their own")
    need(_opcode_sweep(c, d0, False) == 0, "an opcode above 63 is handled")


@claim
def opcode_default(c: Ctx):
    rows, far, labels, d0 = _handled(c)
    dflt = c.s["dflt"]
    need(d0 == [dflt] * len(PAIRS), f"an unhandled opcode returns {d0}, the sentence says {dflt}")
    need(all(r == d0 for op, r in enumerate(rows) if op not in labels) and all(r == d0 for r in far), "the unhandled opcodes do not all return the default")
    need(_opcode_sweep(c, [dflt] * len(PAIRS), False) == 0, f"an opcode above 63 returns something other than {dflt}")


# ====================================================================================================================================
# crc
# ====================================================================================================================================

def _crc_probe(c: Ctx):
    t, p = c.t, c.t.params
    w = p["width"]
    d = w // 4
    rng = random.Random(t.id)
    vecs = [b"123456789", bytes(rng.randrange(256) for _ in range(rng.randint(8, 24))), b"", bytes([rng.randrange(256)])]
    main = ("int main(void) {\n" + "".join(f"    static const uint8_t v{i}[] = {c_data(v) if v else '{0}'};\n    printf(\"%0{d}x\\n\", (unsigned)proc(v{i}, {len(v)}));\n"
                                          for i, v in enumerate(vecs)) + "    return 0;\n}")
    got = [int(x, 16) for x in c.run(main).split()]
    return vecs, got


def _crc_matches(c: Ctx, *, width, poly_lit, refl, init, xorout):
    """The compiled proc equals the CRC with these parameters, and the register the C declares has this width (a reflected register with 16-bit constants
    computes the same values at 32 bits, so the values alone do not tell)."""
    vecs, got = c.once("crc", _crc_probe)
    reg = re.search(r"\buint(\d+)_t proc\(const uint8_t \*p, size_t n\)", c.body)
    need(reg and int(reg.group(1)) == width, f"the C computes in a {reg.group(1) + '-bit' if reg else 'unknown'} register, the sentence says {width}-bit")
    poly = f_crc.reflect(poly_lit, width) if refl else poly_lit          # the constant of a reflected CRC is the reflected polynomial
    want = [f_crc.crc_ref(v, width, poly, init, refl, xorout) for v in vecs]
    need(got == want, f"proc computes {[hex(x) for x in got]}, a {width}-bit CRC with these parameters gives {[hex(x) for x in want]}")


@claim
def crc_poly_form(c: Ctx):
    p = c.t.params
    _crc_matches(c, width=c.s["w"], poly_lit=c.s["lit"], refl=c.s["dir"] == "reflected", init=p["init"], xorout=p["xorout"])


@claim
def crc_init_value(c: Ctx):
    p = c.t.params
    _crc_matches(c, width=c.s["w"], poly_lit=p["lit"], refl=p["refl"], init=c.s["x"], xorout=p["xorout"])


@claim
def crc_xorout_value(c: Ctx):
    p = c.t.params
    _crc_matches(c, width=c.s["w"], poly_lit=p["lit"], refl=p["refl"], init=p["init"], xorout=c.s["x"])


# ====================================================================================================================================
# cipher
# ====================================================================================================================================

def tea_ref(alg, v, k, rounds, delta, decrypt):
    """TEA and XTEA with the delta as a parameter (f_cipher's references fix it): a sentence that states a delta is compared with THIS function."""
    v0, v1 = v
    if not decrypt:
        s = 0
        for _ in range(rounds):
            if alg == "TEA":
                s = (s + delta) & M32
                v0 = (v0 + ((((v1 << 4) & M32) + k[0]) ^ (v1 + s) ^ ((v1 >> 5) + k[1]))) & M32
                v1 = (v1 + ((((v0 << 4) & M32) + k[2]) ^ (v0 + s) ^ ((v0 >> 5) + k[3]))) & M32
            else:
                v0 = (v0 + ((((v1 << 4) & M32) ^ (v1 >> 5)) + v1 ^ (s + k[s & 3]))) & M32
                s = (s + delta) & M32
                v1 = (v1 + ((((v0 << 4) & M32) ^ (v0 >> 5)) + v0 ^ (s + k[(s >> 11) & 3]))) & M32
    else:
        s = (delta * rounds) & M32
        for _ in range(rounds):
            if alg == "TEA":
                v1 = (v1 - ((((v0 << 4) & M32) + k[2]) ^ (v0 + s) ^ ((v0 >> 5) + k[3]))) & M32
                v0 = (v0 - ((((v1 << 4) & M32) + k[0]) ^ (v1 + s) ^ ((v1 >> 5) + k[1]))) & M32
                s = (s - delta) & M32
            else:
                v1 = (v1 - ((((v0 << 4) & M32) ^ (v0 >> 5)) + v0 ^ (s + k[(s >> 11) & 3]))) & M32
                s = (s - delta) & M32
                v0 = (v0 - ((((v1 << 4) & M32) ^ (v1 >> 5)) + v1 ^ (s + k[s & 3]))) & M32
    return v0, v1


@claim
def tea_reference(c: Ctx):
    s, b = c.s, c.body
    variant = c.key.split("/")[1]                                    # tea_enc, xtea_dec ...: what the question's variant is
    alg = s.get("alg", variant.split("_")[0].upper())
    if "verb" in s:
        need(s["verb"] in ("encrypts", "decrypts"), "the verb")
        direction = s["verb"][:-1]
    elif "done" in s:
        need(s["done"] in ("encrypted", "decrypted"), "the participle")
        direction = s["done"][:-2]
    elif "dir" in s:                                                 # a headline: "TEA, encrypt, 32"
        need(s["dir"] in ("encrypt", "decrypt"), "the direction")
        direction = s["dir"]
    else:
        direction = "decrypt" if variant.endswith("dec") else "encrypt"
    need(alg in ("TEA", "XTEA"), f"the sentence names {alg}, proc is a TEA or XTEA")
    delta = s.get("delta", 0x9E3779B9)
    key = [int(x, 16) for x in re.search(r"k\[4\] = \{(.*?)\}", b).group(1).replace("u", "").split(",")]
    if "key_bits" in s:
        need(len(key) * 32 == s["key_bits"], f"the key has {len(key) * 32} bits, the sentence says {s['key_bits']}")
    if "block" in s:
        need(re.search(r"proc\(uint32_t v\[2\]\)", b) and 2 * 4 == s["block"], f"the block is two 32-bit words, the sentence says {s['block']} bytes")
    rounds = small(s["rounds"], 1, 1024, "the iterations")
    rng = random.Random(c.t.id)
    blocks = [(rng.getrandbits(32), rng.getrandbits(32)) for _ in range(4)]
    main = ("int main(void) {\n" + "".join(
        f'    {{ uint32_t v[2] = {{0x{x:x}u, 0x{y:x}u}}; unsigned r = proc(v); printf("%u %u %u\\n", (unsigned)v[0], (unsigned)v[1], r); }}\n' for x, y in blocks)
        + "    return 0;\n}")
    got = [ints(line) for line in c.run(main).strip().split("\n")]
    want = [list(tea_ref(alg, blk, key, rounds, delta, direction == "decrypt")) + [rounds] for blk in blocks]
    need(got == want, f"proc is not {alg} {direction}ion with {rounds} iterations and delta {delta:#x}: {got[:1]} vs {want[:1]}")
    other = list(tea_ref(alg, blocks[0], key, rounds + 1, delta, direction == "decrypt"))
    need(other != got[0][:2], "one more iteration gives the same block: the round count is not tested")


@claim
def rc4_reference(c: Ctx):
    s, b = c.s, c.body
    mt = re.search(r'static const char key\[\] = "(\w+)";', b)
    key = mt.group(1).encode() if mt else bytes(int(x, 16) for x in re.search(r"static const uint8_t key\[\] = \{(.*?)\};", b).group(1).replace(" ", "").split(","))
    need(len(key) == s["n"], f"the key has {len(key)} bytes, the sentence says {s['n']}")
    if "state" in s:
        sz = re.search(r"uint8_t S\[(\d+)\]", b)
        need(sz and int(sz.group(1)) == s["state"], f"the state of the C has {sz.group(1) if sz else '?'} byte values, the sentence says {s['state']}")
    if "alg" in s:
        need(s["alg"] == "RC4", f"the sentence names {s['alg']}, the reference that proc equals is RC4")
    rng = random.Random(c.t.id)
    data = bytes(rng.randrange(256) for _ in range(40))
    out = c.run(f"""int main(void) {{
    uint8_t buf[] = {c_data(data)};
    proc(buf, sizeof buf);
    for (size_t i = 0; i < sizeof buf; i++) printf("%02x", buf[i]);
    printf("\\n");
    return 0;
}}""")
    need(out.strip() == f_cipher.rc4(key, data).hex(), "proc is not RC4 with this key")


STREAM_ROT = {"ChaCha": (16, 12, 8, 7), "Salsa": (7, 9, 13, 18)}
STREAM_REF = {"ChaCha": f_cipher.chacha_block, "Salsa": f_cipher.salsa_block}


@claim
def stream_reference(c: Ctx):
    s, b = c.s, c.body
    alg = s.get("alg", {"chacha": "ChaCha", "salsa": "Salsa"}[c.key.split("/")[1]])
    need(alg in STREAM_REF, f"{alg} is no stream cipher of this family")
    rounds = small(s["rounds"], 1, 1024, "the rounds")
    if "rot0" in s:
        need((s["rot0"], s["rot1"], s["rot2"], s["rot3"]) == STREAM_ROT[alg], f"the rotation amounts of {alg} are {STREAM_ROT[alg]}, the sentence says "
             f"{(s['rot0'], s['rot1'], s['rot2'], s['rot3'])}")
    if "half" in s:
        need(s["half"] * 2 == rounds, f"{rounds} rounds are {rounds / 2} double rounds, the sentence says {s['half']}")
    if "key_bits" in s:
        kw = int(re.search(r"const uint32_t key\[(\d+)\]", b).group(1))
        need(kw * 32 == s["key_bits"], f"the key has {kw * 32} bits, the sentence says {s['key_bits']}")
    if "block" in s:
        ow = int(re.search(r"uint32_t out\[(\d+)\]", b).group(1))
        need(ow * 4 == s["block"], f"the block has {ow * 4} bytes, the sentence says {s['block']}")
    nn = int(re.search(r"const uint32_t nonce\[(\d)\]", b).group(1))
    rng = random.Random(c.t.id)
    key = [rng.getrandbits(32) for _ in range(8)]
    nonce = [rng.getrandbits(32) for _ in range(nn)]
    counter = rng.getrandbits(32)
    out = c.run(f"""int main(void) {{
    static const uint32_t key[8] = {{{core.c_u32s(key)}}};
    static const uint32_t nonce[{nn}] = {{{core.c_u32s(nonce)}}};
    uint32_t out[16];
    unsigned r = proc(out, key, 0x{counter:08x}u, nonce);
    for (int i = 0; i < 16; i++) printf("%u ", (unsigned)out[i]);
    printf("%u\\n", r);
    return 0;
}}""")
    ref = STREAM_REF[alg](key, counter, nonce, rounds)
    need(ints(out) == ref + [rounds], f"proc is not {alg} with {rounds} rounds")
    other = STREAM_REF[alg](key, counter, nonce, rounds + 2)
    need(other != ref, "two more rounds give the same block: the round count is not tested")


# ====================================================================================================================================
# encoding
# ====================================================================================================================================

def _enc_alphabet(body: str) -> str:
    m = re.search(r'static const char A\[\] = "((?:[^"\\]|\\.)*)";', body)
    return m.group(1).replace('\\"', '"').replace("\\\\", "\\")


def _enc_cover(base: str, alpha: str, pad: bool, rng) -> list:
    """Inputs until every character of the alphabet has appeared in an encoding: a check that the character at position i stands for the value i."""
    seen, vecs = set(), []
    for _ in range(600):
        if seen == set(alpha):
            break
        v = bytes(rng.randrange(256) for _ in range(rng.randint(1, 9)))
        enc = f_encoding.encode_ref(base, v, alpha, pad)
        if set(enc) - seen:
            vecs.append(v)
            seen |= set(enc)
    need(seen >= set(alpha), "the covering inputs do not reach every character of the alphabet")
    return vecs


def _enc_probe(c: Ctx):
    t = c.t
    base, direction = t.variant.split("_")
    alpha, pad = _enc_alphabet(c.body), t.params["pad"]
    rng = random.Random(t.id)
    gin = {"base64": 3, "base32": 5}.get(base)
    vecs = f_encoding.vectors(rng) + [b"A", b"AB", b"ABC"] + _enc_cover(base, alpha, pad, rng)
    if gin:                                                              # one whole group, then two
        vecs += [bytes(rng.randrange(256) for _ in range(gin)), bytes(rng.randrange(256) for _ in range(2 * gin))]
    if direction == "encode":
        decls = "".join(f"    static const uint8_t v{i}[] = {c_data(v)};\n" for i, v in enumerate(vecs))
        calls = "".join(f"    proc(v{i}, sizeof v{i}, out); puts(out);\n" for i in range(len(vecs)))
        got = c.run(f"int main(void) {{\n{decls}    char out[256];\n{calls}    return 0;\n}}").split()
    else:
        enc = [f_encoding.encode_ref(base, v, alpha, pad) for v in vecs]
        calls = "".join(f'    n = proc({c_str(e)}, out);\n    for (size_t i = 0; i < n; i++) printf("%02x", out[i]);\n    printf("\\n");\n' for e in enc)
        got = c.run(f"int main(void) {{\n    uint8_t out[128];\n    size_t n;\n{calls}    return 0;\n}}").split()
    return base, direction, alpha, pad, vecs, got


def _enc_matches(c: Ctx, alpha: str):
    base, direction, own, pad, vecs, got = c.once("enc", _enc_probe)
    need(alpha == own, "the alphabet of the sentence is not the one of the C")
    if direction == "encode":
        want = [f_encoding.encode_ref(base, v, alpha, pad) for v in vecs]
        need(all(set(g) <= set(alpha) | {"="} for g in got), "an output character is not a character of the alphabet")
    else:
        want = [v.hex() for v in vecs]
    need(got == want, f"proc is not {base} {direction} with this alphabet: {got[:2]} vs {want[:2]}")
    return base, direction, vecs, got


@claim
def encoding_reference(c: Ctx):
    base, direction, alpha, pad, vecs, got = c.once("enc", _enc_probe)
    s = c.s
    need(len(set(alpha)) == len(alpha) == s["radix"], f"the alphabet has {len(alpha)} characters, the sentence says {s['radix']}")
    _enc_matches(c, alpha)
    if "bits" in s:
        need(2 ** small(s["bits"], 0, 64, "the bits per character") == s["radix"], f"{s['bits']} bits do not make a radix of {s['radix']}")
        need(direction in ("encode", "decode") and base in ("base64", "base32"), "only base64 and base32 have a fixed number of bits per character")
    if "gin" in s:
        small(s["gin"], 1, 64, "the group of input bytes")
        bits = s.get("bits", s["radix"].bit_length() - 1)
        need(2 ** bits == s["radix"], f"a character of {bits} bits is not one of {s['radix']}")
        need(s["gin"] * 8 == s["gout"] * bits, f"{s['gin']} bytes are {s['gin'] * 8} bits, {s['gout']} characters of {bits} bits are {s['gout'] * bits}")
        need(direction == "encode", "a decode sentence states the group sizes of the encoder")
        # a whole group, measured on the compiled proc: gin input bytes give exactly gout output characters (no padding)
        out = ints(c.run(f"int main(void) {{ static const uint8_t v[] = {c_data(bytes(0x41 + i for i in range(s['gin'])))}; char out[256]; proc(v, sizeof v, out); "
                         'printf("%zu\\n", strlen(out)); return 0; }'))
        need(out == [s["gout"]], f"{s['gin']} input bytes become {out[0]} output characters, the sentence says {s['gout']}")


@claim
def encoding_identity(c: Ctx):
    """A headline that names the direction and the encoding ("decode, base32"): the compiled proc is that encoder or decoder of the reference, with the alphabet
    of its own C (the alphabet has the radix of the encoding)."""
    base, direction, alpha, _pad, _v, _g = c.once("enc", _enc_probe)
    need(len(set(alpha)) == len(alpha) == f_encoding.RADIX[c.s["base"]], f"the alphabet of the C has {len(alpha)} characters, {c.s['base']} has {f_encoding.RADIX[c.s['base']]}")
    _enc_matches(c, alpha)                                           # proc IS `base` `direction` (the variant of the task), compared with the reference
    need((c.s["dir"], c.s["base"]) == (direction, base), f"the headline says {c.s['dir']} {c.s['base']}, the compiled proc does {direction} {base}")


@claim
def alphabet_semantics(c: Ctx):
    base, direction, alpha, _pad, _v, _g = c.once("enc", _enc_probe)
    s = c.s
    if "s" in s:                                                     # a headline: the alphabet string itself
        need(s["s"] == alpha, f"the alphabet of the headline is not the one of the C, byte for byte: {s['s']!r} vs {alpha!r}")
    need(s["alen"] == len(alpha) == f_encoding.RADIX[base], f"the alphabet has {len(alpha)} characters, the sentence says {s['alen']}")
    need(c.answer[0] == alpha, "the alphabet of the answer is not the one of the C, byte for byte")
    if "agent" in s:
        need(s["agent"] == ("encoder" if direction == "encode" else "decoder"), f"the sentence names the {s['agent']}, the function is the {'encoder' if direction == 'encode' else 'decoder'}")
    _enc_matches(c, alpha)           # the covering inputs reach every character: the character at position i stands for the value i, as in the reference
    need(s["z"] == 0, "the first character of the alphabet stands for the value 0")


# ====================================================================================================================================
# strobf: every variant says what proc writes
# ====================================================================================================================================

@claim
def writes_exact_string(c: Ctx):
    s = c.s["s"] if "s" in c.s else c.answer[0]                      # a headline states the string; a sentence says "this string" (the truth's)
    n = c.s["len"] if "len" in c.s else len(s)
    need(n == len(s), f"the sentence says {n} characters, the string has {len(s)}")
    out = c.run(f"""int main(void) {{
    unsigned char out[96];
    memset(out, 0xAA, sizeof out);
    proc((char *)out);
    size_t n = strlen((char *)out), spill = 0;
    for (size_t i = n + 1; i < sizeof out; i++) spill += out[i] != 0xAA;
    printf("%zu %zu %s\\n", n, spill, out);
    return 0;
}}""").rstrip("\n")
    need(out == f"{n} 0 {s}", f"proc writes {out!r}, the sentence says {n} characters then a NUL byte, nothing after: the string is {s!r}")


# ====================================================================================================================================
# pktsum
# ====================================================================================================================================

def _pkt_base(c: Ctx) -> bytes:
    """A valid packet of the task's own layout (the Python model of the format), with junk behind the checksum for the length-field variant."""
    t, p = c.t, c.t.params
    rng = random.Random(t.id + "/sweep")
    H, W, kind, magic = p["H"], p["W"], p["kind"], p["magic"]
    lenform = t.variant == "len_field"
    while True:
        m = rng.randint(4, 8)
        head = bytearray(rng.randrange(256) for _ in range(H))
        head[0] = magic
        if lenform:
            head[p["L"]] = m
        body = bytes(head) + bytes(rng.randrange(256) for _ in range(m))
        cov = body[H:H + m] if lenform else body[p["S"]:]
        st = f_pktsum.csum(kind, cov).to_bytes(W, p["endian"] or "little")
        if W == 2 and st[0] == st[1]:
            continue
        return body + st + (bytes(rng.randrange(256) for _ in range(3)) if lenform else b"")


def _pkt_sweep(c: Ctx):
    """For every position of a valid packet and every one of the 256 values of that byte: does the compiled proc accept the packet? ('1'/'0' per value, one row
    per position)."""
    base = _pkt_base(c)
    out = c.run(f"""int main(void) {{
    static const uint8_t base[] = {c_data(base)};
    uint8_t q[sizeof base];
    for (size_t i = 0; i < sizeof base; i++) {{
        for (int v = 0; v < 256; v++) {{
            memcpy(q, base, sizeof base);
            q[i] = (uint8_t)v;
            putchar(proc(q, sizeof q) == 1 ? '1' : '0');
        }}
        putchar('\\n');
    }}
    return 0;
}}""").split()
    need(len(out) == len(base) and all(len(r) == 256 for r in out), "the packet sweep did not finish")
    return base, out


def _pkt_model(c: Ctx, q: bytes, *, S=None, L=None) -> bool:
    """Whether a packet is accepted, by the model of the format (not the generator's code): the magic byte, the range from S to the stored checksum, the checksum
    read in the task's byte order. S: where the checksummed range starts, L: where the length byte is (the sentence's values; the true ones where not stated)."""
    p = c.t.params
    H, W, kind, magic = p["H"], p["W"], p["kind"], p["magic"]
    n = len(q)
    S = p["S"] if S is None else S
    if n < H + W or q[0] != magic:
        return False
    if c.t.variant == "len_field":
        ln = q[p["L"] if L is None else L]
        if H + ln + 1 > n:
            return False
        return f_pktsum.csum(kind, q[S:H + ln]) == q[H + ln]
    return f_pktsum.csum(kind, q[S:n - W]) == int.from_bytes(q[n - W:], p["endian"] or "little")


def _pkt_check(c: Ctx, **model_args):
    """The compiled proc agrees with the model of the sentence (S and/or L as the sentence says) on every single-byte change of a valid packet; cached per model."""
    base, rows = c.once("pktsweep", _pkt_sweep)
    key = ("pktcheck", tuple(sorted(model_args.items())))
    if key not in c.cache:
        bad = []
        if not _pkt_model(c, base, **model_args):
            bad = [("base", 0)]
        else:
            for i, row in enumerate(rows):
                for v in range(256):
                    q = bytearray(base)
                    q[i] = v
                    if (row[v] == "1") != _pkt_model(c, bytes(q), **model_args):
                        bad.append((i, v))
        c.cache[key] = (bad, len(base))
    bad, n = c.cache[key]
    need(not bad, ("the valid packet is not valid under the model of the sentence" if bad and bad[0][0] == "base" else
                   f"proc and the model of the sentence disagree on {len(bad)} of {n * 256} single-byte changes, e.g. byte {bad[0][0]} set to {bad[0][1]:#04x}") if bad else "")


@claim
def pkt_endian_behaviour(c: Ctx):
    p = c.t.params
    need(p["W"] == 2, "the checksum is not 16 bits wide")
    e, first = c.s["e"], c.s.get("first")                             # a headline names the byte order only
    rng = random.Random(c.t.id + "/endian")
    H, kind, magic = p["H"], p["kind"], p["magic"]
    pk = []
    for _ in range(3):
        while True:
            body = bytearray(rng.randrange(256) for _ in range(H + rng.randint(4, 10)))
            body[0] = magic
            cs = f_pktsum.csum(kind, bytes(body[p["S"]:]))
            if (cs >> 8) != (cs & 255):
                break
        for order in ("little", "big"):
            pk.append((order, bytes(body) + cs.to_bytes(2, order)))
    out = ints(c.run("int main(void) {\n" + "".join(f"    {{ static const uint8_t k[] = {c_data(b)}; printf(\"%d\\n\", proc(k, sizeof k)); }}\n" for _o, b in pk) + "    return 0;\n}"))
    acc = {o: [g for (oo, _b), g in zip(pk, out) if oo == o] for o in ("little", "big")}
    accepted = [o for o in acc if set(acc[o]) == {1}]
    need(len(accepted) == 1 and all(g != 1 for g in acc["big" if accepted[0] == "little" else "little"]), f"proc does not accept exactly one byte order: {acc}")
    order = accepted[0]
    need(order == e, f"proc accepts the checksum stored {order}-endian, the sentence says {e}")
    need(first is None or ("low" if order == "little" else "high") == first, f"the byte stored first is the {'low' if order == 'little' else 'high'} one, the sentence says the {first} byte")


@claim
def pkt_range_start(c: Ctx):
    """The range starts at the sentence's offset n: the model with that start agrees with the compiled proc on every change of every byte of a valid packet
    (all 256 values at every position), so the bytes before n are no part of the checksum and the bytes from n on are."""
    _pkt_check(c, S=c.s["n"])


@claim
def pkt_length_byte(c: Ctx):
    need(c.t.variant == "len_field", "no length byte in this variant")
    n = c.s["n"]
    _pkt_check(c, L=n)


# ====================================================================================================================================
# bounds
# ====================================================================================================================================

@claim
def returns_one_exactly_in_range(c: Ctx):
    lo, hi, ret = c.s["lo"], c.s["hi"], c.s["ret"]
    st = region_stats(c, lo, hi)
    const_on(st, "in", ret, f"for the integers from {lo} to {hi}")
    const_on(st, "lt_lo", 0, "below the range")
    const_on(st, "gt_hi", 0, "above the range")
    if "lw" in c.s:
        need(c.s["lo"] == bound_of(c.s["lw"], lo, hi) and c.s["hi"] == bound_of(c.s["hw"], lo, hi), f"the words {c.s['lw']} and {c.s['hw']} do not name {lo} and {hi}")


@claim
def returns_one_for_window(c: Ctx):
    lo, hi, w, ret = c.s["lo"], c.s["hi"], c.s["w"], c.s.get("ret", 1)
    st = region_stats(c, lo, hi)
    const_on(st, "in", ret, f"for the integers from {lo} to {hi}")
    const_on(st, "lt_lo", 0, "below the window")
    const_on(st, "gt_hi", 0, "above the window")
    need(st["in"][2] == w == hi - lo + 1, f"the window from {lo} to {hi} has {st['in'][2]} integers, the sentence says {w}")


def _index_stats(c: Ctx, lo: int, hi: int, neg: int) -> dict:
    out = ints(c.run(f"""static long long cnt[4], first = -1, last = -1, mn = 1LL << 40, mx = -(1LL << 40);
static void visit(unsigned long long i) {{
    int r = proc((unsigned)i);
    int inside = i >= (unsigned long long)({lo}LL) && i <= (unsigned long long)({hi}LL);
    if (r == ({neg})) cnt[inside ? 0 : 1]++;
    else {{
        cnt[inside ? 2 : 3]++;
        if (first < 0) first = (long long)i;
        last = (long long)i;
        if (r < mn) mn = r;
        if (r > mx) mx = r;
    }}
}}
int main(void) {{
    unsigned long long i;
    for (i = 0; i < 262144; i++) visit(i);
    for (i = 262144; i < 4294967296ull; i += 8191) visit(i);
    visit(0xFFFFFFFFull); visit(0x80000000ull); visit(0x7FFFFFFFull);
    printf("%lld %lld %lld %lld %lld %lld %lld %lld\\n", cnt[0], cnt[1], cnt[2], cnt[3], first, last, mn, mx);
    return 0;
}}"""))
    return dict(zip(("in_neg", "out_neg", "in_pos", "out_pos", "first", "last", "mn", "mx"), out))


@claim
def index_limit_range(c: Ctx):
    lo, hi, neg = c.s["lo"], c.s["hi"], c.s["neg"]
    st = _index_stats(c, lo, hi, neg)
    s = c.s
    need(st["mn"] >= 0 and st["mx"] <= 255, f"an accepted index gives {st['mn']} .. {st['mx']}, not a byte value")
    if "o1" in s:
        region_neg = {"outside": "out_pos", "inside": "in_pos"}[s["o1"]]       # a non-negative value in the region the sentence says gives -1
        region_pos = {"inside": "in_neg", "outside": "out_neg"}[s["o2"]]       # a -1 in the region the sentence says gives a non-negative value
        need(st[region_neg] == 0, f"{st[region_neg]} indexes {s['o1']} the range do not give {neg}")
        need(st[region_pos] == 0, f"{st[region_pos]} indexes {s['o2']} the range give {neg}, not a non-negative value")
    if "lw" in s:
        need(lo == (st["first"] if is_low(s["lw"]) else st["last"]) and hi == (st["first"] if is_low(s["hw"]) else st["last"]),
             f"the {s['lw']} accepted index is {st['first'] if is_low(s['lw']) else st['last']}, the {s['hw']} one is {st['first'] if is_low(s['hw']) else st['last']}; "
             f"the sentence says {lo} and {hi}")
    if "o1" not in s and "lw" not in s:
        need(st["in_neg"] == 0 and st["out_pos"] == 0, f"proc does not accept exactly the indexes {lo} .. {hi}: {st}")


@claim
def clamp_interval(c: Ctx):
    s = c.s
    lo, hi = s["lo"], s["hi"]
    st = region_stats(c, lo, hi)
    mn, mx, _n = st["in"]
    need((mn, mx) == (lo, hi), f"proc gives {mn} .. {mx} for the integers from {lo} to {hi}")
    need(st["all"][0] >= lo and st["all"][1] <= hi, f"the results of proc run from {st['all'][0]} to {st['all'][1]}, not between {lo} and {hi}")
    if "d1" in s:
        const_on(st, {"below": "lt_lo", "above": "gt_lo"}[s["d1"]], lo, f"a value {s['d1']} {lo}")
        const_on(st, {"above": "gt_hi", "below": "lt_hi"}[s["d2"]], hi, f"a value {s['d2']} {hi}")
    else:
        const_on(st, "lt_lo", lo, "a value below the interval")
        const_on(st, "gt_hi", hi, "a value above the interval")
    if "lb" in s:
        need(lo == (st["all"][0] if is_low(s["lb"]) else st["all"][1]) and hi == (st["all"][0] if is_low(s["ub"]) else st["all"][1]),
             f"the {s['lb']} bound of the results is {st['all'][0] if is_low(s['lb']) else st['all'][1]}, the {s['ub']} bound is "
             f"{st['all'][0] if is_low(s['ub']) else st['all'][1]}; the sentence says {lo} and {hi}")


@claim
def returns_loop_value(c: Ctx):
    n0, n = c.s["n0"], c.s["n"]
    asked = int(re.search(r"called with\s+(\d+)", c.rec["question"]).group(1))
    need(n0 == asked, f"the sentence is about the argument {n0}, the question asks about {asked}")
    out = ints(c.run(f'int main(void) {{ printf("%d\\n", proc({n0})); return 0; }}'))
    need(out == [n], f"proc({n0}) returns {out}, the sentence says {n}")


@claim
def char_class_count(c: Ctx):
    s = c.s
    out = ints(c.run("""int main(void) {
    for (int c = 0; c < 256; c++) printf("%d ", proc((unsigned char)c));
    printf("%d\\n", (int)(unsigned char)-1);
    return 0;
}"""))
    res, top_c = out[:256], out[256]
    ret, n = s["ret"], s["n"]
    z, top = s.get("z", 0), s.get("top", top_c)
    need(re.search(r"proc\(unsigned char \w+\)", c.body) is not None, "proc does not take a byte")
    if "total" in s:
        need(s["total"] == top_c + 1 == 256, f"a byte has {top_c + 1} values, the sentence says {s['total']}")
    if "z" in s:
        need(z == 0 and top == top_c, f"the byte values run from 0 to {top_c}, the sentence says {z} to {top}")
    need(all(r in (0, ret) for r in res), f"proc returns something other than 0 or {ret}")
    need(sum(1 for x in range(z, top + 1) if res[x] == ret) == n, f"proc returns {ret} for {sum(1 for x in range(z, top + 1) if res[x] == ret)} byte values, the sentence says {n}")


# ====================================================================================================================================
# antidebug: the fakes of the task's own harness, other mains
# ====================================================================================================================================

def _pre(c: Ctx) -> str:
    return c.t.check[2]


@claim
def ptrace_request(c: Ctx):
    s = c.s
    raw = c.key.endswith("raw")
    call, req = s["call"], s.get("req", "PTRACE_TRACEME")
    if "tech" in s:                                                  # a headline names the technique of the question's list
        need(s["tech"] == "ptrace", f"the headline names the technique {s['tech']}, proc asks to be traced by its parent: ptrace")
    if raw:
        proof = f"last_no == SYS_{call} && last_req == {req}"
    else:
        need(re.search(rf"\b{call}\(", c.t.body), f"the C does not call {call}")
        proof = f"last_req == {req}"
    out = ints(c.run(f"""int main(void) {{
    fake_rv = 0;
    int ok = proc();
    int asked = {proof};
    fake_rv = -1;
    int fail = proc();
    printf("%d %d %d\\n", ok, fail, asked);
    return 0;
}}""", _pre(c)))
    ok, fail, asked = out
    need(asked == 1, f"the request that proc makes is not {call}({req})")
    got = {"fails": fail, "refused": fail, "succeeds": ok, "granted": ok}[s["outcome"]]
    need(got == s["rv"], f"proc returns {got} when the request {s['outcome']}, the sentence says {s['rv']} (a request that succeeds gives {ok}, one that fails {fail})")
    need(ok == 0, f"proc returns {ok} for a request that succeeds")


@claim
def tracerpid_status(c: Ctx):
    s = c.s
    field, rv = s["field"], s["rv"]
    if "tech" in s:
        need(s["tech"].lower() == "tracerpid", f"the headline names the technique {s['tech']}, proc reads the TracerPid line")
    other = "" if field == "TracerPid" else "TracerPid:\\t0\\n"
    pre = """static char fake_buf[256], last_path[160];
static FILE *my_fopen(const char *path, const char *mode) {
    (void)mode;
    snprintf(last_path, sizeof last_path, "%s", path);
    return fmemopen(fake_buf, strlen(fake_buf), "r");
}
#define fopen my_fopen
"""
    path_ok = f'strcmp(last_path, "{s["file"]}") == 0' if "file" in s else "1"
    out = ints(c.run(f"""int main(void) {{
    snprintf(fake_buf, sizeof fake_buf, "Name:\\tx\\n{field}:\\t4242\\n{other}Uid:\\t1\\n");
    int named = proc();
    int path = {path_ok};
    snprintf(fake_buf, sizeof fake_buf, "Name:\\tx\\nTracerPid:\\t0\\nUid:\\t1\\n");
    int none = proc();
    snprintf(fake_buf, sizeof fake_buf, "Name:\\tx\\nTracerPid:\\t1\\nUid:\\t1\\n");
    int one = proc();
    printf("%d %d %d %d\\n", named, path, none, one);
    return 0;
}}""", pre))
    named, path, none, one = out
    need(path == 1, f"proc does not open {s.get('file')}")
    need(none == 0, f"proc returns {none} when TracerPid is 0")
    need(named == rv, f"proc returns {named} when the {field} line shows a tracer (TracerPid is {'4242' if field == 'TracerPid' else '0'}), the sentence says {rv}")


@claim
def env_variable_read(c: Ctx):
    name = c.s["NAME"]
    if "tech" in c.s:
        need(c.s["tech"] == "env", f"the headline names the technique {c.s['tech']}, proc reads an environment variable: env")
    pre = """static const char *last_name, *fake_val;
static char *fake_getenv(const char *n) { last_name = n; return (char *)fake_val; }
#define getenv fake_getenv
"""
    out = c.run(f"""int main(void) {{
    fake_val = NULL;
    int a = proc();
    int asked = last_name && strcmp(last_name, "{name}") == 0;
    fake_val = "1";
    int b = proc();
    printf("%d %d %d\\n", a, b, asked);
    return 0;
}}""", pre)
    a, b, asked = ints(out)
    need(asked == 1, f"proc asks for another environment variable than {name}")
    need(a != b, "the result does not depend on the variable")


SCALE_NS = {"nanoseconds": 1, "microseconds": 1000, "milliseconds": 10 ** 6, "seconds": 10 ** 9}
UNIT_NAME = {"ns": "nanoseconds", "us": "microseconds", "ms": "milliseconds", "s": "seconds"}      # the abbreviations of a headline


@claim
def timing_threshold(c: Ctx):
    """The clock is faked in NANOSECONDS whatever API the task reads (clock_gettime or gettimeofday), and the elapsed time is the sentence's number of the
    sentence's unit: a wrong unit moves the threshold by a factor of 1000."""
    s = c.s
    thr, unit = s["thr"], UNIT_NAME.get(s["unit"], s["unit"])
    if "tech" in s:
        need(s["tech"] == "timing", f"the headline names the technique {s['tech']}, proc compares two clock readings: timing")
    ns = SCALE_NS[unit]
    if "clock_gettime(" in c.body:
        pre = """static long long fake_ns;
static int fake_calls;
static int fake_clock_gettime(clockid_t id, struct timespec *t) {
    long long v = (fake_calls++ & 1) ? fake_ns : 0;
    (void)id;
    t->tv_sec = v / 1000000000LL;
    t->tv_nsec = (long)(v % 1000000000LL);
    return 0;
}
#define clock_gettime fake_clock_gettime
"""
    else:
        need("gettimeofday(" in c.body, "the C reads no clock")
        pre = """static long long fake_ns;
static int fake_calls;
static int fake_gettimeofday(struct timeval *t, void *tz) {
    long long v = (fake_calls++ & 1) ? fake_ns : 0;
    (void)tz;
    t->tv_sec = (time_t)(v / 1000000000LL);
    t->tv_usec = (suseconds_t)((v % 1000000000LL) / 1000);
    return 0;
}
#define gettimeofday fake_gettimeofday
"""
    ks = sorted({0, max(thr - 1, 0), thr, thr + 1, thr * 2 + 5})
    out = c.run("int main(void) {\n" + "".join(f'    fake_ns = {k * ns}LL; fake_calls = 0; {{ int r = proc(); printf("%d %d\\n", r, fake_calls); }}\n' for k in ks) + "    return 0;\n}", pre)
    rows = [ints(line) for line in out.strip().split("\n")]
    above = s.get("cmp", "above") == "above"
    need([r[0] != 0 for r in rows] == [(k > thr) if above else (k < thr) for k in ks],
         f"proc reports a debugger for the elapsed times {ks} {unit} as {[r[0] for r in rows]}; the sentence says it does so {s.get('cmp', 'above')} {thr} {unit}")
    need(all(r[1] == 2 for r in rows), f"proc reads the clock {[r[1] for r in rows]} times, not two")


# ====================================================================================================================================
# multi
# ====================================================================================================================================

@claim
def returns_for_argument(c: Ctx):
    n0, n = c.s["n0"], c.s["n"]
    asked = int(re.search(r"called with\s+(\d+)", c.rec["question"]).group(1))
    need(n0 == asked, f"the sentence is about the argument {n0}, the question asks about {asked}")
    out = ints(c.run(f'int main(void) {{ printf("%u\\n", (unsigned)proc({n0}u)); return 0; }}'))
    need(out == [n], f"proc({n0}) returns {out}, the sentence says {n}")


@claim
def xors_key_byte(c: Ctx):
    seed, k = c.s["seed"], c.s["k"]
    asked = int(re.search(r"seed\s*=\s*(\d+)", c.rec["question"]).group(1))
    need(seed == asked, f"the sentence is about the seed {seed}, the question asks about {asked}")
    out = c.run(f'int main(void) {{ uint8_t z[300] = {{0}}, b[300]; for (int i = 0; i < 300; i++) b[i] = (uint8_t)i; proc(z, 300, {seed}u); proc(b, 300, {seed}u); '
                'for (int i = 0; i < 300; i++) printf("%d %d\\n", z[i], b[i]); return 0; }').split("\n")
    rows = [ints(line) for line in out if line.strip()]
    need(rows == [[k, (i & 255) ^ k] for i in range(300)], f"proc does not XOR {k:#x} into every byte of the buffer for the seed {seed} (every byte value, 300 positions)")


def _neighbours(c: Ctx, secret: str, ret: int) -> list:
    """proc(const char *) on the secret and on every input one edit away from it (each position x each byte 1..255: replaced, inserted; each position deleted),
    on the empty string and on the secret plus a byte: [result for the secret, number of other inputs that give `ret`, number of inputs tried]."""
    n = len(secret)
    return ints(c.run(f"""int main(void) {{
    static const char S[] = {c_str(secret)};
    size_t L = sizeof S - 1;
    char b[160];
    unsigned long long bad = 0, tried = 0;
    printf("%d ", proc(S));
    for (size_t pos = 0; pos < L; pos++) for (int v = 1; v < 256; v++) {{
        if ((char)v == S[pos]) continue;
        memcpy(b, S, L + 1); b[pos] = (char)v;
        tried++; if (proc(b) == ({ret})) bad++;
    }}
    for (size_t pos = 0; pos <= L; pos++) for (int v = 1; v < 256; v++) {{
        memcpy(b, S, pos); b[pos] = (char)v; memcpy(b + pos + 1, S + pos, L - pos + 1);
        tried++; if (proc(b) == ({ret})) bad++;
    }}
    for (size_t pos = 0; pos < L; pos++) {{
        memcpy(b, S, pos); memcpy(b + pos, S + pos + 1, L - pos);
        tried++; if (proc(b) == ({ret})) bad++;
    }}
    tried++; if (proc("") == ({ret})) bad++;
    printf("%llu %llu\\n", bad, tried);
    return 0;
}}"""))


@claim
def accepts_only_the_secret(c: Ctx):
    s = c.s["s"] if "s" in c.s else c.answer[0]
    ret, n = c.s.get("ret", 1), c.s["len"] if "len" in c.s else len(s)
    res, bad, tried = _neighbours(c, s, ret)
    need(n == len(s), f"the sentence says {n} characters, the secret has {len(s)}")
    need(res == ret, f"proc returns {res} for the secret, the sentence says {ret}")
    need(bad == 0, f"proc returns {ret} for {bad} of the {tried} inputs one edit away from the secret")


@claim
def limit_for_mode(c: Ctx):
    mode, n, ret = c.s["mode"], c.s["n"], c.s.get("ret", 1)
    asked = int(re.search(r"mode\s*=\s*(\d+)", c.rec["question"]).group(1))
    need(mode == asked, f"the sentence is about the mode {mode}, the question asks about {asked}")
    out = ints(c.run('int main(void) {\n    char s[100];\n    int fills[3] = {97, 255, 1};\n    for (int f = 0; f < 3; f++) for (int len = 0; len < 90; len++) '
                     '{ memset(s, fills[f], (size_t)len); s[len] = 0; printf("%%d ", proc(s, %du)); }\n    return 0;\n}' % mode))
    want = [ret if len_ <= n else None for _f in range(3) for len_ in range(90)]
    need(all((r == ret) if w is not None else (r != ret) for r, w in zip(out, want)),
         f"proc accepts the lengths {[i for i, r in enumerate(out[:90]) if r == ret][:3]}..{max([i for i, r in enumerate(out[:90]) if r == ret] or [-1])}, the sentence says up to {n}, "
         f"returning {ret}; (three fill bytes, every length 0..89)")


# ====================================================================================================================================
# legacy
# ====================================================================================================================================

@claim
def xor_key_repeats(c: Ctx):
    s = c.s["s"] if "s" in c.s else c.answer[0]
    n = c.s["len"] if "len" in c.s else len(s)
    need(n == len(s), f"the sentence says {n} bytes, the key has {len(s)}")
    N = 256 + 2 * len(s)
    out = c.run(f'int main(void) {{ uint8_t z[{N}] = {{0}}, b[{N}]; for (int i = 0; i < {N}; i++) b[i] = (uint8_t)i; proc(z, {N}); proc(b, {N}); '
                f'for (int i = 0; i < {N}; i++) printf("%02x%02x", z[i], b[i]); return 0; }}')
    want = "".join(f"{ord(s[i % len(s)]):02x}{(i & 255) ^ ord(s[i % len(s)]):02x}" for i in range(N))
    need(out == want, "the buffer is not XORed with the key repeated along it (every byte value at every position of a long buffer)")
    per = next(p for p in range(1, len(s) + 1) if len(s) % p == 0 and all(s[i] == s[i % p] for i in range(len(s))))
    need(per == len(s), "the key repeats with a shorter period than its length")


@claim
def password_equal_only(c: Ctx):
    s = c.s["s"] if "s" in c.s else c.answer[0]
    ret, n = c.s.get("ret", 1), c.s.get("len", len(s))
    res, bad, tried = _neighbours(c, s, ret)
    need(n == len(s), f"the sentence says {n} characters, the password has {len(s)}")
    need(res == ret, f"proc returns {res} for this string, the sentence says {ret}")
    need(bad == 0, f"proc returns {ret} for {bad} of the {tried} inputs one edit away from the password")


def key_lines(k: str) -> list:
    """Every shape of one line (round 7, m3: the domain `lines`): 5 prefixes (none, a space, a tab, `#`, a letter) x 5 spellings of the key (itself, a longer one, a shorter
    one, in upper case, with a letter before it) x 4 separators (`=`, `:`, a space, none) x 8 values (empty, one character, a space, several characters, an equals sign,
    an equals sign inside, a carriage return): 800 lines."""
    spellings = [k, k + "x", k[:-1], k.upper() if k.upper() != k else k + "X", "x" + k]
    return [p + kv + sep + v for p in ("", " ", "\t", "#", "x") for kv in spellings for sep in ("=", ":", " ", "") for v in ("", "1", "v", " ", "  x", "=", "a=b", "\r")]


@claim
def keycount_lines(c: Ctx):
    k = c.s["s"] if "s" in c.s else c.answer[0]
    text = f"{k}=1\\n#{k}=2\\n{k}=\\n{k}x=3\\n{k}=4\\n x{k}=5\\n"
    out = ints(c.run(f'int main(void) {{ printf("%d %d %d %d\\n", proc("{text}"), proc("{k}="), proc("{k}=v"), proc("#{k}=v")); return 0; }}'))
    need(out == [2, 0, 1, 0], f"proc counts {out} lines: key=1 and key=4 count, key= with nothing after, a comment, another key and an indented line do not")
    # the whole domain of line shapes, one line at a time and all together: a line is counted when it begins with the key and an equals sign and has something after it
    lines = key_lines(k)
    want = [int(ln.startswith(k + "=") and len(ln) > len(k) + 1) for ln in lines]
    out = ints(c.run("int main(void) {\n" + "".join(f'    printf("%d\\n", proc({c_bytes_str(ln.encode())}));\n' for ln in lines)
                     + f'    printf("%d\\n", proc({c_bytes_str(chr(10).join(lines).encode())}));\n    return 0;\n}}'))
    bad = [ln for ln, w, g in zip(lines, want, out) if w != g]
    need(not bad and len(out) == len(lines) + 1, f"proc counts {len(bad)} of the {len(lines)} line shapes wrongly, e.g. {bad[:2]!r}: a line is counted when it begins with the key and an equals sign and has something after it")
    need(out[-1] == sum(want), f"proc counts {out[-1]} of the lines of the whole text, {sum(want)} begin with the key and an equals sign and have something after it")
    if "len" in c.s:
        need(c.s["len"] == len(k), f"the key has {len(k)} characters, the sentence says {c.s['len']}")


@claim
def alphabet_position(c: Ctx):
    s = c.s["s"] if "s" in c.s else c.answer[0]
    out = ints(c.run(f'int main(void) {{ const char *a = {c_str(s)}; for (int i = 0; a[i]; i++) printf("%d ", proc(a[i])); '
                     'for (int ch = 0x20; ch < 0x7f; ch++) printf("%d ", proc((char)ch)); return 0; }'))
    n = c.s["len"] if "len" in c.s else len(s)
    need(n == len(s) == 32, f"the alphabet has {len(s)} characters, the sentence says {n}")
    need(out[:len(s)] == list(range(len(s))), "proc does not return the position of a character of the string")
    if "miss" in c.s:
        printable = out[len(s):]
        want = [s.index(chr(ch)) if chr(ch) in s else c.s["miss"] for ch in range(0x20, 0x7f)]
        need(printable == want, f"proc does not return {c.s['miss']} for every printable character that is not in the string (all 95 printable characters probed)")


@claim
def magic_header(c: Ctx):
    s = c.s
    m, v, e = s["m"], small(s["v"], 0, 1024, "the highest version"), s.get("e", "little")
    need(0 <= m < 2 ** 32, f"the magic {m:#x} is not a 32-bit word")
    mb = m.to_bytes(4, e)
    def hdr(magic_bytes, ver, n=8):
        return (magic_bytes + struct.pack("<I", ver))[:n]
    cases = [("ok", hdr(mb, ver)) for ver in range(v + 1)] + [("hi", hdr(mb, ver)) for ver in (v + 1, v + 2, 255, 256, 0x7fffffff, 0xffffffff)]
    cases += [("short", hdr(mb, 0, 7)), ("short", hdr(mb, 0, 4)), ("short", b"")]
    for byte in range(4):                                              # every single-bit change of the magic is rejected
        for bit in range(8):
            q = bytearray(mb)
            q[byte] ^= 1 << bit
            cases.append(("bad", hdr(bytes(q), 0)))
    other = m.to_bytes(4, "big" if e == "little" else "little")
    if other != mb:
        cases.append(("other", hdr(other, 0)))
    main = "int main(void) {\n" + "".join(f"    {{ static const uint8_t h[] = {c_data(b) if b else '{0}'}; printf(\"%d\\n\", proc(h, {len(b)})); }}\n" for _t, b in cases) + "    return 0;\n}"
    got = ints(c.run(main))
    res = {}
    for (tag, _b), g in zip(cases, got):
        res.setdefault(tag, []).append(g)
    need(res["ok"] == list(range(v + 1)), f"headers with the {e}-endian magic {m:#x} and the version 0 .. {v} give {res['ok']}")
    need(all(g < 0 for g in res["hi"]), f"a version above {v} is not rejected: {res['hi']}")
    need(all(g < 0 for g in res["short"]), f"a header shorter than eight bytes is not rejected: {res['short']}")
    need(all(g < 0 for g in res["bad"]), "a magic that differs in one bit is accepted")
    need(all(g < 0 for g in res.get("other", [-1])), f"the magic written {'big' if e == 'little' else 'little'}-endian is accepted: the bytes of the header are the {e}-endian bytes of the value")


@claim
def lcg_reference(c: Ctx):
    a, r = c.s["a"], small(c.s["r"], 1, 64, "the rounds")
    mm = re.search(r"seed = \(seed \* \d+u \+ (\d+)u\) & 0x7fffffffu;", c.t.body)
    need(mm, "the loop of the C is not a linear congruential step")
    add = int(mm.group(1))
    mask = c.s.get("mask", 0x7fffffff)
    seeds = [1, 12345, 0xdeadbeef]
    out = ints(c.run("int main(void) {\n" + "".join(f'    printf("%u\\n", proc({s}u));\n' for s in seeds) + "    return 0;\n}"))

    def ref(s):
        for _ in range(r):
            s = ((s * a + add) & M32) & mask
        return s
    need(out == [ref(s) for s in seeds], f"proc is not {r} rounds of seed * {a} + c masked with {mask:#x}")


def h_fnv(b: bytes) -> int:
    h = 2166136261
    for x in b:
        h = ((h ^ x) * 16777619) & M32
    return h


def h_djb2(b: bytes) -> int:
    h = 5381
    for x in b:
        h = (h * 33 + x) & M32
    return h


def h_sdbm(b: bytes) -> int:
    h = 0
    for x in b:
        h = (x + (h << 6) + (h << 16) - h) & M32
    return h


def h_murmur_style(b: bytes) -> int:
    """The legacy per-byte loop: MurmurHash2's multiplier and seed, one byte at a time. NOT MurmurHash2."""
    h = 0x9747b28c
    for x in b:
        h ^= x
        h = (h * 0x5bd1e995) & M32
        h ^= h >> 15
    return h


def murmur2(data: bytes, seed: int) -> int:
    """Austin Appleby's MurmurHash2 (32-bit), ported from his C and compared with it in tests/test_rea_claims.py."""
    m, r = 0x5bd1e995, 24
    h = (seed ^ len(data)) & M32
    i = 0
    while len(data) - i >= 4:
        k = int.from_bytes(data[i:i + 4], "little")
        k = (k * m) & M32
        k ^= k >> r
        k = (k * m) & M32
        h = (h * m) & M32
        h ^= k
        i += 4
    rest = len(data) - i
    if rest == 3:
        h ^= data[i + 2] << 16
    if rest >= 2:
        h ^= data[i + 1] << 8
    if rest >= 1:
        h ^= data[i]
        h = (h * m) & M32
    h ^= h >> 13
    h = (h * m) & M32
    h ^= h >> 15
    return h


HASHES = {"FNV-1a": h_fnv, "djb2": h_djb2, "sdbm": h_sdbm, "Murmur-style": h_murmur_style,
          "MurmurHash2": lambda w: murmur2(w, 0x9747b28c), "Murmur": lambda w: murmur2(w, 0x9747b28c)}
HASH_INIT = {"FNV-1a": 2166136261, "djb2": 5381, "sdbm": 0, "Murmur-style": 0x9747b28c}
STEM_NAME = {"fnv": "FNV-1a", "djb2": "djb2", "sdbm": "sdbm", "murmur": "Murmur-style"}      # the name of the hash that the question's answer stem stands for
HASH_WORDS = [b"", b"a", b"abc", b"hello world", b"The quick brown fox", b"admin_cache42", b"/tmp/x/y"]


def hash_inputs(c: Ctx) -> list:
    rng = random.Random(c.t.id + "/hash")
    return HASH_WORDS + [bytes([x]) for x in range(1, 256)] + [bytes(rng.randrange(1, 256) for _ in range(rng.randint(2, 40))) for _ in range(120)]


def _hash_probe(c: Ctx):
    words = hash_inputs(c)
    main = "int main(void) {\n" + "".join(f'    printf("%u\\n", (unsigned)proc({c_bytes_str(w)}));\n' for w in words) + "    return 0;\n}"
    return words, ints(c.run(main))


def _hash_name(c: Ctx) -> str:
    return c.s.get("name", STEM_NAME[c.answer[0][0]])


@claim
def hash_final_xor(c: Ctx):
    name, k = _hash_name(c), c.s["c"]
    words, got = c.once("hash", _hash_probe)
    need(name in HASHES, f"{name!r} is not an algorithm this test has a reference for")
    ref = HASHES[name]
    need(got == [ref(w) ^ k for w in words], f"proc is not the {name} hash XORed with {k:#x} (7 words, every one-byte string, 120 random strings)")


@claim
def hash_start_value(c: Ctx):
    k = c.s["c"]
    words, got = c.once("hash", _hash_probe)
    need(got[0] ^ k == c.s["init"], f"the hash of the empty string is {got[0] ^ k:#x}, the sentence says the hash starts from {c.s['init']:#x}")


@claim
def hash_name_is_the_algorithm(c: Ctx):
    name, k = c.s["name"], c.s["c"]
    words, got = c.once("hash", _hash_probe)
    need(name in HASHES, f"{name!r} is not an algorithm this test has a reference for")
    if name == "Murmur-style":
        need("0x5bd1e995u" in c.body and "0x9747b28cu" in c.body, "the loop does not use MurmurHash2's multiplier and seed")
        need(got != [murmur2(w, 0x9747b28c) ^ k for w in words], "this IS MurmurHash2: the name would be Murmur, not Murmur-style")
    need(got == [HASHES[name](w) ^ k for w in words], f"proc is not {name}")


@claim
def skip_one_value(c: Ctx):
    b = c.s["b"]
    rng = random.Random(c.t.id + "/skip")
    multi = [bytes(rng.choice([b, rng.randrange(256)]) for _ in range(rng.randint(0, 30))) for _ in range(40)]
    out = ints(c.run('int main(void) {\n    for (int v = 0; v < 256; v++) { uint8_t x = (uint8_t)v; printf("%u ", (unsigned)proc(&x, 1)); }\n    printf("\\n");\n'
                     + "".join(f'    {{ static const uint8_t t[] = {c_data(m) if m else "{0}"}; printf("%u ", (unsigned)proc(t, {len(m)})); }}\n' for m in multi) + "    return 0;\n}"))
    want = [(0 if v == b else v) + 1 for v in range(256)] + [sum(x for x in m if x != b) + len(m) for m in multi]
    need(out == want, f"proc does not skip exactly the byte {b:#04x} and add the length (all 256 one-byte inputs, 40 longer ones)")


def _rot(x, r, left):
    r %= 32
    if not left:
        r = (32 - r) % 32
    return ((x << r) | (x >> (32 - r))) & M32


@claim
def rotate_reference(c: Ctx):
    k, r, left = c.s["k"], c.s["r"], c.s["dir"] == "left"
    xs = [0, 1, 0xdeadbeef, 0x80000000, 0x12345678]
    out = ints(c.run("int main(void) {\n" + "".join(f'    printf("%u\\n", (unsigned)proc({x}u));\n' for x in xs) + "    return 0;\n}"))
    need(out == [_rot(x ^ k, r, left) for x in xs], f"proc is not XOR with {k:#x} then a rotation to the {c.s['dir']} by {r} bits")


@claim
def threshold_values(c: Ctx):
    s = c.s
    lo, hi = s["lo"], s["hi"]
    st = region_stats(c, lo, hi)
    mn, mx, cnt = st["in"]
    need(mn == mx and cnt == hi - lo + 1, f"proc does not return one value for the integers from {lo} to {hi}: {mn} .. {mx}")
    inside = mn
    for region in ("lt_lo", "gt_hi"):
        rmn, rmx, _n = st[region]
        need(rmn == rmx != inside, f"proc returns {rmn} .. {rmx} {'below' if region == 'lt_lo' else 'above'} the range and {inside} inside it")
    if "inside" in s:
        need(inside == s["inside"], f"proc returns {inside} for the integers from {lo} to {hi}, the sentence says {s['inside']}")
    if "c1" in s:                                                      # a smaller v / a larger v
        const_on(st, {"smaller": "lt_lo", "larger": "gt_hi"}[s["c1"]], s["below"], f"for a {s['c1']} v")
        const_on(st, {"larger": "gt_hi", "smaller": "lt_lo"}[s["c2"]], s["above"], f"for a {s['c2']} v")
    if "d1" in s:                                                      # Below lo / Above lo, above hi / below hi
        const_on(st, {"below": "lt_lo", "above": "gt_lo"}[s["d1"].lower()], s["below"], f"{s['d1']} {lo}")
        const_on(st, {"above": "gt_hi", "below": "lt_hi"}[s["d2"]], s["above"], f"{s['d2']} {hi}")
    if "lb" in s:
        need(lo == bound_of(s["lb"], lo, hi) and hi == bound_of(s["ub"], lo, hi), f"the {s['lb']} bound is {bound_of(s['lb'], lo, hi)} and the {s['ub']} bound is "
             f"{bound_of(s['ub'], lo, hi)}, the sentence says {lo} and {hi}")
