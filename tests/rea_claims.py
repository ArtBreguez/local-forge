"""The checkers of the claims that the truth templates make (round 5). tests/test_rea_claims.py holds the table that says which sentence makes which claim.

A claim is a function of a `Ctx` (the generated task, the facts its sentence states, the key of the template) that returns None when it holds and
raises `ClaimFailed(why)` when it does not. Where a claim is about behaviour the checker COMPILES THE TASK'S C with gcc and RUNS it (one program per task
and probe, cached in the Ctx), and compares with the fact: a Python reference implementation (CRC, TEA, XTEA, RC4, ChaCha, Salsa, base64, base32, base58, FNV-1a,
djb2, sdbm, the legacy "Murmur-style" loop), a scan of the whole domain, a parsed model of the machine, `offsetof`/`sizeof` of the compiler, a fake libc call
(`ptrace`, `getenv`, `fopen`, a clock) that records what proc asks for. Where a claim is about a constant of the template it parses the C source. No claim reads a
sentence: the sentence is the table's business, the checker only knows the fact.
"""
from __future__ import annotations

import itertools
import os
import pathlib
import random
import re
import struct
import subprocess
import sys
import tempfile

from .helpers import ROOT

sys.path.insert(0, str(ROOT / "tools" / "rea-bench"))

from rea_bench import core, f_cipher, f_crc, f_encoding, f_pktsum, targets  # noqa: E402

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


class Ctx:
    def __init__(self, task, facts: dict, key: str):
        self.t, self.f, self.key = task, facts, key
        self.cache: dict = {}

    def once(self, name, fn):
        if name not in self.cache:
            self.cache[name] = fn(self)
        return self.cache[name]

    @property
    def body(self) -> str:
        """The C of the task without its decoy statements."""
        return core.strip_decoys(self.t.body)

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
        r = subprocess.run([exe], capture_output=True, text=True, timeout=30)
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
    need(size == c.f["n"], f"sizeof(record) is {size}, the sentence says {c.f['n']}")
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
    s.{g} = 1;
    printf("%llu\\n", (unsigned long long)proc(&s));
    s.{g} = (__typeof__(s.{g}))-1;
    printf("%llu\\n", (unsigned long long)proc(&s));
    return 0;
}}"""))
    need(out[0] == c.f["off"], f"the returned field is at offset {out[0]:#x}, the sentence says {c.f['off']:#x}")
    need(out[1:] == [0x2a + int(k)] * 2, f"proc returns {out[1:]} for a non-zero flag, expected the field + {k}")


def _init_probe(c: Ctx):
    b = c.t.body
    st = re.search(r"void proc\(struct (\w+) \*p\)", b).group(1)
    stores = re.findall(r"p->(\w+) = (0x[0-9a-f]+)(?:u|ull)?;", b)
    hit = [f for f, lit in stores if int(lit, 16) == c.f["magic"]]
    need(len(hit) == 1, f"{len(hit)} stores write the magic constant")
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
    for off, val, _other in c.once("init", _init_probe):
        need(off == c.f["off"], f"the field that receives the constant is at {off:#x}, the sentence says {c.f['off']:#x}")
        need(val == c.f["magic"], f"proc stores {val:#x}, not {c.f['magic']:#x}")


@claim
def init_other_fields(c: Ctx):
    need(max(r[2] for r in c.once("init", _init_probe)) > 0, "proc writes no other field")


def _alloc_probe(c: Ctx):
    st = re.search(r"struct (\w+) \*proc\(uint32_t id\)", c.t.body).group(1)
    pre = """static size_t last_elem, last_count;
static void *my_malloc(size_t n) { last_elem = n; last_count = 1; return __builtin_calloc(1, n); }
static void *my_calloc(size_t c, size_t n) { last_elem = n; last_count = c; return __builtin_calloc(c, n); }
#define malloc my_malloc
#define calloc my_calloc
"""
    return ints(c.run(f'int main(void) {{ proc(7); printf("%zu %zu %zu\\n", sizeof(struct {st}), last_elem, last_count); return 0; }}', pre))


@claim
def alloc_structure_size(c: Ctx):
    size, elem, count = c.once("alloc", _alloc_probe)
    need(size == c.f["n"], f"sizeof(structure) is {size}, the sentence says {c.f['n']}")
    need(elem == size and count >= 1, f"the allocator is asked for elements of {elem} bytes (x{count}), the structure is {size}")


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
    need(off == c.f["off"], f"the nested field is at {off:#x} of the outer structure, the sentence says {c.f['off']:#x}")
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
    need(off == c.f["off"], f"the pointer to the next node is at {off:#x}, the sentence says {c.f['off']:#x}")
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


def _bfs(delta):
    dist, front = {0: 0}, [0]
    while front:
        nxt = []
        for s in front:
            for t in delta[s]:
                if t not in dist:
                    dist[t] = dist[s] + 1
                    nxt.append(t)
        front = nxt
    return dist


def _fsm_probe(c: Ctx):
    alpha, delta = parse_machine(c.body)
    rng = random.Random(c.t.id)
    strings = ["".join(rng.choice(alpha) for _ in range(rng.randint(0, 9))) for _ in range(40)]
    if "s" in c.f:
        strings.append(c.f["s"])
    out = c.run("int main(void) {\n" + "".join(f'    printf("%d\\n", proc({c_str(s)}));\n' for s in strings) + "    return 0;\n}")
    got = ints(out)
    need(got == [_sim(alpha, delta, s) for s in strings], "the machine parsed from the C is not what the compiled proc computes")
    return alpha, delta, strings, got


@claim
def fsm_input_final(c: Ctx):
    alpha, delta, strings, got = c.once("fsm", _fsm_probe)
    need(got[-1] == c.f["final"], f"proc returns {got[-1]} for the input, the sentence says state {c.f['final']}")
    need(c.f["d"] == len(c.f["s"]) and set(c.f["s"]) <= set(alpha), "the input is not d letters of the machine's alphabet")


@claim
def fsm_input_shortest(c: Ctx):
    alpha, delta, _s, _g = c.once("fsm", _fsm_probe)
    need(_bfs(delta).get(c.f["final"]) == c.f["d"], f"the shortest way to state {c.f['final']} is {_bfs(delta).get(c.f['final'])} letters, not {c.f['d']}")


@claim
def fsm_states_count(c: Ctx):
    alpha, delta, _s, _g = c.once("fsm", _fsm_probe)
    need(len(delta) == c.f["n"] and c.f["last"] == c.f["n"] - 1, f"the machine has {len(delta)} states, the sentence says {c.f['n']}")
    need(sorted(_bfs(delta)) == list(range(c.f["n"])), "the states reachable from 0 are not numbered 0 .. n-1")


@claim
def fsm_states_reachable(c: Ctx):
    alpha, delta, _s, _g = c.once("fsm", _fsm_probe)
    need(len(_bfs(delta)) == c.f["n"], f"{len(_bfs(delta))} of the {c.f['n']} states are reachable from state 0")


def _opcode_probe(c: Ctx):
    pairs = [(0x1234abcd, 7), (0xdeadbeef, 0x12345679), (0x0f0f0f0f, 3)]
    out = c.run("int main(void) {\n    for (uint32_t op = 0; op < 64; op++) {\n" + "".join(
        f'        printf("%u ", proc(op, 0x{a:x}u, {b}u));\n' for a, b in pairs) + '        printf("\\n");\n    }\n    return 0;\n}')
    rows = [ints(line) for line in out.strip().split("\n")]
    need(len(rows) == 64, "the scan did not run")
    return pairs, rows


def _eval_expr(expr: str, a: int, b: int) -> int:
    return eval(expr, {"__builtins__": {}}, {"a": a, "b": b}) & M32


@claim
def opcode_returns_expr(c: Ctx):
    pairs, rows = c.once("opcodes", _opcode_probe)
    want = [_eval_expr(c.f["expr"], a, b) for a, b in pairs]
    need(rows[c.f["n"]] == want, f"proc({c.f['n']}, a, b) returns {rows[c.f['n']]}, not {c.f['expr']}: {want}")


@claim
def opcode_unique(c: Ctx):
    pairs, rows = c.once("opcodes", _opcode_probe)
    want = [_eval_expr(c.f["expr"], a, b) for a, b in pairs]
    ops = [op for op, r in enumerate(rows) if r == want]
    need(ops == [c.f["n"]], f"the opcodes that return {c.f['expr']} are {ops}")


@claim
def opcode_count_handled(c: Ctx):
    pairs, rows = c.once("opcodes", _opcode_probe)
    handled = [op for op, r in enumerate(rows) if any(r)]
    need(len(handled) == c.f["n"], f"{len(handled)} opcodes are handled, the sentence says {c.f['n']}")


@claim
def opcode_default(c: Ctx):
    pairs, rows = c.once("opcodes", _opcode_probe)
    need(c.f["dflt"] == 0 and all(r == [c.f["dflt"]] * len(pairs) for r in rows if not any(r)), "an unhandled opcode returns something else")


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
    vecs, got = c.once("crc", _crc_probe)
    poly = f_crc.reflect(poly_lit, width) if refl else poly_lit          # the constant of a reflected CRC is the reflected polynomial
    want = [f_crc.crc_ref(v, width, poly, init, refl, xorout) for v in vecs]
    need(got == want, f"proc computes {[hex(x) for x in got]}, a {width}-bit CRC with these parameters gives {[hex(x) for x in want]}")


@claim
def crc_poly_form(c: Ctx):
    p = c.t.params
    need(c.f["refl"] == (c.f["dir"] == "reflected"), "the form word and the flag differ")
    _crc_matches(c, width=c.f["w"], poly_lit=c.f["lit"], refl=c.f["refl"], init=p["init"], xorout=p["xorout"])


@claim
def crc_init_value(c: Ctx):
    p = c.t.params
    _crc_matches(c, width=c.f["w"], poly_lit=p["lit"], refl=p["refl"], init=c.f["x"], xorout=p["xorout"])


@claim
def crc_xorout_value(c: Ctx):
    p = c.t.params
    _crc_matches(c, width=c.f["w"], poly_lit=p["lit"], refl=p["refl"], init=p["init"], xorout=c.f["x"])


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
    f, b = c.f, c.body
    key = [int(x, 16) for x in re.search(r"k\[4\] = \{(.*?)\}", b).group(1).replace("u", "").split(",")]
    need(len(key) * 32 == f["key_bits"], f"the key has {len(key) * 32} bits, the sentence says {f['key_bits']}")
    need((f["verb"], f["done"]) == (f["dir"] + "s", f["dir"] + "ed") and f["dir"] in ("encrypt", "decrypt"), "the verbs do not follow the direction")
    need(re.search(r"proc\(uint32_t v\[2\]\)", b) and 2 * 4 == f["block"], "the block is not two 32-bit words")
    rng = random.Random(c.t.id)
    blocks = [(rng.getrandbits(32), rng.getrandbits(32)) for _ in range(4)]
    main = ("int main(void) {\n" + "".join(
        f'    {{ uint32_t v[2] = {{0x{x:x}u, 0x{y:x}u}}; unsigned r = proc(v); printf("%u %u %u\\n", (unsigned)v[0], (unsigned)v[1], r); }}\n' for x, y in blocks)
        + "    return 0;\n}")
    got = [ints(line) for line in c.run(main).strip().split("\n")]
    want = [list(tea_ref(f["alg"], blk, key, f["rounds"], f["delta"], f["dir"] == "decrypt")) + [f["rounds"]] for blk in blocks]
    need(got == want, f"proc is not {f['alg']} {f['dir']}ion with {f['rounds']} iterations and delta {f['delta']:#x}: {got[:1]} vs {want[:1]}")
    other = list(tea_ref(f["alg"], blocks[0], key, f["rounds"] + 1, f["delta"], f["dir"] == "decrypt"))
    need(other != got[0][:2], "one more iteration gives the same block: the round count is not tested")


@claim
def rc4_reference(c: Ctx):
    f, b = c.f, c.body
    mt = re.search(r'static const char key\[\] = "(\w+)";', b)
    key = mt.group(1).encode() if mt else bytes(int(x, 16) for x in re.search(r"static const uint8_t key\[\] = \{(.*?)\};", b).group(1).replace(" ", "").split(","))
    need(len(key) == f["n"], f"the key has {len(key)} bytes, the sentence says {f['n']}")
    need(f["state"] == 256 and f["alg"] == "RC4", "RC4 keeps a state of 256 byte values")
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


@claim
def stream_reference(c: Ctx):
    f, b = c.f, c.body
    alg = c.key.split("/")[1]
    need(tuple(f["rot"]) == ((16, 12, 8, 7) if alg == "chacha" else (7, 9, 13, 18)), "the rotation amounts are not those of the algorithm")
    need(f["half"] * 2 == f["rounds"] and f["key_bits"] == 256 and f["block"] == 64, "rounds, key size or block size are inconsistent")
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
    ref = (f_cipher.chacha_block if alg == "chacha" else f_cipher.salsa_block)(key, counter, nonce, f["rounds"])
    need(ints(out) == ref + [f["rounds"]], f"proc is not {f['alg']} with {f['rounds']} rounds")
    other = (f_cipher.chacha_block if alg == "chacha" else f_cipher.salsa_block)(key, counter, nonce, f["rounds"] + 2)
    need(other != ref, "two more rounds give the same block: the round count is not tested")


# ====================================================================================================================================
# encoding
# ====================================================================================================================================

def _enc_alphabet(body: str) -> str:
    m = re.search(r'static const char A\[\] = "((?:[^"\\]|\\.)*)";', body)
    return m.group(1).replace('\\"', '"').replace("\\\\", "\\")


def _enc_probe(c: Ctx):
    t = c.t
    base, direction = t.variant.split("_")
    alpha, pad = _enc_alphabet(c.body), t.params["pad"]
    rng = random.Random(t.id)
    vecs = f_encoding.vectors(rng) + [b"A", b"AB", b"ABC"]
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
    if alpha != own:
        raise ClaimFailed("the alphabet of the sentence is not the one of the C")
    if direction == "encode":
        want = [f_encoding.encode_ref(base, v, alpha, pad) for v in vecs]
    else:
        want = [v.hex() for v in vecs]
    need(got == want, f"proc is not {base} {direction} with this alphabet: {got[:2]} vs {want[:2]}")
    return base, direction


@claim
def encoding_reference(c: Ctx):
    base, direction, alpha, _pad, _v, _g = c.once("enc", _enc_probe)
    f = c.f
    need((f["base"], f["dir"]) == (base, direction), "the sentence names another encoding or direction")
    need(len(set(alpha)) == len(alpha) == f["radix"], f"the alphabet has {len(alpha)} characters, the sentence says {f['radix']}")
    _enc_matches(c, alpha)
    if base != "base58":
        need((f["gin"], f["gout"], f["bits"]) == {"base64": (3, 4, 6), "base32": (5, 8, 5)}[base] and 2 ** f["bits"] == f["radix"] and f["gin"] * 8 == f["gout"] * f["bits"],
             "groups of bytes and bits per character are inconsistent")


@claim
def alphabet_semantics(c: Ctx):
    base, direction, alpha, _pad, _v, _g = c.once("enc", _enc_probe)
    f = c.f
    need(f["alen"] == len(alpha) == f["radix"], f"the alphabet has {len(alpha)} characters, the sentence says {f['alen']}")
    if "s" in f:
        need(f["s"] == alpha, "the alphabet of the sentence is not the one of the C, byte for byte")
    need(f["agent"] == ("encoder" if direction == "encode" else "decoder"), "the sentence names the other direction")
    _enc_matches(c, alpha)           # the character at position i stands for the value i: the output equals the reference that uses this very string


# ====================================================================================================================================
# strobf: every variant says what proc writes
# ====================================================================================================================================

@claim
def writes_exact_string(c: Ctx):
    s = c.f["s"]
    need(c.f["len"] == len(s), f"the sentence says {c.f['len']} characters, the string has {len(s)}")
    out = c.run(f"""int main(void) {{
    unsigned char out[96];
    memset(out, 0xAA, sizeof out);
    proc((char *)out);
    size_t n = strlen((char *)out), spill = 0;
    for (size_t i = n + 1; i < sizeof out; i++) spill += out[i] != 0xAA;
    printf("%zu %zu %s\\n", n, spill, out);
    return 0;
}}""").rstrip("\n")
    need(out == f"{len(s)} 0 {s}", f"proc writes {out!r}, the sentence says {s!r} ({len(s)} characters then a NUL byte, nothing after)")


# ====================================================================================================================================
# pktsum
# ====================================================================================================================================

def _pkt_models(c: Ctx):
    """Packets built from the FACTS of a sentence with the Python model of the format, and what proc says about each."""
    t, p = c.t, c.t.params
    rng = random.Random(t.id)
    H, W, kind, magic = p["H"], p["W"], p["kind"], p["magic"]
    lenform = t.variant == "len_field"
    out = []                                           # (name, packet bytes, bool: valid expected)

    def build(S=None, L=None, endian="little", junk=0):
        while True:
            m = rng.randint(4, 12)
            head = bytearray(rng.randrange(256) for _ in range(H))
            head[0] = magic
            if lenform:
                head[L] = m
                pay = bytes(rng.randrange(256) for _ in range(m))
                body = bytes(head) + pay
                cov = body[H:H + m]
            else:
                pay = bytes(rng.randrange(256) for _ in range(m))
                body = bytes(head) + pay
                cov = body[S:]
            st = f_pktsum.csum(kind, cov).to_bytes(W, endian)
            if W == 2 and st[0] == st[1]:
                continue
            return body, st, cov

    return rng, build, (H, W, kind, magic, lenform)


def _pkt_probe(c: Ctx):
    t, p = c.t, c.t.params
    rng, build, (H, W, kind, magic, lenform) = _pkt_models(c)
    packets = []                                       # (tag, bytes)

    def add(tag, b):
        packets.append((tag, bytes(b)))

    # byte order: the checksum stored little- and big-endian (only when the checksum is 16 bits wide)
    if W == 2:
        for _ in range(3):
            for e in ("little", "big"):
                body, st, _cov = build(S=p["S"], L=p["L"], endian=e)
                add(f"end:{e}", body + st)
    # the start of the range: valid packets, a flipped covered byte, a flipped byte before the range
    for _ in range(3):
        body, st, cov = build(S=p["S"], L=p["L"], endian=p["endian"] or "little")
        pkt = bytearray(body + st)
        add("valid", pkt)
        lo, hi = (H, H + len(cov)) if lenform else (p["S"], len(body))
        for i in range(1, len(body)):
            if lenform and i == p["L"]:
                continue
            q = bytearray(pkt)
            q[i] ^= 0x5A
            add(f"flip:{'cov' if lo <= i < hi else 'out'}", q)
        if lenform:                                    # the length byte: a longer packet with junk behind the checksum, and a shortened length
            add("junk", pkt + bytes(rng.randrange(256) for _ in range(5)))
            q = bytearray(pkt)
            q[p["L"]] = len(cov) - 1
            # the model's verdict for the shortened length: the checksum is then read from the last payload byte, which equals the checksum of the
            # shortened range once in 256 packets: a coincidence is no failure of the claim, so the expectation comes from the model
            coincident = f_pktsum.csum(kind, cov[:-1]).to_bytes(W, "little") == bytes(pkt[H + len(cov) - 1:H + len(cov) - 1 + W])
            add("shorter" if not coincident else "shorter-coincident", q)
            q = bytearray(pkt)
            q[p["L"]] = len(cov) + 1
            add("longer", q)
    main = ("int main(void) {\n" + "".join(f"    {{ static const uint8_t k[] = {c_data(b)}; printf(\"%d\\n\", proc(k, sizeof k)); }}\n" for _tag, b in packets) + "    return 0;\n}")
    got = ints(c.run(main))
    need(len(got) == len(packets), "the packet run did not finish")
    return [(tag, g) for (tag, _b), g in zip(packets, got)]


def _pkt_fact(c: Ctx, name: str, param: str):
    """The value the sentence states about this task: the rendered fact when this very question was asked, else the generator's parameter."""
    return c.f["n"] if c.f.get("ask") == name else None


@claim
def pkt_endian_behaviour(c: Ctx):
    e = c.f["e"]
    need(c.t.params["W"] == 2 and c.f["first"] == ("low" if e == "little" else "high"), "the checksum is not 16 bits wide or the first byte is the wrong one")
    res = c.once("pkt", _pkt_probe)
    mine, other = [g for tag, g in res if tag == f"end:{e}"], [g for tag, g in res if tag == f"end:{'big' if e == 'little' else 'little'}"]
    need(mine and set(mine) == {1}, f"a checksum stored {e}-endian is not accepted: {mine}")
    need(set(other) == {0}, f"a checksum stored the other way round is accepted: {other}")


@claim
def pkt_range_start(c: Ctx):
    n = c.f["n"]
    p = c.t.params
    want = p["H"] if c.t.variant == "len_field" else p["S"]
    need(n == want, f"the range starts at {want}, the sentence says {n}")
    res = c.once("pkt", _pkt_probe)
    need([g for tag, g in res if tag == "valid"] == [1] * 3 and all(g == 0 for tag, g in res if tag == "flip:cov") and
         all(g == 1 for tag, g in res if tag == "flip:out"),
         "a flipped byte from the start of the range does not fail the checksum, or a byte before it does")
    need(any(tag == "flip:cov" for tag, _g in res), "no covered byte was flipped")


@claim
def pkt_length_byte(c: Ctx):
    need(c.t.variant == "len_field", "no length byte in this variant")
    need(c.f["n"] == c.t.params["L"], f"the length byte is at {c.t.params['L']}, the sentence says {c.f['n']}")
    res = c.once("pkt", _pkt_probe)
    need(all(g == 1 for tag, g in res if tag in ("valid", "junk")), "a valid packet (with junk behind the checksum) is not accepted: the checksum does not follow the payload")
    need(all(g != 1 for tag, g in res if tag in ("shorter", "longer")), "a packet with another length byte is accepted")
    need(all(g == 1 for tag, g in res if tag == "shorter-coincident"), "a packet whose shortened range happens to have the stored checksum is not accepted")


# ====================================================================================================================================
# bounds
# ====================================================================================================================================

def _scan_probe(c: Ctx, lo: int, hi: int):
    return ints(c.run("""int main(void) {
    int first = 0, last = 0, cnt = 0, other = 0;
    for (int v = %d; v <= %d; v++) {
        int r = proc(v);
        if (r != 0 && r != 1) other++;
        if (r) { if (!cnt) first = v; last = v; cnt++; }
    }
    printf("%%d %%d %%d %%d %%d %%d %%d %%d\\n", first, last, cnt, other, proc(-2147483647 - 1), proc(2147483647), proc(-1000000), proc(1000000));
    return 0;
}""" % (lo, hi)))


@claim
def returns_one_exactly_in_range(c: Ctx):
    lo, hi = c.f["lo"], c.f["hi"]
    first, last, cnt, other, *far = c.once("scan", lambda c: _scan_probe(c, -2000, 7000))
    need((first, last, cnt) == (lo, hi, hi - lo + 1), f"proc returns 1 for {first} .. {last} ({cnt} integers), the sentence says {lo} .. {hi}")
    need(other == 0 and not any(far), "proc returns something other than 0 or 1, or accepts a far-away integer")


@claim
def returns_one_for_window(c: Ctx):
    lo, hi, w = c.f["lo"], c.f["hi"], c.f["w"]
    first, last, cnt, other, *far = c.once("scan", lambda c: _scan_probe(c, -2000, 7000))
    need((first, last, cnt) == (lo, hi, w) and hi - lo + 1 == w, f"proc returns 1 for {first} .. {last} ({cnt} integers), the sentence says {lo} .. {hi} ({w})")
    need(other == 0 and not any(far), "proc returns something other than 0 or 1, or accepts a far-away integer")


@claim
def index_limit_range(c: Ctx):
    lo, hi = c.f["lo"], c.f["hi"]
    n = int(re.search(r"T\[(\d+)\]", c.t.body).group(1))
    out = ints(c.run(f"""int main(void) {{
    int first = -1, last = -1, cnt = 0, lowest = 1000;
    for (unsigned i = 0; i < {n + 40}; i++) {{
        int r = proc(i);
        if (r != -1) {{ if (first < 0) first = (int)i; last = (int)i; cnt++; if (r < lowest) lowest = r; }}
    }}
    printf("%d %d %d %d %d %d %d\\n", first, last, cnt, lowest, proc(0xffffffffu), proc(1000000u), proc((unsigned)-1));
    return 0;
}}"""))
    first, last, cnt, lowest, *far = out
    need((first, last, cnt) == (lo, hi, hi - lo + 1), f"proc accepts the indexes {first} .. {last} ({cnt}), the sentence says {lo} .. {hi}")
    need(lowest >= 0 and far == [-1, -1, -1], "an accepted index gives a negative value, or a far index is accepted")


@claim
def clamp_interval(c: Ctx):
    lo, hi = c.f["lo"], c.f["hi"]
    probes = [-2147483647 - 1, lo - 1000, lo - 1, lo, (lo + hi) // 2, hi, hi + 1, hi + 1000, 2147483647]
    out = ints(c.run("int main(void) {\n" + "".join(f'    printf("%d\\n", proc({p if p != -2147483647 - 1 else "-2147483647 - 1"}));\n' for p in probes) + "    return 0;\n}"))
    want = [min(max(p, lo), hi) for p in probes]
    need(out == want, f"proc gives {out}, a clamp to {lo} .. {hi} gives {want}")


@claim
def returns_loop_value(c: Ctx):
    out = ints(c.run(f'int main(void) {{ printf("%d\\n", proc({c.f["n0"]})); return 0; }}'))
    need(out == [c.f["n"]], f"proc({c.f['n0']}) returns {out}, the sentence says {c.f['n']}")


@claim
def char_class_count(c: Ctx):
    out = ints(c.run("""int main(void) {
    int cnt = 0, other = 0;
    for (int c = 0; c < 256; c++) { int r = proc((unsigned char)c); cnt += r == 1; other += r != 0 && r != 1; }
    printf("%d %d\\n", cnt, other);
    return 0;
}"""))
    need(out[0] == c.f["n"] and c.f["total"] == 256 and c.f["top"] == 255, f"proc returns 1 for {out[0]} byte values, the sentence says {c.f['n']}")
    need(out[1] == 0, "proc returns something other than 0 or 1")


# ====================================================================================================================================
# antidebug: the fakes of the task's own harness, other mains
# ====================================================================================================================================

def _pre(c: Ctx) -> str:
    return c.t.check[2]


@claim
def ptrace_request(c: Ctx):
    raw = c.key.endswith("raw")
    proof = "last_no == SYS_ptrace && last_req == PTRACE_TRACEME" if raw else "last_req == PTRACE_TRACEME"
    out = ints(c.run(f"""int main(void) {{
    fake_rv = 0;
    int ok = proc();
    int asked = {proof};
    fake_rv = -1;
    int fail = proc();
    printf("%d %d %d\\n", ok, fail, asked);
    return 0;
}}""", _pre(c)))
    need(out == [0, c.f["rv"], 1], f"proc returns {out[:2]} for a request that succeeds and one that fails, the sentence says 0 and {c.f['rv']}; request asked for PTRACE_TRACEME: {out[2]}")
    need(c.f["tech"] == "ptrace", "the technique word")


@claim
def tracerpid_status(c: Ctx):
    out = ints(c.run("""int main(void) {
    snprintf(fake_buf, sizeof fake_buf, "Name:\\tx\\nTracerPid:\\t0\\nUid:\\t1\\n");
    int a = proc();
    snprintf(fake_buf, sizeof fake_buf, "Name:\\tx\\nTracerPid:\\t4242\\nUid:\\t1\\n");
    int b = proc();
    snprintf(fake_buf, sizeof fake_buf, "Name:\\tx\\nTracerPid:\\t1\\nUid:\\t1\\n");
    int c = proc();
    printf("%d %d %d\\n", a, b, c);
    return 0;
}""", _pre(c)))
    need(out == [0, c.f["rv"], c.f["rv"]], f"proc returns {out} for TracerPid 0, 4242, 1; the sentence says 0 and {c.f['rv']}")


@claim
def env_variable_read(c: Ctx):
    name = c.f["NAME"]
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


@claim
def timing_threshold(c: Ctx):
    thr = c.f["thr"]
    ds = [0, thr - 1, thr, thr + 1, thr * 2 + 5]
    out = c.run("int main(void) {\n" + "".join(f'    fake_d = {d}LL; fake_calls = 0; {{ int r = proc(); printf("%d %d\\n", r, fake_calls); }}\n' for d in ds) + "    return 0;\n}", _pre(c))
    rows = [ints(line) for line in out.strip().split("\n")]
    need([r[0] != 0 for r in rows] == [d > thr for d in ds], f"proc reports a debugger for the elapsed times {ds} as {[r[0] for r in rows]}; the sentence says it does so above {thr}")
    need(all(r[1] == 2 for r in rows), f"proc reads the clock {[r[1] for r in rows]} times, not two")
    need(c.f["scale"] == (10 ** 9 if c.f["unit"] == "nanoseconds" else 10 ** 6), "the unit does not match its scale")
    need(c.f["unit"] == ("nanoseconds" if c.key.endswith("ns") else "microseconds"), "the unit is not the one of this variant's clock")


# ====================================================================================================================================
# multi
# ====================================================================================================================================

@claim
def returns_for_argument(c: Ctx):
    out = ints(c.run(f'int main(void) {{ printf("%u\\n", (unsigned)proc({c.f["n0"]}u)); return 0; }}'))
    need(out == [c.f["n"]], f"proc({c.f['n0']}) returns {out}, the sentence says {c.f['n']}")


@claim
def xors_key_byte(c: Ctx):
    out = ints(c.run(f'int main(void) {{ uint8_t z[8] = {{0}}, b[8] = {{0, 1, 2, 3, 4, 5, 6, 7}}; proc(z, 8, {c.f["seed"]}u); proc(b, 8, {c.f["seed"]}u); '
                     'for (int i = 0; i < 8; i++) printf("%d %d ", z[i], b[i]); return 0; }'))
    k = c.f["k"]
    need(out == [x for i in range(8) for x in (k, i ^ k)], f"proc XORs {out[:2]} into the buffer for the seed {c.f['seed']}, the sentence says {k:#x}")


@claim
def accepts_only_the_secret(c: Ctx):
    s = c.f["s"]
    other = ("y" if s[0] != "y" else "z") + s[1:]
    out = ints(c.run(f'int main(void) {{ printf("%d %d %d %d %d\\n", proc({c_str(s)}), proc({c_str(s + "x")}), proc({c_str(s[:-1])}), proc(""), proc({c_str(other)})); return 0; }}'))
    need(out == [1, 0, 0, 0, 0] and c.f["len"] == len(s), f"proc returns {out} for the secret, the secret plus a character, a shorter string, nothing, another string")


@claim
def limit_for_mode(c: Ctx):
    out = ints(c.run('int main(void) {\n    char s[100];\n    for (int len = 0; len < 90; len++) { memset(s, 97, (size_t)len); s[len] = 0; printf("%%d ", proc(s, %du)); }\n    return 0;\n}' % c.f["mode"]))
    n = c.f["n"]
    need(out == [int(len <= n) for len in range(90)], f"proc accepts the lengths {[i for i, v in enumerate(out) if v][:3]}..{max(i for i, v in enumerate(out) if v)}, the sentence says up to {n}")


# ====================================================================================================================================
# legacy
# ====================================================================================================================================

@claim
def xor_key_repeats(c: Ctx):
    s = c.f["s"]
    n = 3 * len(s) + 2
    out = c.run(f'int main(void) {{ uint8_t b[{n}] = {{0}}; proc(b, {n}); for (int i = 0; i < {n}; i++) printf("%02x", b[i]); return 0; }}')
    need(out == "".join(f"{ord(s[i % len(s)]):02x}" for i in range(n)) and c.f["len"] == len(s), "the buffer is not XORed with the key repeated along it")
    per = next(p for p in range(1, len(s) + 1) if len(s) % p == 0 and all(s[i] == s[i % p] for i in range(len(s))))
    need(per == len(s), "the key repeats with a shorter period than its length")


@claim
def password_equal_only(c: Ctx):
    s = c.f["s"]
    other = ("y" if s[0] != "y" else "z") + s[1:]
    out = ints(c.run(f'int main(void) {{ printf("%d %d %d %d %d\\n", proc({c_str(s)}), proc({c_str(s + "x")}), proc({c_str(s[:-1])}), proc(""), proc({c_str(other)})); return 0; }}'))
    need(out == [1, 0, 0, 0, 0] and c.f["len"] == len(s), f"proc returns {out}: it is not 1 for this string and no other input")


@claim
def keycount_lines(c: Ctx):
    k = c.f["s"]
    text = f"{k}=1\\n#{k}=2\\n{k}=\\n{k}x=3\\n{k}=4\\n x{k}=5\\n"
    out = ints(c.run(f'int main(void) {{ printf("%d %d %d %d\\n", proc("{text}"), proc("{k}="), proc("{k}=v"), proc("#{k}=v")); return 0; }}'))
    need(out == [2, 0, 1, 0] and c.f["len"] == len(k), f"proc counts {out} lines: key=1 and key=4 count, key= with nothing after, a comment, another key and an indented line do not")


@claim
def alphabet_position(c: Ctx):
    s = c.f["s"]
    out = ints(c.run(f'int main(void) {{ const char *a = {c_str(s)}; for (int i = 0; a[i]; i++) printf("%d ", proc(a[i])); '
                     'const char *x = "01899ABC!~ "; for (int i = 0; x[i]; i++) printf("%d ", proc(x[i])); return 0; }'))
    miss = [-1 if ch not in s else s.index(ch) for ch in "01899ABC!~ "]
    need(out == list(range(len(s))) + miss and c.f["miss"] == -1 and c.f["len"] == len(s) == 32, "proc does not return the position in the string, or -1 for a character that is not in it")


@claim
def magic_header(c: Ctx):
    m, v = c.f["m"], c.f["v"]
    def hdr(magic_bytes, ver, n=8):
        return (magic_bytes + struct.pack("<I", ver))[:n]
    cases = [("ok", hdr(struct.pack("<I", m), ver)) for ver in range(v + 1)] + [("hi", hdr(struct.pack("<I", m), v + 1)), ("bad", hdr(struct.pack("<I", m ^ 1), 0)),
             ("short", hdr(struct.pack("<I", m), 0, 7))]
    if struct.pack(">I", m) != struct.pack("<I", m):
        cases.append(("big", hdr(struct.pack(">I", m), 0)))
    main = "int main(void) {\n" + "".join(f"    {{ static const uint8_t h[] = {c_data(b)}; printf(\"%d\\n\", proc(h, sizeof h)); }}\n" for _t, b in cases) + "    return 0;\n}"
    got = ints(c.run(main))
    res = {}
    for (tag, _b), g in zip(cases, got):
        res.setdefault(tag, []).append(g)
    need(res["ok"] == list(range(v + 1)), f"headers with the version 0 .. {v} give {res['ok']}")
    need(res["hi"] == [-3] and res["bad"] == [-2] and res["short"] == [-1], f"version {v + 1}, a wrong magic or a short header give {res['hi']} {res['bad']} {res['short']}")
    need(res.get("big", [-2]) == [-2], "the magic written as big-endian bytes is accepted: the bytes of the header are the little-endian bytes of the value")
    need(c.f["m"] == m, "")


@claim
def lcg_reference(c: Ctx):
    a, r, mask = c.f["a"], c.f["r"], c.f["mask"]
    mm = re.search(r"for \(int i = 0; i < (\d+); i\+\+\) seed = \(seed \* (\d+)u \+ (\d+)u\) & 0x7fffffffu;", c.t.body)
    need(mm and (int(mm.group(1)), int(mm.group(2))) == (r, a) and mask == 0x7fffffff, "rounds, multiplier or mask differ from the C")
    add = int(mm.group(3))
    seeds = [1, 12345, 0xdeadbeef]
    out = ints(c.run("int main(void) {\n" + "".join(f'    printf("%u\\n", proc({s}u));\n' for s in seeds) + "    return 0;\n}"))
    def ref(s):
        for _ in range(r):
            s = ((s * a + add) & M32) & mask
        return s
    need(out == [ref(s) for s in seeds], "proc is not r rounds of seed * a + c, masked")


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


HASHES = {"FNV-1a": h_fnv, "djb2": h_djb2, "sdbm": h_sdbm, "Murmur-style": h_murmur_style}
HASH_INIT = {"FNV-1a": 2166136261, "djb2": 5381, "sdbm": 0, "Murmur-style": 0x9747b28c}
HASH_WORDS = [b"", b"a", b"abc", b"hello world", b"The quick brown fox", b"admin_cache42", b"/tmp/x/y"]


def _hash_probe(c: Ctx):
    main = "int main(void) {\n" + "".join(f'    printf("%u\\n", (unsigned)proc({c_str(w.decode())}));\n' for w in HASH_WORDS) + "    return 0;\n}"
    return ints(c.run(main))


@claim
def hash_final_xor(c: Ctx):
    ref, k = HASHES[c.f["name"]], c.f["c"]
    got = c.once("hash", _hash_probe)
    need(got == [ref(w) ^ k for w in HASH_WORDS], f"proc is not the hash XORed with {k:#x}")


@claim
def hash_start_value(c: Ctx):
    got = c.once("hash", _hash_probe)
    need(c.f["init"] == HASH_INIT[c.f["name"]] and got[0] == c.f["init"] ^ c.f["c"], f"the hash of the empty string is {got[0] ^ c.f['c']:#x}, the sentence says {c.f['init']:#x}")


@claim
def hash_name_is_the_algorithm(c: Ctx):
    name = c.f["name"]
    got = c.once("hash", _hash_probe)
    if name == "Murmur-style":
        need("0x5bd1e995u" in c.body and "0x9747b28cu" in c.body, "the loop does not use MurmurHash2's multiplier and seed")
        need(got != [murmur2(w, 0x9747b28c) ^ c.f["c"] for w in HASH_WORDS], "this IS MurmurHash2: the name would be Murmur, not Murmur-style")
    else:
        need(name in ("FNV-1a", "djb2", "sdbm"), f"{name!r} is not an algorithm this test has a reference for")
    need(got == [HASHES[name](w) ^ c.f["c"] for w in HASH_WORDS], f"proc is not {name}")


@claim
def skip_one_value(c: Ctx):
    b = c.f["b"]
    out = ints(c.run('int main(void) {\n    for (int v = 0; v < 256; v++) { uint8_t x = (uint8_t)v; printf("%%u ", (unsigned)proc(&x, 1)); }\n'
                     '    uint8_t t[4] = {0x11, 0x%02x, 0x22, 0x%02x}; printf("%%u\\n", (unsigned)proc(t, 4));\n    return 0;\n}' % (b, b)))
    want = [(0 if v == b else v) + 1 for v in range(256)]
    tail = ((0x11 if b != 0x11 else 0) + (0x22 if b != 0x22 else 0)) + 4
    need(out == want + [tail], f"proc does not skip exactly the byte {b:#04x} and add the length")


@claim
def rotate_reference(c: Ctx):
    k, r = c.f["k"], c.f["r"]
    xs = [0, 1, 0xdeadbeef, 0x80000000]
    out = ints(c.run("int main(void) {\n" + "".join(f'    printf("%u\\n", (unsigned)proc({x}u));\n' for x in xs) + "    return 0;\n}"))
    need(out == [(((x ^ k) << r) | ((x ^ k) >> (32 - r))) & M32 for x in xs], "proc is not XOR then a left rotation")


@claim
def threshold_values(c: Ctx):
    lo, hi = c.f["lo"], c.f["hi"]
    out = ints(c.run("""int main(void) {
    int first = 0, last = 0, cnt = 0, neg = 0, pos = 0, odd = 0;
    for (int v = -2000; v <= 7000; v++) { int x = proc(v); if (x == 0) { if (!cnt) first = v; last = v; cnt++; } else if (x == -1) neg++; else if (x == 1) pos++; else odd++; }
    printf("%d %d %d %d %d %d\\n", first, last, cnt, neg, pos, odd);
    return 0;
}"""))
    need(out[:3] == [lo, hi, hi - lo + 1] and out[3] == lo + 2000 and out[4] == 7000 - hi and out[5] == 0 and (c.f["below"], c.f["inside"], c.f["above"]) == (-1, 0, 1),
         f"proc returns 0 for {out[0]} .. {out[1]}, -1 below and 1 above: {out}")
