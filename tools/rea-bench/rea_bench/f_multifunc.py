"""Multi-function tasks: the answer needs the body of a function that `proc` calls.

The helper is `__attribute__((noinline))` with external linkage and is always called with a value derived from proc's own
arguments (never a constant), so GCC cannot specialise it into `helper.constprop.N` and the symbol the consumer looks up is the
function proc really calls. The task JSON lists the callee names in `helpers`; modal_rea_data.py decompiles them and appends their
pseudocode after proc's (Ghidra names a stripped function FUN_<address> in both places). Questions give the concrete input where
one is needed and ask for a number or the secret string; the truth is evaluated in Python from the same parameters.
"""
from __future__ import annotations

from .core import Task, c_bytes, dec, hx, rword, with_decoys

FAMILY = "multi"
M32 = 0xFFFFFFFF
NOINLINE = "__attribute__((noinline)) "
CALLEE = "proc calls another function (its pseudocode follows proc's) and uses the result."
WORDS = ["admin", "login", "token", "secret", "config", "cache", "debug", "server", "client", "socket", "packet", "buffer"]


def helper_form(rng):
    """(C expression in x, python function, description) for a small uint32 -> uint32 helper."""
    f = rng.randrange(4)
    if f == 0:
        a, b, m = rng.randint(3, 13), rng.randint(1, 50), rng.choice([97, 101, 127, 251])
        return f"(x * {a}u + {b}u) % {m}u", (lambda x: ((x * a + b) & M32) % m), {"form": "affine_mod", "a": a, "b": b, "m": m}
    if f == 1:
        k, s = rng.randint(1, 255), rng.randint(1, 3)
        return f"(x ^ {k}u) + (x >> {s})", (lambda x: ((x ^ k) + (x >> s)) & M32), {"form": "xor_shift", "k": k, "s": s}
    if f == 2:
        s = rng.randint(1, 7)
        return f"((x << {s}) | (x >> {8 - s})) & 0xffu", (lambda x: (((x << s) | (x >> (8 - s))) & 0xff)), {"form": "rotl8", "s": s}
    k = rng.randint(1, 200)
    return f"x * x + {k}u", (lambda x: (x * x + k) & M32), {"form": "square", "k": k}


def _main_eval(probes):
    return "int main(void) {\n" + "".join(f'    printf("%u\\n", (unsigned)proc({p}u));\n' for p in probes) + "    return 0;\n}"


def v_eval_arith(rng):
    expr, fn, hp = helper_form(rng)
    shape = rng.randrange(3)
    k2, d = rng.randint(1, 99), rng.randint(1, 9)
    if shape == 0:
        call, ref = f"helper(x) + {k2}u", lambda x: (fn(x) + k2) & M32
    elif shape == 1:
        call, ref = f"helper(x) ^ helper(x + {d}u)", lambda x: fn(x) ^ fn((x + d) & M32)
    else:
        call, ref = "helper(helper(x))", lambda x: fn(fn(x))
    body = f"""{NOINLINE}uint32_t helper(uint32_t x) {{
    return {expr};
}}

uint32_t proc(uint32_t x) {{
    return {call};
}}"""
    body = with_decoys(body, rng, rng.choice([0, 1, 1, 2]))
    n0 = rng.randint(3, 60)
    probes = [n0, n0 + 1, 0, 5]
    return Task(kind=FAMILY, variant="eval_arith", body=body, helpers=["helper"],
                question=f"{CALLEE} What does proc return when it is called with {n0} (decimal)?",
                truth={"score": "tokens", "accept": [dec(ref(n0))]}, params={**hp, "shape": shape, "n0": n0},
                wrong=str(ref(n0) + 1), check=(_main_eval(probes), "".join(f"{ref(p)}\n" for p in probes)))


def v_key_helper(rng):
    a, b, s, k = rng.randint(3, 40), rng.randint(1, 90), rng.randint(1, 4), rng.randint(1, 255)
    key = lambda seed: ((((seed * a + b) & M32) >> s) ^ k) & 0xff
    body = f"""{NOINLINE}uint8_t helper(uint32_t seed) {{
    return (uint8_t)((((seed * {a}u + {b}u) >> {s}) ^ {k}u) & 0xffu);
}}

void proc(uint8_t *buf, size_t n, uint32_t seed) {{
    uint8_t key = helper(seed);
    for (size_t i = 0; i < n; i++) buf[i] ^= key;
}}"""
    s0 = rng.randint(2, 200)
    main = (f'int main(void) {{\n    uint8_t buf[8] = {{0, 1, 2, 3, 4, 5, 6, 7}};\n    proc(buf, sizeof buf, {s0}u);\n'
            '    for (int i = 0; i < 8; i++) printf("%02x", buf[i]);\n    printf("\\n");\n    return 0;\n}')
    kv = key(s0)
    return Task(kind=FAMILY, variant="key_helper", body=body, helpers=["helper"],
                question=f"{CALLEE} If proc is called with seed = {s0}, which byte value (hex) does it XOR into every byte of the buffer?",
                truth={"score": "tokens", "accept": [hx(kv, 2)]}, params={"a": a, "b": b, "s": s, "k": k, "seed": s0, "key": kv},
                wrong=f"0x{kv ^ 0x10:02x}", check=(main, bytes(i ^ kv for i in range(8)).hex() + "\n"))


def v_decode_secret(rng):
    a, b = rng.sample(WORDS, 2)
    secret = f"{a}_{b}{rng.randint(10, 99)}"
    k = rng.randint(1, 255)
    enc = bytes(ord(c) ^ k for c in secret)
    body = f"""{NOINLINE}void helper(char *out) {{
    static const uint8_t enc[] = {{{c_bytes(enc)}}};
    for (size_t i = 0; i < sizeof enc; i++) out[i] = (char)(enc[i] ^ 0x{k:02x});
    out[sizeof enc] = 0;
}}

int proc(const char *pw) {{
    char tmp[48];
    helper(tmp);
    return strcmp(pw, tmp) == 0;
}}"""
    main = (f'int main(void) {{\n    printf("%d\\n", proc("{secret}"));\n    printf("%d\\n", proc("{secret}x"));\n'
            '    printf("%d\\n", proc(""));\n    return 0;\n}')
    return Task(kind=FAMILY, variant="decode_secret", body=body, helpers=["helper"],
                question=f"{CALLEE} Which input string makes proc return 1?", truth={"score": "exact_head", "accept": [secret]},
                params={"secret": secret, "k": k}, wrong=secret[:-1] + ("0" if secret[-1] != "0" else "1"), check=(main, "1\n0\n0\n"))


def v_chain(rng):
    a, b, k, d, mask = rng.randint(3, 30), rng.randint(1, 99), rng.randint(1, 255), rng.randint(1, 9), rng.choice([0xff, 0x3ff, 0xfff])
    h2 = lambda x: (x * a + b) & M32
    h1 = lambda x: h2(x) ^ k
    ref = lambda x: h1((x + d) & M32) & mask
    body = f"""{NOINLINE}uint32_t helper2(uint32_t x) {{
    return x * {a}u + {b}u;
}}

{NOINLINE}uint32_t helper(uint32_t x) {{
    return helper2(x) ^ {k}u;
}}

uint32_t proc(uint32_t x) {{
    return helper(x + {d}u) & 0x{mask:x}u;
}}"""
    n0 = rng.randint(3, 60)
    probes = [n0, n0 + 1, 0, 5]
    return Task(kind=FAMILY, variant="chain", body=body, helpers=["helper", "helper2"],
                question="proc calls a function that itself calls a third one; the pseudocode of all three is shown. "
                         f"What does proc return when it is called with {n0} (decimal)?",
                truth={"score": "tokens", "accept": [dec(ref(n0))]}, params={"a": a, "b": b, "k": k, "d": d, "mask": mask, "n0": n0},
                wrong=str(ref(n0) ^ 1), check=(_main_eval(probes), "".join(f"{ref(p)}\n" for p in probes)))


def v_limit_helper(rng):
    a, b, m, c = rng.randint(2, 9), rng.randint(1, 30), rng.choice([7, 11, 13, 17]), rng.randint(3, 20)
    limit = lambda mode: ((mode * a + b) % m) + c
    body = f"""{NOINLINE}unsigned helper(unsigned mode) {{
    return (mode * {a}u + {b}u) % {m}u + {c}u;
}}

int proc(const char *s, unsigned mode) {{
    return strlen(s) <= helper(mode);
}}"""
    m0 = rng.randint(2, 40)
    lim = limit(m0)
    main = (f'int main(void) {{\n    char s[80];\n    int best = -1;\n    for (int len = 0; len < 60; len++) {{\n        memset(s, \'a\', (size_t)len);\n'
            f'        s[len] = 0;\n        if (proc(s, {m0}u)) best = len;\n    }}\n    printf("%d\\n", best);\n    return 0;\n}}')
    return Task(kind=FAMILY, variant="limit_helper", body=body, helpers=["helper"],
                question=f"{CALLEE} With mode = {m0}, what is the longest string length (decimal) for which proc returns 1?",
                truth={"score": "tokens", "accept": [dec(lim)]}, params={"a": a, "b": b, "m": m, "c": c, "mode": m0, "limit": lim},
                wrong=str(lim + 1), check=(main, f"{lim}\n"))


VARIANTS = {"eval_arith": v_eval_arith, "key_helper": v_key_helper, "decode_secret": v_decode_secret, "chain": v_chain,
            "limit_helper": v_limit_helper}
