"""String obfuscation: recover the plaintext `proc` writes into its argument.

Plaintext is a plausible string (word pairs with a random number, paths, hostnames) so a model cannot memorise a short list.
Forms: a stack string assigned byte by byte in random order, xor-at-use with one byte, a rolling key (K + i*S), alternating keys,
a constant subtracted, an xor over the reversed array, and rot-N over letters. The generator derives the encoded bytes from the
plaintext, so the plaintext is the truth by construction; the harness runs the C and prints what it writes.

Round 5 (M1): an encoded variant is only a task when the compiled object at its own cc/opt still holds the string ENCODED. The compiler
often decodes a constant array at compile time (gcc -O2/-O3, clang -O1 and up: the string becomes immediates or an XMM constant, no XOR
is left). `Task.object_gate` says what the object must show (survive.py checks it and the generator draws again otherwise): the plaintext
must not be in it, the stored bytes (or the key) must be. A stack string has no encoded form, so it carries no gate.
"""
from __future__ import annotations

from .core import Task, c_bytes

FAMILY = "strobf"
WORDS = ["admin", "login", "token", "secret", "config", "cache", "debug", "server", "client", "socket", "packet", "buffer", "license",
         "update", "backup", "session", "cookie", "handler", "module", "plugin", "status", "proxy", "tunnel", "beacon", "crypto",
         "kernel", "driver", "shadow"]
SHAPES = ["{a}_{b}{n}", "{a} {b} {n}", "/tmp/{a}/{b}", "{a}.{b}.cfg", "{a}-{b}-{n}", "http://{a}.local/{b}", "{a}/{b}/{n}"]


def plaintext(rng) -> str:
    a, b = rng.sample(WORDS, 2)
    return rng.choice(SHAPES).format(a=a, b=b, n=rng.randint(10, 99))


def _other(v: int, k: int) -> int:
    """v, unless it equals k: then the next value (1..255, wrapping). Only the draws that would repeat k change (0.46%)."""
    return v % 255 + 1 if v == k else v


def _gate(p: str, stored: bytes, *keys: int) -> dict:
    """What the compiled object must show (survive.object_gate_check): the plaintext hidden, the stored bytes present (`keys`, the decode constants, is a record: since round 6 a key alone proves nothing)."""
    return {"hidden": p, "stored": stored.hex(), "keys": list(keys)}


def _finish(variant, body, p, params, gate=None):
    main = 'int main(void) {\n    char out[96];\n    proc(out);\n    puts(out);\n    return 0;\n}'
    wrong = p[:-1] + ("x" if p[-1] != "x" else "y")
    q = "What string does proc write into its argument? Give it as plain ASCII text."
    return Task(kind=FAMILY, variant=variant, body=body, question=q, truth={"score": "exact_head", "accept": [p]},
                params={"plain": p, **params}, wrong=wrong, check=(main, p + "\n"), object_gate=gate)


def v_stack_string(rng):
    p = plaintext(rng)
    order = list(range(len(p)))
    rng.shuffle(order)
    stores = "\n".join(f"    s[{i}] = '{p[i]}';" for i in order)
    body = f"""void proc(char *out) {{
    char s[{len(p) + 1}];
{stores}
    s[{len(p)}] = 0;
    memcpy(out, s, sizeof s);
}}"""
    return _finish("stack_string", body, p, {})


def _array_variant(variant, enc_fn, use_fn, rng, extra=None):
    p = plaintext(rng)
    k = rng.randint(1, 255)
    extra = extra(rng, k) if extra else {}
    enc = enc_fn(p, k, extra)
    body = f"""void proc(char *out) {{
    static const uint8_t enc[] = {{{c_bytes(enc)}}};
    for (size_t i = 0; i < sizeof enc; i++) out[i] = {use_fn(k, extra)};
    out[sizeof enc] = 0;
}}"""
    return _finish(variant, body, p, {"key": k, **extra}, _gate(p, enc, k, *([extra["k1"]] if "k1" in extra else [])))


def v_xor_const(rng):
    return _array_variant("xor_const", lambda p, k, e: bytes(ord(c) ^ k for c in p), lambda k, e: f"(char)(enc[i] ^ 0x{k:02x})", rng)


def v_xor_rolling(rng):
    return _array_variant("xor_rolling", lambda p, k, e: bytes(ord(c) ^ ((k + i * e["step"]) & 0xff) for i, c in enumerate(p)),
                          lambda k, e: f"(char)(enc[i] ^ (uint8_t)(0x{k:02x} + i * {e['step']}))", rng, lambda r, k: {"step": r.randint(1, 7)})


def v_alt_xor(rng):
    return _array_variant("alt_xor", lambda p, k, e: bytes(ord(c) ^ (e["k1"] if i & 1 else k) for i, c in enumerate(p)),
                          lambda k, e: f"(char)(enc[i] ^ ((i & 1) ? 0x{e['k1']:02x} : 0x{k:02x}))", rng, lambda r, k: {"k1": _other(r.randint(1, 255), k)})


def v_sub_const(rng):
    return _array_variant("sub_const", lambda p, k, e: bytes((ord(c) + k) & 0xff for c in p), lambda k, e: f"(char)(enc[i] - {k})", rng)


def v_reverse_xor(rng):
    p = plaintext(rng)
    k = rng.randint(1, 255)
    enc = bytes(ord(p[len(p) - 1 - j]) ^ k for j in range(len(p)))
    body = f"""void proc(char *out) {{
    static const uint8_t enc[] = {{{c_bytes(enc)}}};
    size_t n = sizeof enc;
    for (size_t i = 0; i < n; i++) out[i] = (char)(enc[n - 1 - i] ^ 0x{k:02x});
    out[n] = 0;
}}"""
    return _finish("reverse_xor", body, p, {"key": k}, _gate(p, enc, k))


def v_rot_n(rng):
    p = plaintext(rng)
    n = rng.choice([n for n in range(1, 26) if n != 13] + [13])   # 13 is legal, just not over-weighted
    enc = "".join(chr((ord(c) - 97 - n) % 26 + 97) if "a" <= c <= "z" else c for c in p)
    body = f"""void proc(char *out) {{
    static const char enc[] = "{enc}";
    size_t i;
    for (i = 0; enc[i]; i++) {{
        char c = enc[i];
        if (c >= 'a' && c <= 'z') c = (char)('a' + (c - 'a' + {n}) % 26);
        out[i] = c;
    }}
    out[i] = 0;
}}"""
    return _finish("rot_n", body, p, {"shift": n, "enc": enc}, _gate(p, enc.encode(), n))


VARIANTS = {"stack_string": v_stack_string, "xor_const": v_xor_const, "xor_rolling": v_xor_rolling, "alt_xor": v_alt_xor,
            "sub_const": v_sub_const, "reverse_xor": v_reverse_xor, "rot_n": v_rot_n}
