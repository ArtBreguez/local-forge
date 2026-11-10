"""Base64 / base58 / base32 encoders and decoders, standard or custom alphabet.

Questions: encode or decode, and which encoding (tokens, with `reject` for the other directions and bases), or, for a custom
alphabet, the alphabet itself in index order (`exact_case`: alphabets are case-sensitive and the other scoring modes lower-case
both sides). Truth is the generator's own choice of direction, base and alphabet. The harness encodes seven vectors with Python
(`base64` from the stdlib, translated to the custom alphabet; a plain big-integer base58) and either compares the C encoder's
output with that or feeds it to the C decoder and expects the original bytes back.
"""
from __future__ import annotations

import base64
import string

from .core import Task, c_bytes, is_held, with_decoys

FAMILY = "encoding"

B64 = string.ascii_uppercase + string.ascii_lowercase + string.digits + "+/"
B32 = string.ascii_uppercase + "234567"
B58 = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
STD = {"base64": B64, "base32": B32, "base58": B58}
RADIX = {"base64": 64, "base32": 32, "base58": 58}
BASE_NAMES = {"base64": ["base64", "base-64"], "base32": ["base32", "base-32"], "base58": ["base58", "base-58"]}
OTHER_BASES = ["base16", "base36", "base45", "base62", "base85", "base91", "base32hex", "base64url", "base58check", "ascii85", "z85", "uuencode",
               "radix64", "bech32", "hex"]
BASE_SHAPE = r"base-?\d+[a-z]*"      # any other baseNN in a headline is a second candidate, listed or not
ENCODE = ["encode", "encodes", "encoder", "encoding", "encodes"]
DECODE = ["decode", "decodes", "decoder", "decoding"]


# ---- Python references ----------------------------------------------------------------------------------------------

def b58encode(data: bytes, alpha: str) -> str:
    n, out = int.from_bytes(data, "big"), []
    while n:
        n, r = divmod(n, 58)
        out.append(alpha[r])
    zeros = len(data) - len(data.lstrip(b"\0"))
    return alpha[0] * zeros + "".join(reversed(out))


def encode_ref(base: str, data: bytes, alpha: str, pad: bool) -> str:
    if base == "base58":
        return b58encode(data, alpha)
    std = base64.b64encode(data) if base == "base64" else base64.b32encode(data)
    s = std.decode()
    if not pad:
        s = s.rstrip("=")
    return s.translate(str.maketrans(STD[base], alpha))


def vectors(rng) -> list:
    out = [bytes(rng.randrange(256) for _ in range(n)) for n in rng.sample(range(1, 21), 6)]
    return out + [b"\x00\x00" + bytes(rng.randrange(1, 256) for _ in range(5))]   # leading zeros matter to base58


def c_str(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


# ---- C sources ------------------------------------------------------------------------------------------------------

def src_b64_enc(alpha, pad):
    p2 = "out[o++] = '='; out[o++] = '=';" if pad else ""
    p1 = "out[o++] = '=';" if pad else ""
    return f"""size_t proc(const uint8_t *in, size_t n, char *out) {{
    static const char A[] = {c_str(alpha)};
    size_t i = 0, o = 0;
    for (; i + 2 < n; i += 3) {{
        uint32_t v = ((uint32_t)in[i] << 16) | ((uint32_t)in[i + 1] << 8) | (uint32_t)in[i + 2];
        out[o++] = A[(v >> 18) & 63];
        out[o++] = A[(v >> 12) & 63];
        out[o++] = A[(v >> 6) & 63];
        out[o++] = A[v & 63];
    }}
    if (n - i == 1) {{
        uint32_t v = (uint32_t)in[i] << 16;
        out[o++] = A[(v >> 18) & 63];
        out[o++] = A[(v >> 12) & 63];
        {p2}
    }} else if (n - i == 2) {{
        uint32_t v = ((uint32_t)in[i] << 16) | ((uint32_t)in[i + 1] << 8);
        out[o++] = A[(v >> 18) & 63];
        out[o++] = A[(v >> 12) & 63];
        out[o++] = A[(v >> 6) & 63];
        {p1}
    }}
    out[o] = 0;
    return o;
}}"""


def src_b32_enc(alpha, pad):
    tail = "while (o % 8) out[o++] = '=';" if pad else ""
    return f"""size_t proc(const uint8_t *in, size_t n, char *out) {{
    static const char A[] = {c_str(alpha)};
    uint32_t acc = 0;
    int bits = 0;
    size_t o = 0;
    for (size_t i = 0; i < n; i++) {{
        acc = (acc << 8) | in[i];
        bits += 8;
        while (bits >= 5) {{
            bits -= 5;
            out[o++] = A[(acc >> bits) & 31];
        }}
    }}
    if (bits > 0) out[o++] = A[(acc << (5 - bits)) & 31];
    {tail}
    out[o] = 0;
    return o;
}}"""


def src_b58_enc(alpha, pad):
    return f"""size_t proc(const uint8_t *in, size_t n, char *out) {{
    static const char A[] = {c_str(alpha)};
    uint8_t buf[128];
    size_t len = 0, zeros = 0, o = 0;
    while (zeros < n && in[zeros] == 0) zeros++;
    for (size_t i = zeros; i < n; i++) {{
        unsigned carry = in[i];
        for (size_t j = 0; j < len; j++) {{
            carry += (unsigned)buf[j] << 8;
            buf[j] = (uint8_t)(carry % 58u);
            carry /= 58u;
        }}
        while (carry) {{
            buf[len++] = (uint8_t)(carry % 58u);
            carry /= 58u;
        }}
    }}
    for (size_t i = 0; i < zeros; i++) out[o++] = A[0];
    while (len) out[o++] = A[buf[--len]];
    out[o] = 0;
    return o;
}}"""


def _lookup(radix, table):
    """Either strchr per character or a reverse table built per call; returns (setup, expression for the value or -1)."""
    if table:
        return (f"    int8_t rev[256];\n    memset(rev, -1, sizeof rev);\n    for (int i = 0; i < {radix}; i++) rev[(uint8_t)A[i]] = (int8_t)i;\n",
                "int v = rev[(uint8_t)*s];\n        if (v < 0) return 0;")
    return "", "const char *q = strchr(A, *s);\n        if (!q) return 0;\n        int v = (int)(q - A);"


def src_bits_dec(alpha, shift, table):
    setup, find = _lookup(1 << shift, table)
    return f"""size_t proc(const char *s, uint8_t *out) {{
    static const char A[] = {c_str(alpha)};
{setup}    uint32_t acc = 0;
    int bits = 0;
    size_t o = 0;
    for (; *s && *s != '='; s++) {{
        {find}
        acc = (acc << {shift}) | (uint32_t)v;
        bits += {shift};
        if (bits >= 8) {{
            bits -= 8;
            out[o++] = (uint8_t)(acc >> bits);
        }}
    }}
    return o;
}}"""


def src_b58_dec(alpha, table):
    setup, find = _lookup(58, table)
    return f"""size_t proc(const char *s, uint8_t *out) {{
    static const char A[] = {c_str(alpha)};
{setup}    uint8_t buf[128];
    size_t len = 0, zeros = 0, o = 0;
    while (s[zeros] == A[0]) zeros++;
    for (; *s; s++) {{
        {find}
        unsigned carry = (unsigned)v;
        for (size_t j = 0; j < len; j++) {{
            carry += (unsigned)buf[j] * 58u;
            buf[j] = (uint8_t)carry;
            carry >>= 8;
        }}
        while (carry) {{
            buf[len++] = (uint8_t)carry;
            carry >>= 8;
        }}
    }}
    for (size_t i = 0; i < zeros; i++) out[o++] = 0;
    while (len) out[o++] = buf[--len];
    return o;
}}"""


# ---- tasks ----------------------------------------------------------------------------------------------------------

def _alphabet(rng, base):
    """Standard alphabet (training seeds, half the tasks) or a random shuffle. The held-out seed draws shuffles only: a standard-alphabet
    encoder is one of about 18 programs, so a held-out one would always have a training twin that differs only in its decoys."""
    if rng.random() < 0.5 and not is_held(rng):
        return STD[base], False
    chars = list(STD[base])
    rng.shuffle(chars)
    return "".join(chars), True


def _swap(alpha):
    return alpha[1] + alpha[0] + alpha[2:]


def _make(rng, base, direction):
    alpha, custom = _alphabet(rng, base)
    pad = rng.random() < 0.5
    table = rng.random() < 0.5
    vecs = vectors(rng)
    encoded = [encode_ref(base, v, alpha, pad) for v in vecs]
    shift = {"base64": 6, "base32": 5}.get(base)
    if direction == "encode":
        body = {"base64": src_b64_enc, "base32": src_b32_enc, "base58": src_b58_enc}[base](alpha, pad)
        decls = "".join(f"    static const uint8_t v{i}[] = {{{c_bytes(v)}}};\n" for i, v in enumerate(vecs))
        calls = "".join(f"    proc(v{i}, sizeof v{i}, out); puts(out);\n" for i in range(len(vecs)))
        main = f"int main(void) {{\n{decls}    char out[256];\n{calls}    return 0;\n}}"
        expect = "".join(e + "\n" for e in encoded)
    else:
        body = src_b58_dec(alpha, table) if base == "base58" else src_bits_dec(alpha, shift, table)
        calls = "".join(f'    n = proc({c_str(e)}, out);\n    for (size_t i = 0; i < n; i++) printf("%02x", out[i]);\n    printf("\\n");\n' for e in encoded)
        main = f"int main(void) {{\n    uint8_t out[128];\n    size_t n;\n{calls}    return 0;\n}}"
        expect = "".join(v.hex() + "\n" for v in vecs)
    # a standard alphabet leaves almost nothing random in the source: those always carry decoys (core.decoys), so two tasks are the
    # same program only by accident
    body = with_decoys(body, rng, rng.choice([0, 1, 1, 2]) if custom else rng.choice([1, 1, 2, 2, 3]))
    ask_alpha = custom and rng.random() < 0.6
    if ask_alpha:
        n = RADIX[base]
        q = f"Give the {n}-character alphabet this function uses, in index order (the first character stands for the value 0)."
        truth = {"score": "exact_case", "accept": [alpha]}
        wrong = _swap(alpha)
    else:
        q = "Does this function encode or decode, and which encoding does it implement (base32, base58 or base64)?"
        same, opp = (ENCODE, [w for w in DECODE]) if direction == "encode" else (DECODE, [w for w in ENCODE if w != "encoding"])
        reject = opp + [n for b, ns in BASE_NAMES.items() if b != base for n in ns] + OTHER_BASES
        truth = {"score": "tokens", "accept": [same, BASE_NAMES[base]], "reject": reject, "cand": BASE_SHAPE}
        wrong_dir = "decodes" if direction == "encode" else "encodes"
        wrong = f"It {wrong_dir} {base}"
    return Task(kind=FAMILY, variant=f"{base}_{direction}", body=body, question=q, truth=truth, facts=[[alpha]] if ask_alpha else [],
                params={"base": base, "direction": direction, "alphabet": alpha, "custom": custom, "pad": pad, "asked": "alphabet" if ask_alpha else "ident"},
                wrong=wrong, check=(main, expect))


def _variant(base, direction):
    return lambda rng: _make(rng, base, direction)


VARIANTS = {f"{b}_{d}": _variant(b, d) for b in ("base64", "base58", "base32") for d in ("encode", "decode")}
