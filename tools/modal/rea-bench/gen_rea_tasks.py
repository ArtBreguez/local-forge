"""Generate reverse-engineering training/eval tasks with automatic ground truth.

Each task is a small C program built from templates with random parameters (keys, constants, algorithms, flags). The answer
is known from the parameters, so no human writes labels. Programs are compiled stripped at -O1/-O2 by the consumer
(tools/modal_rea_data.py). Nothing here is copied from any public dataset.

    python3 gen_rea_tasks.py --n 300 --seed 7 --out /tmp/rea-tasks   # writes <id>.c and <id>.json per task
"""
from __future__ import annotations

import argparse
import json
import pathlib
import random
import string

HDR = "#include <stdio.h>\n#include <stdlib.h>\n#include <string.h>\n#include <stdint.h>\n\n"


def rword(rng, n):
    return "".join(rng.choice(string.ascii_lowercase + string.digits) for _ in range(n))


def t_xor_key(rng):
    key = rword(rng, rng.choice([4, 6, 8, 12]))
    body = f'''void proc(uint8_t *buf, size_t n) {{
    static const char key[] = "{key}";
    for (size_t i = 0; i < n; i++) buf[i] ^= (uint8_t)key[i % {len(key)}];
}}'''
    return body, "Recover the repeating XOR key as an ASCII string.", {"score": "exact", "accept": [key]}, "xor-key"


def t_password(rng):
    pw = rword(rng, rng.choice([7, 9, 11, 14]))
    mode = rng.choice(["strcmp", "ct", "len_then_memcmp"])
    if mode == "strcmp":
        chk = f'return strcmp(pw, "{pw}") == 0;'
    elif mode == "ct":
        chk = f'''static const char want[] = "{pw}"; size_t n = strlen(want);
    if (strlen(pw) != n) return 0; unsigned d = 0;
    for (size_t i = 0; i < n; i++) d |= (unsigned)(pw[i] ^ want[i]);
    return d == 0;'''
    else:
        chk = f'if (strlen(pw) != {len(pw)}) return 0; return memcmp(pw, "{pw}", {len(pw)}) == 0;'
    return f"int proc(const char *pw) {{\n    {chk}\n}}", "Which input string makes this function return 1?", {"score": "exact", "accept": [pw]}, "password"


def t_magic(rng):
    magic = rng.getrandbits(32)
    maxv = rng.randint(1, 9)
    body = f'''int proc(const uint8_t *hdr, size_t n) {{
    if (n < 8) return -1;
    uint32_t magic; memcpy(&magic, hdr, 4);
    if (magic != 0x{magic:08X}u) return -2;
    uint32_t ver; memcpy(&ver, hdr + 4, 4);
    return ver > {maxv} ? -3 : (int)ver;
}}'''
    return body, "What 32-bit magic value must the header start with (hex), and what is the highest accepted version?", {"score": "all", "accept": [f"{magic:08x}", str(maxv)]}, "magic"


def t_lcg(rng):
    a = rng.choice([1103515245, 1664525, 22695477, 214013, 69069, 6364136223846793005 & 0xffffffff])
    c = rng.choice([12345, 1013904223, 1, 2531011, 0])
    rounds = rng.randint(3, 40)
    body = f'''uint32_t proc(uint32_t seed) {{
    for (int i = 0; i < {rounds}; i++) seed = (seed * {a}u + {c}u) & 0x7fffffffu;
    return seed;
}}'''
    return body, "This is a linear congruential generator. Give the multiplier (decimal) and the number of rounds.", {"score": "all", "accept": [str(a), str(rounds)]}, "lcg"


def t_hash(rng):
    fam = rng.choice(["fnv1a", "djb2", "sdbm", "murmur-ish"])
    tw = rng.getrandbits(32)
    if fam == "fnv1a":
        core = "h = 2166136261u; for (; *s; s++) { h ^= (uint8_t)*s; h *= 16777619u; }"
        names = ["fnv"]
    elif fam == "djb2":
        core = "h = 5381u; for (; *s; s++) h = h * 33u + (uint8_t)*s;"
        names = ["djb2", "djb", "bernstein"]
    elif fam == "sdbm":
        core = "h = 0; for (; *s; s++) h = (uint8_t)*s + (h << 6) + (h << 16) - h;"
        names = ["sdbm"]
    else:
        core = "h = 0x9747b28cu; for (; *s; s++) { h ^= (uint8_t)*s; h *= 0x5bd1e995u; h ^= h >> 15; }"
        names = ["murmur"]
    body = f"uint32_t proc(const char *s) {{\n    uint32_t {core}\n    return h ^ 0x{tw:08x}u;\n}}"
    return body, "Which well-known hash is this based on, and what constant is XORed at the end (hex)?", {"score": "all_groups", "accept": [names, [f"{tw:08x}"]]}, "hash"


def t_skip(rng):
    b = rng.choice([0x20, 0x0a, 0x00, 0x2c, 0x3b, 0x09, 0xff])
    body = f'''uint32_t proc(const uint8_t *p, size_t n) {{
    uint32_t s = 0;
    for (size_t i = 0; i < n; i++) if (p[i] != 0x{b:02x}) s += p[i];
    return s + (uint32_t)n;
}}'''
    return body, "The checksum skips bytes with one specific value. Which value (hex)?", {"score": "any", "accept": [f"0x{b:02x}", f"{b:02x}", str(b)]}, "skip"


def t_keycount(rng):
    key = rng.choice(["port", "host", "user", "token", "level", "mode", "path", "retries"])
    body = f'''int proc(const char *text) {{
    int k = 0; const char *line = text;
    while (*line) {{
        const char *nl = strchr(line, '\\n'); size_t len = nl ? (size_t)(nl - line) : strlen(line);
        if (len > {len(key) + 1} && line[0] != '#' && strncmp(line, "{key}=", {len(key) + 1}) == 0) k++;
        line += len + (nl ? 1 : 0);
    }}
    return k;
}}'''
    return body, "This parses key=value lines. Which key does it count?", {"score": "exact", "accept": [key]}, "keycount"


def t_alphabet(rng):
    alpha = list(string.ascii_lowercase + string.digits[2:8])
    rng.shuffle(alpha)
    alpha = "".join(alpha[:32])
    body = f'''int proc(char c) {{
    static const char alpha[] = "{alpha}";
    const char *p = strchr(alpha, c);
    return p ? (int)(p - alpha) : -1;
}}'''
    return body, "Give the 32-character alphabet string this decoder uses, in order.", {"score": "exact", "accept": [alpha]}, "alphabet"


def t_rotate(rng):
    r = rng.randint(1, 31)
    k = rng.getrandbits(32)
    body = f'''uint32_t proc(uint32_t x) {{
    x ^= 0x{k:08x}u;
    return (x << {r}) | (x >> {32 - r});
}}'''
    return body, "Give the XOR constant (hex) and the left-rotation amount.", {"score": "all", "accept": [f"{k:08x}", str(r)]}, "rotate"


def t_threshold(rng):
    lo, hi = sorted(rng.sample(range(-500, 5000), 2))
    body = f'''int proc(int v) {{
    if (v < {lo}) return -1;
    if (v > {hi}) return 1;
    return 0;
}}'''
    return body, "What are the lower and upper bounds (inclusive) for which this returns 0?", {"score": "all", "accept": [str(lo), str(hi)]}, "threshold"


TEMPLATES = [t_xor_key, t_password, t_magic, t_lcg, t_hash, t_skip, t_keycount, t_alphabet, t_rotate, t_threshold]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    a.out.mkdir(parents=True, exist_ok=True)
    for i in range(a.n):
        fn = TEMPLATES[i % len(TEMPLATES)]
        body, q, truth, kind = fn(rng)
        opt = rng.choice(["-O1", "-O2", "-Os"])
        tid = f"{kind}-{a.seed}-{i:04d}"
        src = HDR + body + '\n\nint main(int argc, char **argv) { (void)argc; (void)argv; puts("ok"); return 0; }\n'
        (a.out / f"{tid}.c").write_text(src)
        (a.out / f"{tid}.json").write_text(json.dumps({"id": tid, "kind": kind, "question": q, "opt": opt, "function": "proc", **truth}))
    print(f"{a.n} tasks -> {a.out}")


if __name__ == "__main__":
    main()
