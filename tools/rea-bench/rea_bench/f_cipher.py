"""Cipher recognition from constants and structure: RC4, TEA, XTEA (encrypt and decrypt), ChaCha, Salsa20.

Each task bakes a random key into a textbook implementation. The question asks which cipher it is plus one number that is
read off the code: RC4's key length, or the number of loop iterations (TEA/XTEA) or total rounds (ChaCha/Salsa, a double round
counts as two). TEA and XTEA also ask for the direction. Scoring is token-based with a `reject` list of every other cipher name,
so "XTEA" never passes for "TEA" and "TEA or XTEA" fails. Python references (`rc4`, `tea`, `xtea`, `chacha_block`, `salsa_block`)
produce the harness output; the tests anchor them to published vectors (RC4 "Key"/"Plaintext", RFC 7539 2.3.2, the zero-key
TEA and XTEA vectors).
"""
from __future__ import annotations

from .core import Task, c_bytes, c_u32s, dec, rword, with_decoys

FAMILY = "cipher"
M = 0xFFFFFFFF
DELTA = 0x9E3779B9

NAMES = {"rc4": ["rc4", "arc4", "arcfour"], "tea": ["tea"], "xtea": ["xtea", "x-tea"],
         "chacha": ["chacha", "chacha20", "chacha12", "chacha8", "chacha-20"],
         "salsa": ["salsa", "salsa20", "salsa12", "salsa8", "salsa-20"]}
OTHER_NAMES = ["xxtea", "btea", "aes", "des", "blowfish", "rc5", "rc6", "twofish", "serpent", "poly1305"]
ENC = ["encrypt", "encrypts", "encrypting", "encryption", "encipher", "enciphers"]
DEC = ["decrypt", "decrypts", "decrypting", "decryption", "decipher", "deciphers"]


def reject_for(alg: str) -> list:
    """Every other cipher name. For XTEA plain "tea" stays allowed ("eXtended TEA"); answering just "TEA" still fails
    because the accept group wants "xtea"."""
    return [n for a, ns in NAMES.items() if a != alg for n in ns if not (alg == "xtea" and n == "tea")] + OTHER_NAMES


def rotl(x, n):
    return ((x << n) | (x >> (32 - n))) & M


# ---- Python references ---------------------------------------------------------------------------------------------

def rc4(key: bytes, data: bytes) -> bytes:
    s, j = list(range(256)), 0
    for i in range(256):
        j = (j + s[i] + key[i % len(key)]) & 255
        s[i], s[j] = s[j], s[i]
    i = j = 0
    out = bytearray()
    for b in data:
        i = (i + 1) & 255
        j = (j + s[i]) & 255
        s[i], s[j] = s[j], s[i]
        out.append(b ^ s[(s[i] + s[j]) & 255])
    return bytes(out)


def tea(v, k, rounds, decrypt=False):
    v0, v1 = v
    if not decrypt:
        s = 0
        for _ in range(rounds):
            s = (s + DELTA) & M
            v0 = (v0 + ((((v1 << 4) & M) + k[0]) ^ (v1 + s) ^ ((v1 >> 5) + k[1]))) & M
            v1 = (v1 + ((((v0 << 4) & M) + k[2]) ^ (v0 + s) ^ ((v0 >> 5) + k[3]))) & M
    else:
        s = (DELTA * rounds) & M
        for _ in range(rounds):
            v1 = (v1 - ((((v0 << 4) & M) + k[2]) ^ (v0 + s) ^ ((v0 >> 5) + k[3]))) & M
            v0 = (v0 - ((((v1 << 4) & M) + k[0]) ^ (v1 + s) ^ ((v1 >> 5) + k[1]))) & M
            s = (s - DELTA) & M
    return v0, v1


def xtea(v, k, rounds, decrypt=False):
    v0, v1 = v
    if not decrypt:
        s = 0
        for _ in range(rounds):
            v0 = (v0 + ((((v1 << 4) & M) ^ (v1 >> 5)) + v1 ^ (s + k[s & 3]))) & M
            s = (s + DELTA) & M
            v1 = (v1 + ((((v0 << 4) & M) ^ (v0 >> 5)) + v0 ^ (s + k[(s >> 11) & 3]))) & M
    else:
        s = (DELTA * rounds) & M
        for _ in range(rounds):
            v1 = (v1 - ((((v0 << 4) & M) ^ (v0 >> 5)) + v0 ^ (s + k[(s >> 11) & 3]))) & M
            s = (s - DELTA) & M
            v0 = (v0 - ((((v1 << 4) & M) ^ (v1 >> 5)) + v1 ^ (s + k[s & 3]))) & M
    return v0, v1


SIGMA = [0x61707865, 0x3320646E, 0x79622D32, 0x6B206574]


def chacha_block(key, counter, nonce, rounds):
    st = SIGMA + list(key) + [counter] + list(nonce)
    x = list(st)

    def qr(a, b, c, d):
        x[a] = (x[a] + x[b]) & M; x[d] = rotl(x[d] ^ x[a], 16)
        x[c] = (x[c] + x[d]) & M; x[b] = rotl(x[b] ^ x[c], 12)
        x[a] = (x[a] + x[b]) & M; x[d] = rotl(x[d] ^ x[a], 8)
        x[c] = (x[c] + x[d]) & M; x[b] = rotl(x[b] ^ x[c], 7)

    for _ in range(rounds // 2):
        qr(0, 4, 8, 12); qr(1, 5, 9, 13); qr(2, 6, 10, 14); qr(3, 7, 11, 15)
        qr(0, 5, 10, 15); qr(1, 6, 11, 12); qr(2, 7, 8, 13); qr(3, 4, 9, 14)
    return [(x[i] + st[i]) & M for i in range(16)]


def salsa_core(inp, rounds):
    x = list(inp)

    def qr(a, b, c, d):
        x[b] ^= rotl((x[a] + x[d]) & M, 7)
        x[c] ^= rotl((x[b] + x[a]) & M, 9)
        x[d] ^= rotl((x[c] + x[b]) & M, 13)
        x[a] ^= rotl((x[d] + x[c]) & M, 18)

    for _ in range(rounds // 2):
        qr(0, 4, 8, 12); qr(5, 9, 13, 1); qr(10, 14, 2, 6); qr(15, 3, 7, 11)
        qr(0, 1, 2, 3); qr(5, 6, 7, 4); qr(10, 11, 8, 9); qr(15, 12, 13, 14)
    return [(x[i] + inp[i]) & M for i in range(16)]


def salsa_block(key, counter, nonce, rounds):
    k = list(key)
    inp = [SIGMA[0], *k[0:4], SIGMA[1], nonce[0], nonce[1], counter, 0, SIGMA[2], *k[4:8], SIGMA[3]]
    return salsa_core(inp, rounds)


# ---- task builders ---------------------------------------------------------------------------------------------------

def _truth(alg, extra_groups, direction=None):
    accept = [NAMES[alg]] + ([ENC if direction == "enc" else DEC] if direction else []) + extra_groups
    reject = reject_for(alg)
    if direction:
        reject += DEC if direction == "enc" else [w for w in ENC if w != "encryption"]   # "Tiny Encryption Algorithm" is not a direction
    return {"score": "tokens", "accept": accept, "reject": reject}


def v_rc4(rng):
    klen = rng.randint(5, 16)
    as_text = rng.random() < 0.5
    key = (rword(rng, klen).encode() if as_text else bytes(rng.randrange(1, 256) for _ in range(klen)))
    if as_text:
        decl, size = f'static const char key[] = "{key.decode()}";', f"(sizeof key - 1)"
        idx = f"(uint8_t)key[i % {size}]"
    else:
        decl, size = f"static const uint8_t key[] = {{{c_bytes(key)}}};", "sizeof key"
        idx = f"key[i % {size}]"
    body = f"""void proc(uint8_t *buf, size_t n) {{
    {decl}
    uint8_t S[256];
    unsigned i, j = 0;
    for (i = 0; i < 256; i++) S[i] = (uint8_t)i;
    for (i = 0; i < 256; i++) {{
        j = (j + S[i] + {idx}) & 255;
        uint8_t t = S[i]; S[i] = S[j]; S[j] = t;
    }}
    i = j = 0;
    for (size_t k = 0; k < n; k++) {{
        i = (i + 1) & 255;
        j = (j + S[i]) & 255;
        uint8_t t = S[i]; S[i] = S[j]; S[j] = t;
        buf[k] ^= S[(S[i] + S[j]) & 255];
    }}
}}"""
    body = with_decoys(body, rng, rng.choice([0, 0, 1, 2]))
    data = bytes(rng.randrange(256) for _ in range(rng.randint(10, 24)))
    main = f"""int main(void) {{
    uint8_t buf[] = {{{c_bytes(data)}}};
    proc(buf, sizeof buf);
    for (size_t i = 0; i < sizeof buf; i++) printf("%02x", buf[i]);
    printf("\\n");
    return 0;
}}"""
    q = "This function transforms a buffer in place. Which cipher is it, and how many bytes long is its key (decimal)?"
    return Task(kind=FAMILY, variant="rc4", body=body, question=q, truth=_truth("rc4", [dec(klen)]),
                params={"key": key, "keylen": klen}, wrong=f"RC4, key length {klen + 1}",
                check=(main, rc4(key, data).hex() + "\n"))


def _tea_task(rng, alg, direction):
    rounds = rng.choice([32, 32, 16, 20, 24, 48, 64])
    key = [rng.getrandbits(32) for _ in range(4)]
    v = [rng.getrandbits(32), rng.getrandbits(32)]
    ref = (tea if alg == "tea" else xtea)
    start = (DELTA * rounds) & M
    if alg == "tea" and direction == "enc":
        loop = f"""    uint32_t sum = 0;
    for (int i = 0; i < {rounds}; i++) {{
        sum += 0x9e3779b9u;
        v0 += ((v1 << 4) + k[0]) ^ (v1 + sum) ^ ((v1 >> 5) + k[1]);
        v1 += ((v0 << 4) + k[2]) ^ (v0 + sum) ^ ((v0 >> 5) + k[3]);
    }}"""
    elif alg == "tea":
        loop = f"""    uint32_t sum = 0x{start:08x}u;
    for (int i = 0; i < {rounds}; i++) {{
        v1 -= ((v0 << 4) + k[2]) ^ (v0 + sum) ^ ((v0 >> 5) + k[3]);
        v0 -= ((v1 << 4) + k[0]) ^ (v1 + sum) ^ ((v1 >> 5) + k[1]);
        sum -= 0x9e3779b9u;
    }}"""
    elif direction == "enc":
        loop = f"""    uint32_t sum = 0;
    for (int i = 0; i < {rounds}; i++) {{
        v0 += (((v1 << 4) ^ (v1 >> 5)) + v1) ^ (sum + k[sum & 3]);
        sum += 0x9e3779b9u;
        v1 += (((v0 << 4) ^ (v0 >> 5)) + v0) ^ (sum + k[(sum >> 11) & 3]);
    }}"""
    else:
        loop = f"""    uint32_t sum = 0x{start:08x}u;
    for (int i = 0; i < {rounds}; i++) {{
        v1 -= (((v0 << 4) ^ (v0 >> 5)) + v0) ^ (sum + k[(sum >> 11) & 3]);
        sum -= 0x9e3779b9u;
        v0 -= (((v1 << 4) ^ (v1 >> 5)) + v1) ^ (sum + k[sum & 3]);
    }}"""
    body = f"""void proc(uint32_t v[2]) {{
    static const uint32_t k[4] = {{{c_u32s(key)}}};
    uint32_t v0 = v[0], v1 = v[1];
{loop}
    v[0] = v0;
    v[1] = v1;
}}"""
    body = with_decoys(body, rng, rng.choice([0, 0, 1, 2]))
    main = f"""int main(void) {{
    uint32_t v[2] = {{0x{v[0]:08x}u, 0x{v[1]:08x}u}};
    proc(v);
    printf("%08x %08x\\n", (unsigned)v[0], (unsigned)v[1]);
    return 0;
}}"""
    out = ref(v, key, rounds, decrypt=(direction == "dec"))
    q = (f"This function transforms an 8-byte block (two 32-bit words) with a 128-bit key. Which cipher is it, does it encrypt or decrypt, "
         "and how many iterations does its main loop run (decimal)?")
    other = "xtea" if alg == "tea" else "tea"
    flip = "decrypt" if direction == "enc" else "encrypt"
    wrong = {("tea", "enc"): f"TEA, {flip}, {rounds}", ("tea", "dec"): f"TEA, decrypt, {rounds + 16}",
             ("xtea", "enc"): f"{other}, encrypt, {rounds}", ("xtea", "dec"): f"XTEA, {flip}, {rounds}"}[(alg, direction)]
    return Task(kind=FAMILY, variant=f"{alg}_{direction}", body=body, question=q,
                truth=_truth(alg, [dec(rounds)], direction), params={"key": key, "rounds": rounds, "v": v, "alg": alg},
                wrong=wrong, check=(main, f"{out[0]:08x} {out[1]:08x}\n"))


def _stream_task(rng, alg):
    rounds = rng.choice([20, 20, 12, 8] + list(range(4, 41, 2)))
    key = [rng.getrandbits(32) for _ in range(8)]
    counter = rng.getrandbits(32)
    nonce = [rng.getrandbits(32) for _ in range(3 if alg == "chacha" else 2)]
    style = rng.randrange(3)
    loop_head = [f"for (int i = 0; i < {rounds}; i += 2)", f"for (int i = 0; i < {rounds // 2}; i++)",
                 f"for (int i = {rounds}; i > 0; i -= 2)"][style]
    inline_sigma = rng.random() < 0.5
    sig = ("    static const uint32_t sigma[4] = {0x61707865u, 0x3320646eu, 0x79622d32u, 0x6b206574u};\n" if not inline_sigma else "")
    s = (lambda i: f"sigma[{i}]") if not inline_sigma else (lambda i: f"0x{SIGMA[i]:08x}u")
    if alg == "chacha":
        macro = """#define QR(a, b, c, d) \\
    a += b; d ^= a; d = (d << 16) | (d >> 16); \\
    c += d; b ^= c; b = (b << 12) | (b >> 20); \\
    a += b; d ^= a; d = (d << 8) | (d >> 24); \\
    c += d; b ^= c; b = (b << 7) | (b >> 25);

"""
        init = f"""    x[0] = {s(0)}; x[1] = {s(1)}; x[2] = {s(2)}; x[3] = {s(3)};
    for (int i = 0; i < 8; i++) x[4 + i] = key[i];
    x[12] = counter;
    for (int i = 0; i < 3; i++) x[13 + i] = nonce[i];"""
        rnd = """        QR(x[0], x[4], x[8], x[12])
        QR(x[1], x[5], x[9], x[13])
        QR(x[2], x[6], x[10], x[14])
        QR(x[3], x[7], x[11], x[15])
        QR(x[0], x[5], x[10], x[15])
        QR(x[1], x[6], x[11], x[12])
        QR(x[2], x[7], x[8], x[13])
        QR(x[3], x[4], x[9], x[14])"""
        nn = 3
        ref = chacha_block(key, counter, nonce, rounds)
    else:
        macro = """#define R(x, n) (((x) << (n)) | ((x) >> (32 - (n))))
#define QR(a, b, c, d) b ^= R(a + d, 7); c ^= R(b + a, 9); d ^= R(c + b, 13); a ^= R(d + c, 18);

"""
        init = f"""    x[0] = {s(0)}; x[5] = {s(1)}; x[10] = {s(2)}; x[15] = {s(3)};
    for (int i = 0; i < 4; i++) {{ x[1 + i] = key[i]; x[11 + i] = key[4 + i]; }}
    x[6] = nonce[0]; x[7] = nonce[1];
    x[8] = counter; x[9] = 0;"""
        rnd = """        QR(x[0], x[4], x[8], x[12])
        QR(x[5], x[9], x[13], x[1])
        QR(x[10], x[14], x[2], x[6])
        QR(x[15], x[3], x[7], x[11])
        QR(x[0], x[1], x[2], x[3])
        QR(x[5], x[6], x[7], x[4])
        QR(x[10], x[11], x[8], x[9])
        QR(x[15], x[12], x[13], x[14])"""
        nn = 2
        ref = salsa_block(key, counter, nonce, rounds)
    body = f"""{macro}void proc(uint32_t out[16], const uint32_t key[8], uint32_t counter, const uint32_t nonce[{nn}]) {{
    uint32_t x[16], w[16];
{sig}{init}
    memcpy(w, x, sizeof w);
    {loop_head} {{
{rnd}
    }}
    for (int i = 0; i < 16; i++) out[i] = x[i] + w[i];
}}"""
    body = with_decoys(body, rng, rng.choice([1, 1, 2, 2, 3]))
    main = f"""int main(void) {{
    static const uint32_t key[8] = {{{c_u32s(key)}}};
    static const uint32_t nonce[{nn}] = {{{c_u32s(nonce)}}};
    uint32_t out[16];
    proc(out, key, 0x{counter:08x}u, nonce);
    for (int i = 0; i < 16; i++) printf("%08x%c", (unsigned)out[i], i % 4 == 3 ? '\\n' : ' ');
    return 0;
}}"""
    expect = "".join(f"{w:08x}" + ("\n" if i % 4 == 3 else " ") for i, w in enumerate(ref))
    q = ("This function expands a 256-bit key, a counter and a nonce into a 64-byte block. Which cipher family is it, and how many rounds "
         "does it run in total (a double round counts as 2; decimal)?")
    wrong = f"salsa20, {rounds}" if alg == "chacha" else f"Salsa20, {rounds + 4}"
    return Task(kind=FAMILY, variant=alg, body=body, question=q, truth=_truth(alg, [dec(rounds)]),
                params={"key": key, "rounds": rounds, "counter": counter, "nonce": nonce, "style": style},
                wrong=wrong, check=(main, expect))


VARIANTS = {"rc4": v_rc4,
            "tea_enc": lambda rng: _tea_task(rng, "tea", "enc"), "tea_dec": lambda rng: _tea_task(rng, "tea", "dec"),
            "xtea_enc": lambda rng: _tea_task(rng, "xtea", "enc"), "xtea_dec": lambda rng: _tea_task(rng, "xtea", "dec"),
            "chacha": lambda rng: _stream_task(rng, "chacha"), "salsa": lambda rng: _stream_task(rng, "salsa")}
