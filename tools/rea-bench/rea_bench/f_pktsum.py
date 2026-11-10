"""Checksums over packets: which byte range is covered, how many trailing bytes are excluded, little or big endian.

`proc(p, n)` validates a packet: a header of H bytes starting with a magic byte, a payload, then a 1- or 2-byte checksum. The
checksum covers [S, n - W) with 1 <= S <= H, so the header bytes before S are not protected, or (variant len_field) the `len`
bytes that follow the header, where `len` is a byte at offset L. The truth is the generator's own S, W, L and endianness. The
harness builds valid and mutated packets and prints proc's verdict (1 valid, 0 bad checksum, -1 malformed), expected from a
Python model of the same format: flipping a covered byte must fail, flipping an uncovered header byte must not, swapping the two
stored bytes must fail (so the byte order is proven), a wrong magic or a truncated packet is malformed.
"""
from __future__ import annotations

from .core import Task, dec, with_decoys
from .f_crc import crc_ref

FAMILY = "pktsum"
ARC_LIT = 0xA001            # reflected CRC-16/ARC, as it appears in the code (normal form 0x8005)


def csum(kind: str, data: bytes) -> int:
    if kind == "sum8":
        return sum(data) & 0xFF
    if kind == "xor8":
        x = 0
        for b in data:
            x ^= b
        return x
    if kind == "sum16":
        return sum(data) & 0xFFFF
    if kind == "fletcher16":
        c0 = c1 = 0
        for b in data:
            c0 = (c0 + b) % 255
            c1 = (c1 + c0) % 255
        return (c1 << 8) | c0
    if kind == "crc16":
        return crc_ref(data, 16, 0x8005, 0, True, 0)
    raise ValueError(kind)


def c_checksum(kind: str, lo: str, hi: str) -> tuple:
    """(declarations + loop computing `got_sum` over p[lo..hi), C type of the sum)."""
    if kind == "sum8":
        return f"    uint8_t sum = 0;\n    for (size_t i = {lo}; i < {hi}; i++) sum = (uint8_t)(sum + p[i]);\n"
    if kind == "xor8":
        return f"    uint8_t sum = 0;\n    for (size_t i = {lo}; i < {hi}; i++) sum = (uint8_t)(sum ^ p[i]);\n"
    if kind == "sum16":
        return f"    uint16_t sum = 0;\n    for (size_t i = {lo}; i < {hi}; i++) sum = (uint16_t)(sum + p[i]);\n"
    if kind == "fletcher16":
        return (f"    unsigned c0 = 0, c1 = 0;\n    for (size_t i = {lo}; i < {hi}; i++) {{\n        c0 = (c0 + p[i]) % 255;\n"
                f"        c1 = (c1 + c0) % 255;\n    }}\n    uint16_t sum = (uint16_t)((c1 << 8) | c0);\n")
    return (f"    uint16_t sum = 0;\n    for (size_t i = {lo}; i < {hi}; i++) {{\n        sum ^= p[i];\n"
            f"        for (int k = 0; k < 8; k++) sum = (uint16_t)((sum & 1) ? ((sum >> 1) ^ 0x{ARC_LIT:04x}u) : (sum >> 1));\n    }}\n")


WIDTH = {"sum8": 1, "xor8": 1, "sum16": 2, "fletcher16": 2, "crc16": 2}


class Spec:
    def __init__(self, rng, kind, endian=None, len_field=False):
        self.kind, self.W, self.endian, self.len_field = kind, WIDTH[kind], endian, len_field
        self.H = rng.randint(3, 6) if len_field else rng.randint(2, 6)
        self.S = self.H if len_field else rng.randint(1, self.H)
        self.L = rng.randint(1, self.H - 1) if len_field else None
        self.magic = rng.randint(0x10, 0xEE)

    def covered(self, n, pkt):
        if self.len_field:
            ln = pkt[self.L]
            return self.H, self.H + ln
        return self.S, n - self.W

    def stored(self, value: int) -> bytes:
        return value.to_bytes(self.W, self.endian or "little")

    def verdict(self, pkt: bytes) -> int:
        """Python model of what proc returns."""
        n = len(pkt)
        if n < self.H + self.W or pkt[0] != self.magic:
            return -1
        if self.len_field and self.H + pkt[self.L] + 1 > n:
            return -1
        lo, hi = self.covered(n, pkt)
        got = int.from_bytes(pkt[hi:hi + self.W] if self.len_field else pkt[n - self.W:], self.endian or "little")
        return int(csum(self.kind, pkt[lo:hi]) == got)

    def build(self, rng) -> bytes:
        while True:
            m = rng.randint(4, 12)
            head = bytearray(rng.randrange(256) for _ in range(self.H))
            head[0] = self.magic
            if self.len_field:
                head[self.L] = m
            pay = bytes(rng.randrange(256) for _ in range(m if self.len_field else m))
            body = bytes(head) + pay
            lo, hi = (self.H, self.H + m) if self.len_field else (self.S, len(body))
            v = csum(self.kind, body[lo:hi])
            st = self.stored(v)
            if self.W == 2 and st[0] == st[1]:
                continue                      # equal bytes would hide the byte order
            return body + st

    def packets(self, rng) -> list:
        good = self.build(rng)
        n = len(good)
        lo, hi = self.covered(n, good)
        out = [good]

        def flip(i):
            b = bytearray(good)
            b[i] ^= 0x5A
            return bytes(b)

        out.append(flip(lo))                                  # covered: must fail
        out.append(flip(hi - 1))
        spare = [i for i in range(1, self.H) if i < lo and i != self.L]
        if spare:
            out.append(flip(rng.choice(spare)))               # uncovered header byte: must still pass
        if self.W == 2:
            out.append(good[:-2] + good[-1:] + good[-2:-1])   # stored bytes swapped
        out.append(bytes([good[0] ^ 0x01]) + good[1:])        # wrong magic
        out.append(good[:self.H])                             # truncated
        return out


def _main(spec, pkts):
    decls = "".join(f"    static const uint8_t pk{i}[] = {{{', '.join(f'0x{b:02x}' for b in p)}}};\n" for i, p in enumerate(pkts))
    calls = "".join(f'    printf("%d\\n", proc(pk{i}, sizeof pk{i}));\n' for i in range(len(pkts)))
    return f"int main(void) {{\n{decls}{calls}    return 0;\n}}"


def _stored_c(spec):
    if spec.W == 1:
        return "p[n - 1]"
    return "(uint16_t)(p[n - 2] | (p[n - 1] << 8))" if spec.endian == "little" else "(uint16_t)((p[n - 2] << 8) | p[n - 1])"


def _flat_body(spec):
    loop = c_checksum(spec.kind, str(spec.S), f"n - {spec.W}")
    return f"""int proc(const uint8_t *p, size_t n) {{
    if (n < {spec.H + spec.W} || p[0] != 0x{spec.magic:02x}) return -1;
{loop}    return sum == {_stored_c(spec)};
}}"""


def _len_body(spec):
    loop = c_checksum(spec.kind, str(spec.H), f"{spec.H} + len")
    return f"""int proc(const uint8_t *p, size_t n) {{
    if (n < {spec.H + 1} || p[0] != 0x{spec.magic:02x}) return -1;
    size_t len = p[{spec.L}];
    if ({spec.H} + len + 1 > n) return -1;
{loop}    return sum == p[{spec.H} + len];
}}"""


QUESTIONS = {
    "start": "proc validates a packet that ends with a checksum. At which byte offset (decimal, counting from 0) does the checksummed range start?",
    "tail": "proc validates a packet that ends with a checksum. How many bytes at the end of the packet are not part of the checksummed range (decimal)?",
    "endian": "proc validates a packet that ends with a 16-bit checksum. Is the checksum stored little-endian or big-endian in the packet? Answer 'little' or 'big'.",
    "len_offset": "proc validates a packet with a length byte in its header; the checksum follows the payload. At which byte offset (decimal, counting from 0) is that length byte?",
    "len_start": "proc validates a packet with a length byte in its header; the checksum follows the payload. At which byte offset (decimal, counting from 0) does the checksummed range start?",
}


def _make(rng, variant, kind, endian=None, len_field=False):
    spec = Spec(rng, kind, endian, len_field)
    body = with_decoys(_len_body(spec) if len_field else _flat_body(spec), rng, rng.choice([1, 1, 2]), {spec.magic})   # core.decoys
    asks = ["len_offset", "len_start"] if len_field else ["start", "tail"] + (["endian"] if spec.W == 2 else [])
    ask = rng.choice(asks)
    if ask == "endian":
        word, other = ("little", "big") if spec.endian == "little" else ("big", "little")
        truth = {"score": "tokens", "accept": [[word]], "reject": [other]}
        wrong = other
    else:
        val = {"start": spec.S, "tail": spec.W, "len_offset": spec.L, "len_start": spec.H}[ask]
        truth = {"score": "tokens", "accept": [dec(val)]}
        wrong = str(val + 1)
    pkts = spec.packets(rng)
    expect = "".join(f"{spec.verdict(p)}\n" for p in pkts)
    return Task(kind=FAMILY, variant=variant, body=body, question=QUESTIONS[ask], truth=truth,
                params={"H": spec.H, "S": spec.S, "W": spec.W, "L": spec.L, "endian": spec.endian, "kind": kind, "ask": ask, "magic": spec.magic},
                wrong=wrong, check=(_main(spec, pkts), expect))


def _fixed(variant, kind, endian=None, len_field=False):
    return lambda rng: _make(rng, variant, kind, endian, len_field)


def v_fletcher16(rng):
    return _make(rng, "fletcher16", "fletcher16", rng.choice(["little", "big"]))


def v_crc16(rng):
    return _make(rng, "crc16", "crc16", rng.choice(["little", "big"]))


def v_len_field(rng):
    return _make(rng, "len_field", rng.choice(["sum8", "xor8"]), None, True)


VARIANTS = {"sum8": _fixed("sum8", "sum8"), "xor8": _fixed("xor8", "xor8"), "sum16_le": _fixed("sum16_le", "sum16", "little"),
            "sum16_be": _fixed("sum16_be", "sum16", "big"), "fletcher16": v_fletcher16, "crc16": v_crc16, "len_field": v_len_field}
