"""CRC-16/32, standard and random polynomials, reflected (LSB first, shifts right) or normal (MSB first, shifts left).

Two code forms: bitwise (8 shifts per byte, two spellings of the inner step) and a table built at run time on the first call.
Questions ask for one thing each: the polynomial constant exactly as written in the code plus the direction, the initial
register value, or the final XOR value. The truth is the generator's own parameters; the harness runs proc on three inputs and
compares with `crc_ref`, a Python implementation of the Rocksoft model (tests also anchor it to zlib and binascii).
"""
from __future__ import annotations

from .core import Task, hx, pool, with_decoys

FAMILY = "crc"

# name, width, polynomial (normal form), init register, reflected, xorout
STANDARD = [
    ("crc32", 32, 0x04C11DB7, 0xFFFFFFFF, True, 0xFFFFFFFF),
    ("crc32c", 32, 0x1EDC6F41, 0xFFFFFFFF, True, 0xFFFFFFFF),
    ("crc32-mpeg2", 32, 0x04C11DB7, 0xFFFFFFFF, False, 0x00000000),
    ("crc32-bzip2", 32, 0x04C11DB7, 0xFFFFFFFF, False, 0xFFFFFFFF),
    ("crc16-arc", 16, 0x8005, 0x0000, True, 0x0000),
    ("crc16-modbus", 16, 0x8005, 0xFFFF, True, 0x0000),
    ("crc16-usb", 16, 0x8005, 0xFFFF, True, 0xFFFF),
    ("crc16-kermit", 16, 0x1021, 0x0000, True, 0x0000),
    ("crc16-ccitt-false", 16, 0x1021, 0xFFFF, False, 0x0000),
    ("crc16-xmodem", 16, 0x1021, 0x0000, False, 0x0000),
    ("crc32-jamcrc", 32, 0x04C11DB7, 0xFFFFFFFF, True, 0x00000000),
    ("crc32-posix", 32, 0x04C11DB7, 0x00000000, False, 0xFFFFFFFF),
    ("crc32d", 32, 0xA833982B, 0xFFFFFFFF, True, 0xFFFFFFFF),
    ("crc32q", 32, 0x814141AB, 0x00000000, False, 0x00000000),
    ("crc32-xfer", 32, 0x000000AF, 0x00000000, False, 0x00000000),
    ("crc16-genibus", 16, 0x1021, 0xFFFF, False, 0xFFFF),
    ("crc16-mcrf4xx", 16, 0x1021, 0xFFFF, True, 0x0000),
    ("crc16-maxim", 16, 0x8005, 0x0000, True, 0xFFFF),
    ("crc16-t10-dif", 16, 0x8BB7, 0x0000, False, 0x0000),
    ("crc16-cdma2000", 16, 0xC867, 0xFFFF, False, 0x0000),
    ("crc16-dnp", 16, 0x3D65, 0x0000, True, 0xFFFF),
    ("crc16-teledisk", 16, 0xA097, 0x0000, False, 0x0000),
]


# Algorithms only the held-out seed draws (the program of a standard algorithm is fixed by its parameters: without a reserved part a
# held-out task would have a training twin that differs only in its decoys).
HELD_STANDARD = [a for a in STANDARD if a[0] in ("crc32-bzip2", "crc16-usb", "crc16-maxim", "crc16-genibus", "crc32q", "crc16-dnp", "crc16-teledisk")]


def reflect(v: int, width: int) -> int:
    return int(format(v, f"0{width}b")[::-1], 2)


def crc_ref(data: bytes, width: int, poly: int, init: int, refl: bool, xorout: int) -> int:
    """Rocksoft model with `init` as the register's starting value (as the C writes it) and poly in normal form."""
    mask = (1 << width) - 1
    crc = init
    if refl:
        rpoly = reflect(poly, width)
        for b in data:
            crc ^= b
            for _ in range(8):
                crc = (crc >> 1) ^ (rpoly if crc & 1 else 0)
    else:
        top = 1 << (width - 1)
        for b in data:
            crc ^= b << (width - 8)
            for _ in range(8):
                crc = ((crc << 1) ^ poly) & mask if crc & top else (crc << 1) & mask
    return (crc ^ xorout) & mask


def draw_params(rng, need_init=False, need_xor=False):
    """(width, poly, init, refl, xorout, label); half standard algorithms, half random polynomials."""
    for _ in range(50):
        if rng.random() < 0.5:
            label, width, poly, init, refl, xorout = rng.choice(pool(rng, STANDARD, HELD_STANDARD))
        else:
            width = rng.choice([16, 32])
            mask = (1 << width) - 1
            poly = rng.getrandbits(width) | 1
            init = rng.choice([0, mask, rng.getrandbits(width)])
            xorout = rng.choice([0, mask, rng.getrandbits(width)])
            refl = rng.random() < 0.5
            label = "custom"
        if (need_init and init == 0) or (need_xor and xorout == 0):
            continue
        lit = reflect(poly, width) if refl else poly
        if lit in (init, xorout):          # the probe finds the poly literal by its text: it must be the only one of its value
            continue
        return width, poly, init, refl, xorout, label
    return width, poly, init, refl, xorout, label


def source(width, poly, init, refl, xorout, form, step):
    T = f"uint{width}_t"
    d = width // 4
    lit = lambda v: f"0x{v:0{d}x}u"
    poly_lit = lit(reflect(poly, width) if refl else poly)
    top = lit(1 << (width - 1))
    sh = width - 8
    if form == "bitwise":
        if refl:
            feed = "crc ^= p[i];"
            inner = [f"crc = ({T})((crc >> 1) ^ ({poly_lit} & (0u - (crc & 1u))));",
                     f"if (crc & 1u) crc = ({T})((crc >> 1) ^ {poly_lit}); else crc = ({T})(crc >> 1);",
                     f"crc = ({T})((crc >> 1) ^ ((crc & 1u) ? {poly_lit} : 0u));",
                     f"{{ unsigned lsb = crc & 1u; crc = ({T})(crc >> 1); if (lsb) crc = ({T})(crc ^ {poly_lit}); }}"][step]
        else:
            feed = f"crc ^= ({T})((unsigned)p[i] << {sh});"
            inner = [f"crc = ({T})((crc & {top}) ? ((crc << 1) ^ {poly_lit}) : (crc << 1));",
                     f"if (crc & {top}) crc = ({T})((crc << 1) ^ {poly_lit}); else crc = ({T})(crc << 1);",
                     f"crc = ({T})((crc << 1) ^ ((crc & {top}) ? {poly_lit} : 0u));",
                     f"{{ int msb = (crc & {top}) != 0; crc = ({T})(crc << 1); if (msb) crc = ({T})(crc ^ {poly_lit}); }}"][step]
        return f"""{T} proc(const uint8_t *p, size_t n) {{
    {T} crc = {lit(init)};
    for (size_t i = 0; i < n; i++) {{
        {feed}
        for (int k = 0; k < 8; k++) {inner}
    }}
    return ({T})(crc ^ {lit(xorout)});
}}"""
    if refl:
        seed_c = "(uint32_t)i"
        step_c = (f"c = ({T})((c & 1u) ? ((c >> 1) ^ {poly_lit}) : (c >> 1));" if step % 2 == 0 else
                  f"if (c & 1u) c = ({T})((c >> 1) ^ {poly_lit}); else c = ({T})(c >> 1);")
        update = f"crc = ({T})(table[(crc ^ p[i]) & 0xffu] ^ (crc >> 8));"
    else:
        seed_c = f"(uint32_t)i << {sh}"
        step_c = (f"c = ({T})((c & {top}) ? ((c << 1) ^ {poly_lit}) : (c << 1));" if step % 2 == 0 else
                  f"if (c & {top}) c = ({T})((c << 1) ^ {poly_lit}); else c = ({T})(c << 1);")
        update = f"crc = ({T})(table[((crc >> {sh}) ^ p[i]) & 0xffu] ^ (crc << 8));"
    return f"""static {T} table[256];
static int ready;

{T} proc(const uint8_t *p, size_t n) {{
    if (!ready) {{
        for (uint32_t i = 0; i < 256; i++) {{
            {T} c = ({T})({seed_c});
            for (int k = 0; k < 8; k++) {step_c}
            table[i] = c;
        }}
        ready = 1;
    }}
    {T} crc = {lit(init)};
    for (size_t i = 0; i < n; i++) {update}
    return ({T})(crc ^ {lit(xorout)});
}}"""


def harness(width, params, rng):
    data = bytes(rng.randrange(256) for _ in range(rng.randint(12, 24)))
    vecs = [b"123456789", data, b""]
    arr = ", ".join(f"0x{b:02x}" for b in data)
    d = width // 4
    main = f"""int main(void) {{
    static const uint8_t v0[] = "123456789";
    static const uint8_t v1[] = {{{arr}}};
    printf("%0{d}x\\n", (unsigned)proc(v0, 9));
    printf("%0{d}x\\n", (unsigned)proc(v1, sizeof v1));
    printf("%0{d}x\\n", (unsigned)proc(v0, 0));
    printf("%0{d}x\\n", (unsigned)proc(v0, 9));
    return 0;
}}"""
    outs = [crc_ref(v, width, *params) for v in vecs] + [crc_ref(vecs[0], width, *params)]
    return main, "".join(f"{o:0{d}x}\n" for o in outs)


def _make(rng, form, ask):
    width, poly, init, refl, xorout, label = draw_params(rng, need_init=(ask == "init"), need_xor=(ask == "xorout"))
    step = rng.randrange(4)
    body = source(width, poly, init, refl, xorout, form, step)
    d = width // 4
    lit = reflect(poly, width) if refl else poly
    # standard algorithms are the same few programs in every task: decoys (core.decoys) make them differ; random polynomials need fewer
    ones = (1 << width) - 1
    body = with_decoys(body, rng, rng.choice([1, 1, 2, 2, 3]) if label != "custom" else rng.choice([0, 1, 1, 2]),
                       {poly, lit, init, xorout, ones, 0xffffffff, 0xffffffffffffffff, 0x80, 0xff, 1 << (width - 1)})
    params = {"width": width, "poly": poly, "lit": lit, "init": init, "refl": refl, "xorout": xorout, "label": label}
    if ask == "poly":
        q = (f"This function computes a {width}-bit CRC. Give the polynomial constant exactly as it appears in the code (hex), and say "
             "whether the algorithm is 'reflected' (shifts right, LSB first) or 'normal' (shifts left, MSB first).")
        word, other = ("reflected", "normal") if refl else ("normal", "reflected")
        truth = {"score": "tokens", "accept": [hx(lit, d), [word]], "reject": [other]}
        wrong = f"0x{lit:0{d}x} {other}"
    elif ask == "init":
        q = (f"This function computes a {width}-bit CRC. What is the initial value of the CRC register (hex, written at the CRC's width: "
             f"an all-ones value is 0x{(1 << width) - 1:x} even where the code loads a wider -1)?")
        truth = {"score": "tokens", "accept": [hx(init, d)]}
        wrong = f"0x{init ^ 0x1:0{d}x}"
    else:
        q = (f"This function computes a {width}-bit CRC. What value is XORed into the register after the last byte (hex, written at the CRC's "
             f"width: an all-ones value is 0x{(1 << width) - 1:x})? Answer 0 if nothing is XORed.")
        truth = {"score": "tokens", "accept": [hx(xorout, d)]}
        wrong = f"0x{xorout ^ 0x1:0{d}x}"
    asked = {"poly": lit, "init": init, "xorout": xorout}[ask]
    if asked == (1 << width) - 1:                    # all ones is loaded as a wider -1 (`mov edx, 0xffffffff`) in a 16-bit register
        asked = [asked, 0xffffffff, 0xffffffffffffffff]
    # the probe finds the asked literal in the source by its text and replaces it by a marker (survive.py); a polynomial also fails when
    # the compiler turned the bit loop into a 256-entry table (clang -O1: no polynomial in the code, a table in the data)
    at = {"poly": rf"(0x{lit:0{d}x})u", "init": r"uint\d+_t crc = (0x[0-9a-f]+)u;", "xorout": r"return \(uint\d+_t\)\(crc \^ (0x[0-9a-f]+)u\)"}[ask]
    probe = {"at": at, "kind": "exact", "hex": True, "bits": width, "no_table": ask == "poly"}
    return Task(kind=FAMILY, variant=f"{form}_{ask}", body=body, question=q, truth=truth, params=params, wrong=wrong,
                check=harness(width, (poly, init, refl, xorout), rng), facts=[(asked if isinstance(asked, list) else [asked]) + [probe]])


def _variant(form, ask):
    return lambda rng: _make(rng, form, ask)


VARIANTS = {f"{form}_{ask}": _variant(form, ask) for form in ("bitwise", "table") for ask in ("poly", "init", "xorout")}
