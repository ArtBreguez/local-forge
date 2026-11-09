"""Structs and offsets: which byte offset does `proc` touch, how big is the record.

The truth comes from a Python model of the x86-64 SysV layout (size = alignment for scalars, padding before each field, tail
padding to the largest alignment). The harness checks that model against the compiler's own offsetof/sizeof and runs `proc`
on values it wrote itself, so a wrong layout, or a `proc` that reads another field, fails the test.
"""
from __future__ import annotations

from .core import HDR, Task, dec, hx

FAMILY = "structs"
HEADERS = HDR.rstrip("\n") + "\n#include <stddef.h>\n\n"

NAMES = ["id", "flags", "len", "count", "kind", "hash", "size", "mode", "tag", "value", "key", "ts", "crc", "ver", "seq", "cap",
         "mask", "prio", "state", "owner", "limit", "port", "uid", "gen"]
STRUCT_NAMES = ["rec", "hdr", "entry", "item", "pkt", "cfg", "ctx", "msg", "slot", "conn"]


class Scalar:
    def __init__(self, c, size, signed=False, ptr=False):
        self.c, self.size, self.align, self.signed, self.ptr = c, size, size, signed, ptr

    def decl(self, name):
        return f"{self.c} {name};" if not self.c.endswith("*") else f"{self.c}{name};"

    def lit(self, v):
        return f"0x{v:x}" + {"uint32_t": "u", "uint64_t": "ull"}.get(self.c, "")

    def draw(self, rng):
        """A distinctive non-zero value that fits the type and is not a tiny number."""
        hi = {1: 0xfe, 2: 0x7ffe, 4: 0x7ffffffe, 8: 0x7ffffffffffffffe}[self.size]
        lo = {1: 0x10, 2: 0x100, 4: 0x10000, 8: 0x100000000}[self.size]
        return rng.randrange(lo, hi)


class Arr:
    def __init__(self, elem, n):
        self.elem, self.n, self.size, self.align = elem, n, elem.size * n, elem.align

    def decl(self, name):
        return f"{self.elem.c} {name}[{self.n}];"


class Struct:
    def __init__(self, name, fields):
        off, al, self.offsets = 0, 1, {}
        for fname, ty in fields:
            off = -(-off // ty.align) * ty.align
            self.offsets[fname] = off
            off += ty.size
            al = max(al, ty.align)
        self.name, self.fields, self.align = name, fields, al
        self.size = -(-off // al) * al
        self.c = f"struct {name}"

    def decl(self, name):
        return f"{self.c} {name};"

    def definition(self):
        return f"struct {self.name} {{\n" + "".join(f"    {ty.decl(n)}\n" for n, ty in self.fields) + "};\n\n"

    def ty(self, fname):
        return dict(self.fields)[fname]


INTS = [Scalar("uint8_t", 1), Scalar("uint16_t", 2), Scalar("uint32_t", 4), Scalar("uint64_t", 8),
        Scalar("int16_t", 2, True), Scalar("int32_t", 4, True)]
PTR = Scalar("void *", 8, ptr=True)


def rand_fields(rng, names, arrays=True):
    out = []
    for n in names:
        r = rng.random()
        if arrays and r < 0.18:
            ty = Arr(Scalar("uint8_t", 1), rng.choice([3, 5, 6, 7, 12, 16]))
        elif r < 0.28:
            ty = PTR
        else:
            ty = rng.choice(INTS)
        out.append((n, ty))
    return out


def scalars(st):
    return [n for n, ty in st.fields if isinstance(ty, Scalar) and not ty.ptr]


def make(rng, nmin=4, nmax=8):
    names = rng.sample(NAMES, rng.randint(nmin, nmax))
    st = Struct(rng.choice(STRUCT_NAMES), rand_fields(rng, names))
    while len(scalars(st)) < 3:               # need a few integer fields to read and write
        st = Struct(st.name, rand_fields(rng, names))
    return st


def offq(extra=""):
    return f"At which byte offset (hex, like 0x1c) {extra}"


def _off_task(variant, st, body, question, off, params, check_main, expected):
    return Task(kind=FAMILY, variant=variant, headers=HEADERS, body=st.definition() + body, question=question,
                truth={"score": "tokens", "accept": [hx(off)]}, params={**params, "offset": off, "size": st.size},
                wrong=f"0x{off + 4:x}", check=(check_main, expected))


def _fill(rng, st, names, var="s"):
    """C statements that set the named scalar fields of `var` to random values; returns (code, {field: value})."""
    vals = {n: st.ty(n).draw(rng) for n in names}
    return "".join(f"    {var}.{n} = {st.ty(n).lit(v)};\n" for n, v in vals.items()), vals


def v_array_stride(rng):
    st = make(rng)
    f = rng.choice(scalars(st))
    body = f"""uint64_t proc(const struct {st.name} *a, size_t n) {{
    uint64_t s = 0;
    for (size_t i = 0; i < n; i++) s += (uint64_t)a[i].{f};
    return s;
}}"""
    vals = [rng.randrange(1, 200) for _ in range(3)]
    sets = "".join(f"    a[{i}].{f} = {v};\n" for i, v in enumerate(vals))
    main = f"""int main(void) {{
    struct {st.name} a[3];
    memset(a, 0, sizeof a);
{sets}    printf("%zu %zu\\n%llu\\n", sizeof(struct {st.name}), offsetof(struct {st.name}, {f}), (unsigned long long)proc(a, 3));
    return 0;
}}"""
    q = "proc walks an array of fixed-size records through its first argument. What is the size of one record in bytes (decimal)?"
    return Task(kind=FAMILY, variant="array_stride", headers=HEADERS, body=st.definition() + body, question=q,
                truth={"score": "tokens", "accept": [dec(st.size)]}, params={"size": st.size, "offset": st.offsets[f]},
                wrong=str(st.size + 8), check=(main, f"{st.size} {st.offsets[f]}\n{sum(vals)}\n"))


def v_field_read(rng):
    st = make(rng, 5, 9)
    g, f = rng.sample(scalars(st), 2)
    k = rng.randint(1, 90)
    body = f"""uint64_t proc(const struct {st.name} *p) {{
    if (p->{g} == 0) return 0;
    return (uint64_t)p->{f} + {k};
}}"""
    code, vals = _fill(rng, st, [f])
    main = f"""int main(void) {{
    struct {st.name} s;
    memset(&s, 0, sizeof s);
{code}    s.{g} = 1;
    printf("%zu\\n%llu\\n", offsetof(struct {st.name}, {f}), (unsigned long long)proc(&s));
    s.{g} = 0;
    printf("%llu\\n", (unsigned long long)proc(&s));
    return 0;
}}"""
    off = st.offsets[f]
    q = offq("of the structure is the value that proc returns? (proc reads a flag field, tests it against zero, then returns another field plus a constant.)")
    return _off_task("field_read", st, body, q, off, {"field": f}, main, f"{off}\n{vals[f] + k}\n0\n")


def v_init_magic(rng):
    st = make(rng, 5, 9)
    names = rng.sample(scalars(st), rng.randint(3, min(5, len(scalars(st)))))
    seen, vals = set(), {}
    for n in names:                             # distinct constants, so the question has one answer
        v = st.ty(n).draw(rng)
        while v in seen:
            v = st.ty(n).draw(rng)
        seen.add(v)
        vals[n] = v
    target = rng.choice(names)
    stores = "".join(f"    p->{n} = {st.ty(n).lit(vals[n])};\n" for n in names)
    body = f"void proc(struct {st.name} *p) {{\n{stores}}}"
    ty = st.ty(target)
    main = f"""int main(void) {{
    struct {st.name} s;
    unsigned long long v = 0;
    memset(&s, 0, sizeof s);
    proc(&s);
    memcpy(&v, (const char *)&s + {st.offsets[target]}, {ty.size});
    printf("%zu\\n%llu\\n", offsetof(struct {st.name}, {target}), v);
    return 0;
}}"""
    off = st.offsets[target]
    q = offq(f"of the structure does proc store the constant 0x{vals[target]:x}?")
    return _off_task("init_magic", st, body, q, off, {"field": target}, main, f"{off}\n{vals[target]}\n")


def v_alloc_size(rng):
    st = make(rng)
    f = rng.choice(scalars(st))
    form = rng.choice(["calloc1", "malloc_memset", "calloc_n"])
    if form == "calloc1":
        body = f"""struct {st.name} *proc(uint32_t id) {{
    struct {st.name} *p = calloc(1, sizeof(struct {st.name}));
    if (!p) return NULL;
    p->{f} = id;
    return p;
}}"""
    elif form == "malloc_memset":
        body = f"""struct {st.name} *proc(uint32_t id) {{
    struct {st.name} *p = malloc(sizeof *p);
    if (!p) return NULL;
    memset(p, 0, sizeof *p);
    p->{f} = id;
    return p;
}}"""
    else:
        body = f"""struct {st.name} *proc(uint32_t id) {{
    struct {st.name} *p = calloc(4, sizeof(struct {st.name}));
    if (!p) return NULL;
    for (int i = 0; i < 4; i++) p[i].{f} = id + (uint32_t)i;
    return p;
}}"""
    main = f"""int main(void) {{
    struct {st.name} *p = proc(77);
    printf("%zu\\n%llu\\n", sizeof(struct {st.name}), p ? (unsigned long long)p->{f} : 0ull);
    free(p);
    return 0;
}}"""
    q = ("proc allocates memory for a structure (or, if it allocates several, an array of identical structures). "
         "What is the size in bytes (decimal) of one structure?")
    return Task(kind=FAMILY, variant="alloc_size", headers=HEADERS, body=st.definition() + body, question=q,
                truth={"score": "tokens", "accept": [dec(st.size)]}, params={"size": st.size, "form": form},
                wrong=str(st.size + 8), check=(main, f"{st.size}\n77\n"))


def v_nested_offset(rng):
    names = rng.sample(NAMES, 8)
    inner = Struct("sub" + rng.choice("abc"), rand_fields(rng, names[:rng.randint(3, 4)], arrays=False))
    while len(scalars(inner)) < 2:
        inner = Struct(inner.name, rand_fields(rng, names[:3], arrays=False))
    pre = rand_fields(rng, names[4:4 + rng.randint(1, 3)])
    post = rand_fields(rng, names[7:7 + rng.randint(0, 1)])
    outer = Struct(rng.choice(STRUCT_NAMES), pre + [("sub", inner)] + post)
    f = rng.choice(scalars(inner))
    off = outer.offsets["sub"] + inner.offsets[f]
    body = f"""uint64_t proc(const struct {outer.name} *o) {{
    return (uint64_t)o->sub.{f};
}}"""
    val = inner.ty(f).draw(rng)
    main = f"""int main(void) {{
    struct {outer.name} s;
    memset(&s, 0, sizeof s);
    s.sub.{f} = {inner.ty(f).lit(val)};
    printf("%zu\\n%llu\\n", offsetof(struct {outer.name}, sub.{f}), (unsigned long long)proc(&s));
    return 0;
}}"""
    q = offq("of the outer structure is the value that proc returns? (the field lives inside a nested structure; count from the start of the outer one.)")
    t = Task(kind=FAMILY, variant="nested_offset", headers=HEADERS, body=inner.definition() + outer.definition() + body, question=q,
             truth={"score": "tokens", "accept": [hx(off)]}, params={"offset": off, "size": outer.size, "field": f},
             wrong=f"0x{inner.offsets[f]:x}" if inner.offsets[f] != off else f"0x{off + 4:x}", check=(main, f"{off}\n{val}\n"))
    return t


def v_linked_next(rng):
    names = rng.sample(NAMES, rng.randint(4, 7))
    nf, vf = names[0], names[1]                       # the next pointer and the value that is summed
    st_name = rng.choice(STRUCT_NAMES)
    fields = rand_fields(rng, names[2:]) + [(nf, Scalar(f"struct {st_name} *", 8, ptr=True)), (vf, rng.choice(INTS))]
    rng.shuffle(fields)
    st = Struct(st_name, fields)
    body = f"""uint64_t proc(const struct {st.name} *n) {{
    uint64_t s = 0;
    while (n) {{
        s += (uint64_t)n->{vf};
        n = n->{nf};
    }}
    return s;
}}"""
    vals = [rng.randrange(1, 100) for _ in range(3)]
    main = f"""int main(void) {{
    struct {st.name} a[3];
    memset(a, 0, sizeof a);
    a[0].{vf} = {vals[0]}; a[0].{nf} = &a[1];
    a[1].{vf} = {vals[1]}; a[1].{nf} = &a[2];
    a[2].{vf} = {vals[2]};
    printf("%zu\\n%llu\\n", offsetof(struct {st.name}, {nf}), (unsigned long long)proc(a));
    return 0;
}}"""
    off = st.offsets[nf]
    q = offq("of the node is the pointer that proc follows to reach the next node of the list?")
    return _off_task("linked_next", st, body, q, off, {"field": nf}, main, f"{off}\n{sum(vals)}\n")


VARIANTS = {"array_stride": v_array_stride, "field_read": v_field_read, "init_magic": v_init_magic,
            "alloc_size": v_alloc_size, "nested_offset": v_nested_offset, "linked_next": v_linked_next}
