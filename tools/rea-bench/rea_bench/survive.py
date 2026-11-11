"""The "fact survives" gate: is the thing a task asks for still in the compiled code?

A task's `facts` are groups of alternatives; every group must hold. An int must appear as a number in the `objdump -d` of the
compiled functions (every function but main: proc, its helpers, anything the compiler split off), compared by magnitude and by the
two's-complement pattern of its negation (an offset prints as `[rdi+0x18]`, -500 as `+0x1f4`, `x - 0xa001` as `add edi, 0xffff5fff`;
core.forms). A truncation of the value (its low byte, a 16-bit part of a 32-bit constant) is NOT the value: round 2 found `0xa001`
"present" because the code had `cmp rsi, 1`. A string must appear in the object file (read-only data, or as immediates). The check runs
at the task's own compiler and -O level.

Plain presence is weak evidence for a value below 0x20 (it is in most programs: `sub rsp, 8`, `cmp rsi, 1`, a shift count): it can
fail a task (nothing of the group is in the code) but proves nothing when it holds, so the value alone never decides. What carries
the weight there is a causal probe: a group may end with a dict
    {"at": regex with one group around the asked literal in the C source, "kind": "exact" | "near" | "half", "hex": bool, "bits": int,
     "no_table": bool}
or with {"no_stack": true} alone (round 9, m3: the asked value is no literal of the source to probe, `sizeof` of a struct, and a frame offset must not count as its
presence: `[rbp-0x18]` made the record size 24 "present" in a gcc -O0 loop that scales the index with `lea` and `shl`)
and then the gate (1) finds that literal in the source - it must be one of the group's own values, so a source in which the asked
constant was changed is not the task's fact; (2) compiles the source again with the literal replaced by a marker value that is nowhere in
the first compile, and requires the marker (or, for "near", a neighbour; for "half", its half or double: a loop of N double rounds) to show
up as a number in the code. A fact the compiler folded away (TEA's counter into `sum`, a CRC polynomial into a vector table) does not
show its marker. `no_table` also fails a CRC polynomial when the object holds 256 bytes or more of read-only data that is not a string: a
table the compiler made from the bit loop, not in the source.

String obfuscation (round 5, M1) has no constant to ask for: the answer is the string the code writes. Its tasks carry an `object_gate`
instead of `facts`: {"hidden": plaintext, "stored": hex of the stored bytes, "keys": [decode constants]}. The object compiled at the task's own
cc/opt must NOT hold the plaintext (a 4-byte window of it with at least 3 letters, searched in the code and data sections: the compiler
decoded the string at compile time and no decoding is left to read) and must hold the stored bytes (a 4-byte window of them). Round 6, m6: it used to
accept a decode constant among the immediates of the disassembly instead; a small key, rot_n's shift and the plaintext letters themselves (immediates when a
compiler stores a string byte by byte) are in almost every object, and over 3,600 objects (seed 99, 60 tasks per variant, every cc and -O) the clause decided
none (1,998 accepted on the stored bytes, 1,602 refused for the plaintext, 0 on a key alone), so it is gone; "keys" stays in the gate as a record. The
sections are read from the ELF object itself (not symbols, not debug info: clang's UBSan strings and the build path are no part of the code).

This is a necessary condition, not proof that the decompiler shows it. What it removes is the tasks the reviewers found unanswerable:
TEA's round count (the loop counter becomes a compare against DELTA*rounds), a CRC polynomial in a vectorised or table-ised loop, a
character class that became one bit-test against a 42-bit mask.

Round 9, B1: a table the answer is read from must be SHOWN, not only present. The model sees the bytes `refdata` prints at the data references of `proc`
(64 forward from each address, visible.py), so a `static const` array of which a byte lies before every reference, or past the first window, cannot be read:
gcc compiled `enc[n - 1 - i]` to a pointer that runs down from the last element, and 15 of 16 encoded bytes of a strobf/reverse_xor task were never printed. `check_visible`
links the task as the sample stage builds it and asks, for every `static const` array of the body that the binary keeps in .rodata, that every byte is inside a window;
every family with such an array is gated by it (strobf, cipher, encoding, fsm tables, bounds/index_limit, multi/decode_secret, the frozen legacy strings), a draw that fails is
drawn again (a frozen legacy task is dropped: families.untrainable).

A "derived form" (hi - lo for a range tested as `v - lo <= width`, N/2 for a loop over double rounds) is a listed alternative of
its group, and the question of that variant says the compiler may show it that way.

    python3 tools/rea-bench survival --family all --n 120 --seed 5       survival rate per variant, gcc and clang, every -O level
"""
from __future__ import annotations

import pathlib
import re
import shutil
import struct
import subprocess
import sys
import tempfile

from . import visible
from .core import OPTS, forms

OBJDUMP = shutil.which("objdump")
_HEX = re.compile(r"(?<![\w.])-?0x([0-9a-f]+)\b")
_DEC = re.compile(r"(?<![\w.])(\d+)(?![\w.])")
_FUNC = re.compile(r"^[0-9a-f]+ <([^>]+)>:$")


def toolchain(cc: str):
    """Path of the compiler or None."""
    return shutil.which(cc)


def compile_object(src: str, cc: str, opt: str, out: pathlib.Path) -> subprocess.CompletedProcess:
    return subprocess.run([cc, opt, "-fno-lto", "-w", "-c", "-o", str(out), "-x", "c", "-"], input=src, capture_output=True, text=True, timeout=60)


_STACK = re.compile(r"\b[re]?[sb]p\b")


def numbers(disasm: str, stack: bool = True) -> set:
    """Magnitudes of every number in the instructions of the disassembly (functions other than main). `stack=False` leaves out the instructions that name the stack or frame
    pointer (`[rbp-0x30]`, `sub rsp, 0x30`): a frame offset is not a constant of the program (round 9, m3)."""
    out, keep = set(), False
    for line in disasm.splitlines():
        m = _FUNC.match(line.strip())
        if m:
            keep = m.group(1) != "main"
            continue
        if not keep or "\t" not in line:
            continue
        insn = re.sub(r"#.*|<[^>]*>", "", line.split("\t", 1)[1])
        if not stack and _STACK.search(insn):
            continue
        out |= {int(h, 16) for h in _HEX.findall(insn)}
        out |= {int(d) for d in _DEC.findall(_HEX.sub(" ", insn))}
    return out


_forms = forms          # the name the reviewers' scripts use
GATE_MIN = 0x20         # below this a number is in most programs: presence is weak evidence (a probe carries the weight)
TABLE_BYTES = 256       # read-only data (not strings) from this size on is a table
_ROSEC = re.compile(r"^\s*\d+\s+(\.rodata\S*)\s+([0-9a-f]+)\s", re.M)
MARKERS = {"dec": [1237, 1693, 2087, 2411, 3041, 3673, 4349, 5003,
                   6007, 7019, 8053, 9091, 10007, 11057, 12097, 13109],     # round 10, M1: the first eight are unchanged (a draw that passed keeps its marker), the rest are for a large -O0 frame
           16: [0xb3a5, 0x9c4d, 0xd1e7, 0x8f3b, 0xc4f1, 0xa2d9],
           32: [0x9a3c5e71, 0xb7d4a2c9, 0xc3e1957d, 0xd5a18f33, 0x8e47b1d5]}


def split_group(group: list) -> tuple:
    """(alternatives, probe spec or None): the dict at the end of a group is the probe."""
    return [a for a in group if not isinstance(a, dict)], next((a for a in group if isinstance(a, dict)), None)


def group_present(group: list, nums: set, blob: bytes) -> bool:
    """Plain presence of one group's values: some alternative (an int in one of its forms, or a string) is in the code. For a value
    below GATE_MIN this is weak evidence (see the module doc) but never vacuous: a task whose bound, width and neighbours are all absent
    from the code is unanswerable, and passing it because 8 is "in every program" let 8 of 40 char_class tasks through."""
    alts, _spec = split_group(group)
    for alt in alts:
        if isinstance(alt, str):
            if alt.encode() in blob:
                return True
        elif forms(alt) & nums:
            return True
    return False


def _alternatives(kind: str, m: int) -> list:
    return {"exact": [m], "near": [m - 1, m, m + 1], "half": [m - 1, m, m + 1, m // 2 - 1, m // 2, m // 2 + 1, 2 * m - 1, 2 * m, 2 * m + 1]}[kind]


def _marker(spec: dict, avoid: set) -> int | None:
    """A value for the asked literal whose forms (and its neighbours') are not in the first compile, or None when every marker of the table is already a number of
    the code (round 10, M1: a clang -O0 salsa task with 40 rounds, whose `half` probe has nine alternatives per marker, found a frame in which all eight were)."""
    for m in MARKERS[spec.get("bits", 32) if spec.get("hex") else "dec"]:
        if not any(forms(a) & avoid for a in _alternatives(spec.get("kind", "exact"), m)):
            return m
    return None


def cheap_constant(v: int) -> bool:
    """A small value a compiler writes as a shift, a negation or an lea instead of an immediate (`cond ? 16 : 0` is `setne; shl eax, 4`,
    `x * 5` is `lea eax, [rdi+rdi*4]`): 2^k, 2^k +- 1, 3/5/9 times 2^k. The marker probe cannot see that for the real value (a generic
    marker is no power of two), so such a value is not accepted as the answer of a probe."""
    return any(v in (2 ** k - 1, 2 ** k, 2 ** k + 1) or (v >> k) in (3, 5, 9) and v == (v >> k) << k for k in range(0, 12))


def _parse_int(lit: str) -> int:
    return int(lit, 16) if lit.lower().startswith("0x") else int(lit)


def _objdump(args: list, obj: pathlib.Path) -> str:
    return subprocess.run([OBJDUMP, *args, str(obj)], capture_output=True, text=True, timeout=60).stdout


def _compile(src: str, cc: str, opt: str, d: str, name: str) -> pathlib.Path:
    obj = pathlib.Path(d, name)
    r = compile_object(src, cc, opt, obj)
    if r.returncode:
        raise RuntimeError(f"{cc} {opt} failed: {r.stderr.strip()[:200]}")
    return obj


def _probe(spec: dict, alts: list, src: str, cc: str, opt: str, d: str, nums: set, obj: pathlib.Path) -> tuple:
    """(ok, why) for one probe: the literal is in the source and is the group's own value, no table, the marker shows up."""
    m = re.search(spec["at"], src)
    if not m:
        return False, "the asked constant is not at its place in the source"
    try:
        lit = _parse_int(m.group(1))
    except ValueError:
        return False, "the asked literal is not a number"
    if lit not in [a for a in alts if isinstance(a, int)]:
        return False, f"the source holds {m.group(1)} at the asked place, not the task's value"
    if spec.get("kind") == "exact" and 0 < lit < GATE_MIN and cheap_constant(lit):
        return False, f"{lit} is a constant a compiler writes as a shift or an lea"
    if spec.get("no_table"):
        rodata = sum(int(size, 16) for name, size in _ROSEC.findall(_objdump(["-h"], obj)) if not name.startswith(".rodata.str"))
        if rodata >= TABLE_BYTES:
            return False, f"{rodata} bytes of read-only data: a table the compiler made, not in the source"
    marker = _marker(spec, nums)
    if marker is None:         # round 10, M1: a probe that cannot be made is a probe that fails, so the draw is resampled; it never raises out of the gate
        return False, "no free probe marker: every one is already a number of the code"
    digits = m.group(1)
    text = f"0x{marker:0{max(len(digits) - 2, 1)}x}" if spec.get("hex") else str(marker)
    mobj = _compile(src[:m.start(1)] + text + src[m.end(1):], cc, opt, d, "marker.o")
    seen = numbers(_objdump(["-d", "--no-show-raw-insn", "-M", "intel"], mobj))
    if not any(forms(a) & seen for a in _alternatives(spec.get("kind", "exact"), marker)):
        return False, f"a marker in place of {digits} does not show up in the code: the compiler folded the constant away"
    return True, ""


# ---- the object gate (strobf) --------------------------------------------------------------------------------------------------

SECTION_PREFIXES = (".text", ".rodata", ".data")


_BUILD_PATH = re.compile(rb"/[\x20-\x7e]*\.c(?=\x00)")


def object_blob(path: pathlib.Path) -> bytes:
    """The contents of the code and data sections (.text*, .rodata*, .data*) of a relocatable ELF64 little-endian object: what the program
    holds, without symbol names, debug info or comments, and without the path of the C file: the zig-cc wrapper that stands in for clang here
    compiles with UBSan source locations and keeps `/home/<user>/.cache/zig/tmp/<hash>-stdin.c` in .rodata.str, which holds "cach" and "tmp/" (a
    plaintext window) in 15-25% of the clang -O0 objects. A string that starts with / and ends with .c is a path, never a plaintext of this generator."""
    data = pathlib.Path(path).read_bytes()
    if data[:4] != b"\x7fELF" or data[4] != 2 or data[5] != 1:
        raise RuntimeError("the object gate reads ELF64 little-endian objects only")
    shoff, = struct.unpack_from("<Q", data, 0x28)
    shentsize, shnum, shstrndx = struct.unpack_from("<HHH", data, 0x3A)
    heads = [struct.unpack_from("<IIQQQQ", data, shoff + i * shentsize) for i in range(shnum)]       # name, type, flags, addr, offset, size
    names_at = heads[shstrndx][4]
    out = []
    for name, typ, _flags, _addr, offset, size in heads:
        end = data.index(b"\0", names_at + name)
        if typ == 1 and data[names_at + name:end].startswith(tuple(p.encode() for p in SECTION_PREFIXES)):     # PROGBITS
            out.append(data[offset:offset + size])
    return _BUILD_PATH.sub(b"\0", b"\0".join(out))


def plaintext_visible(plain: str, blob: bytes) -> bool:
    """A 4-byte window of the plaintext with at least 3 letters is in the blob (a window of digits and punctuation is in every object)."""
    s = plain.encode()
    return any(s[i:i + 4] in blob for i in range(len(s) - 3) if sum(chr(x).isalpha() for x in s[i:i + 4]) >= 3)


def stored_visible(stored_hex: str, blob: bytes) -> bool:
    b = bytes.fromhex(stored_hex)
    return any(b[i:i + 4] in blob for i in range(max(len(b) - 3, 1)))


def object_gate_check(gate: dict, obj: pathlib.Path) -> tuple:
    """(ok, why) for one compiled object."""
    blob = object_blob(obj)
    if plaintext_visible(gate["hidden"], blob):
        return False, "the plaintext is in the object: the compiler decoded the string at compile time"
    if stored_visible(gate["stored"], blob):
        return True, ""
    return False, "the stored bytes are not in the object"


def check_object(gate: dict, src: str, cc: str, opt: str) -> tuple:
    if not OBJDUMP or not toolchain(cc):
        raise RuntimeError(f"the object gate needs {cc} and objdump on PATH (gen --no-survive skips it)")
    with tempfile.TemporaryDirectory() as d:
        ok, why = object_gate_check(gate, _compile(src, cc, opt, d, "o.o"))
    return ok, (f"{cc} {opt}: {why}" if not ok else "")


def check_visible(src: str, cc: str, opt: str, helpers=()) -> tuple:
    """(ok, why): every byte of every `static const` array of `src` that the binary (linked as the sample stage builds it, a PIE) holds in .rodata is in a window `refdata`
    prints for a data reference of `proc` or of one of `helpers` (visible.py). A source without such an array passes without a compile. Raises RuntimeError when the
    toolchain is missing or the file does not link."""
    arrays = visible.static_arrays(src)
    if not arrays:
        return True, ""
    if not OBJDUMP or not toolchain(cc):
        raise RuntimeError(f"the visibility gate needs {cc} and objdump on PATH (gen --no-survive skips it)")
    with tempfile.TemporaryDirectory() as d:
        exe = pathlib.Path(d, "t")
        r = visible.link(src, cc, opt, exe)
        if r.returncode:
            raise RuntimeError(f"{cc} {opt} failed to link: {r.stderr.strip()[:200]}")
        ok, why = visible.check_exe(exe, arrays, ("proc", *helpers))
    return ok, (f"{cc} {opt}: {why}" if not ok else "")


def has_arrays(t) -> bool:
    return bool(visible.static_arrays(t.body))


def gated(t) -> bool:
    """True when survive checks something of the task: a fact, an object gate, or a table that must be shown."""
    return bool(t.facts or t.object_gate or has_arrays(t))


def check(facts: list, src: str, cc: str, opt: str) -> tuple:
    """(True, "") when every fact group is present, else (False, what is missing). Raises RuntimeError when the toolchain is missing
    or the file does not compile."""
    if not OBJDUMP or not toolchain(cc):
        raise RuntimeError(f"the fact gate needs {cc} and objdump on PATH (gen --no-survive skips it)")
    with tempfile.TemporaryDirectory() as d:
        obj = _compile(src, cc, opt, d, "o.o")
        dis = _objdump(["-d", "--no-show-raw-insn", "-M", "intel"], obj)
        nums, nums_off_stack, blob = numbers(dis), numbers(dis, stack=False), obj.read_bytes()
        for g in facts:
            alts, spec = split_group(g)
            if not group_present(g, nums_off_stack if spec and spec.get("no_stack") else nums, blob):
                return False, f"{cc} {opt}: none of {alts[:6]} in the code" + (" outside the stack frame" if spec and spec.get("no_stack") else "")
            if spec and "at" in spec:
                ok, why = _probe(spec, alts, src, cc, opt, d, nums, obj)
                if not ok:
                    return False, f"{cc} {opt}: {why} ({alts[:3]})"
    return True, ""


def legacy_facts(kind: str, accept: list) -> list:
    """Fact groups of a frozen legacy template, which carry none in their JSON (their output is pinned): the multiplier of an `lcg` task must be an immediate of the
    code, or the answer cannot be read from it (a loop of 3-5 rounds with an increment of 0 is multiplied out by the compiler into one `imul` by a^rounds).
    Used by the training path only (families.untrainable, the Modal-side re-check), not by `check`."""
    if kind == "lcg":
        return [[int(accept[0])]]
    return []


def check_task(t, cc: str | None = None, opt: str | None = None) -> tuple:
    """The gate for one task at its own compiler and -O level (or the ones given): its facts, its object gate (strobf), and that the tables of its code are shown
    (check_visible). Tasks with none of the three pass."""
    cc, opt = cc or t.cc or "gcc", opt or t.opt
    if t.facts:
        ok, why = check(t.facts, t.source(), cc, opt)
        if not ok:
            return False, why
    if t.object_gate:
        ok, why = check_object(t.object_gate, t.source(), cc, opt)
        if not ok:
            return False, why
    return check_visible(t.source(), cc, opt, t.helpers)


class Gate:
    """Callable for `core.build`: True when the task's facts survive at its own cc/opt. Counts what it did.

    The toolchain is looked up when a task with facts is drawn, not when the gate is made: a run of the frozen legacy family (no facts)
    needs neither compiler nor objdump, and a machine without clang gates the gcc draws, warns once, and counts the clang draws it could
    not check (`unchecked`), which the summary reports."""

    def __init__(self, warn=None):
        self.tried = self.rejected = self.unshown = 0
        self.by_variant: dict = {}
        self.unchecked: dict = {}
        self.warn = warn or (lambda msg: print(f"warning: {msg}", file=sys.stderr))

    def available(self, cc: str) -> bool:
        return bool(OBJDUMP and toolchain(cc))

    def __call__(self, t) -> bool:
        if not gated(t):
            return True
        cc = t.cc or "gcc"
        if not self.available(cc):
            if cc not in self.unchecked:
                self.warn(f"the fact gate needs {cc} and objdump on PATH: tasks drawn with {cc} are NOT checked against the compiled code "
                          f"(--cc gcc gates every task if only gcc is installed)")
            self.unchecked[cc] = self.unchecked.get(cc, 0) + 1
            return True
        ok, why = check_task(t)
        self.tried += 1
        self.rejected += not ok
        self.unshown += not ok and "refdata prints" in why
        row = self.by_variant.setdefault(f"{t.kind}/{t.variant}", [0, 0])
        row[0] += 1
        row[1] += not ok
        return ok

    def legacy_visible(self, t) -> tuple:
        """(ok, why) for a frozen legacy task (legacy_facts): its asked constant is in the code compiled with gcc at its -O level. A machine without gcc or objdump
        cannot tell and passes the task (counted in `unchecked`; the Modal-side re-check applies)."""
        facts = legacy_facts(t.kind, t.truth["accept"])
        if not facts:
            return True, ""
        if not self.available("gcc"):
            self.unchecked["gcc"] = self.unchecked.get("gcc", 0) + 1
            return True, ""
        return check(facts, t.source(), "gcc", t.opt)         # not counted in tried/rejected: those are redraws, a legacy task is dropped (families.untrainable)

    def legacy_shown(self, t) -> tuple:
        """(ok, why) for a frozen legacy task: the `static const` strings of its code are shown to the model when it is compiled with gcc at its -O level (check_visible). Dropped, not
        redrawn, like legacy_visible; a machine without gcc or objdump passes it (counted in `unchecked`)."""
        if not has_arrays(t):
            return True, ""
        if not self.available("gcc"):
            self.unchecked["gcc"] = self.unchecked.get("gcc", 0) + 1
            return True, ""
        return check_visible(t.source(), "gcc", t.opt)

    def summary(self) -> str:
        un = "".join(f"; {n} draws with {cc} not checked ({cc} or objdump missing)" for cc, n in sorted(self.unchecked.items()))
        shown = f" (of them {self.unshown} because a byte of a table is not in any window refdata prints)" if self.unshown else ""
        return f"fact gate: {self.rejected} of {self.tried} draws rejected and resampled{shown}{un}"


def survival_rates(tasks, ccs=("gcc", "clang")) -> dict:
    """{(family/variant, cc): [present, total]} over every task and every -O level; tasks without facts or an object gate are skipped."""
    out: dict = {}
    for t in tasks:
        if not t.facts and not t.object_gate:
            continue
        for cc in ccs:
            if not toolchain(cc):
                continue
            row = out.setdefault((f"{t.kind}/{t.variant}", cc), [0, 0])
            for opt in OPTS:
                row[0] += check_task(t, cc, opt)[0]
                row[1] += 1
    return out
