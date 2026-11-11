"""Does the model see the bytes the answer is made of?  (round 9, B1)

The sample stage shows the model two things: the Ghidra pseudocode of `proc` (and of its helpers) and, for every `DAT_`/`PTR_`/`s_` label in it, the bytes
`tools/modal/rea-bench/refdata` prints at that address: `refdata.N` = 64 of them FORWARD from the address, plus the ASCII up to the first NUL when they are printable
(Ghidra puts such a string inline). A byte of a table that lies before every referenced address, or more than 64 bytes after it, is never shown. Round 9 found it for
`strobf/reverse_xor`: gcc turned `enc[n - 1 - i]` into a descending pointer whose only data operand is the address of the LAST element, so 15 of the 16 encoded bytes were
never printed and the plaintext could not be read (20 of 24 production tasks).

This module is the one definition of what is shown, used by the generator's gate (`survive.check_task`, the Modal-side re-check) and by the byte-flip scanner
(`bytescan.py`, which finds the bytes the program's behaviour depends on and asks this model whether each is shown):

  - the data references of a function are the rip-relative operands of its instructions (`lea rcx,[rip+0xea9]  # 201f`) that point into .rodata of the linked
    PIE; Ghidra loads a PIE at 0x100000 and labels each such address;
  - the window of a reference is [address, address + WINDOW), and the printable run up to the first NUL (a string is shown whole);
  - a reference that is a jump table (`lea r,[rip+..]; movsxd x,[r+i*4]; jmp x`) is resolved by Ghidra into the cases of a switch: it shows no window, and the table's own
    entries (the 4-byte offsets that land inside the function) count as shown;
  - an address DERIVED from a reference (`lea rsi,[rax-0x12]`, a pointer that runs on) is not a reference: Ghidra labels no address there, so nothing is shown at it.

`static_arrays` reads the `static const` arrays of a task's body: the tables, keys, alphabets and encoded strings that live in .rodata. `check_exe` asks that every byte of
every such array that the linked binary keeps in .rodata lies in a shown window; an array the compiler folded into immediates is in the pseudocode itself and needs no window.
"""
from __future__ import annotations

import pathlib
import re
import shutil
import struct
import subprocess

WINDOW = 64                  # refdata.N (tools/modal/rea-bench/refdata): the bytes it prints at an address
MIN_ARRAY = 4                # a shorter array cannot be told from other bytes of .rodata
OBJDUMP = shutil.which("objdump")

_REF = re.compile(r"\[rip[+-]0x[0-9a-f]+\]\s+# ([0-9a-f]+)")
_FUNC_HEAD = re.compile(r"^([0-9a-f]+) <([^>]+)>:$")
_INSN = re.compile(r"^\s*([0-9a-f]+):\t(.*)$")
_LEA = re.compile(r"lea\s+(\w+),")

# ---- the `static const` arrays of a body ----------------------------------------------------------------------------------------

_WIDTH = {"uint8_t": 1, "char": 1, "int8_t": 1, "uint16_t": 2, "int16_t": 2, "uint32_t": 4, "int32_t": 4, "int": 4, "unsigned": 4, "uint64_t": 8, "int64_t": 8}
_ARRAY = re.compile(r"static\s+const\s+(?:unsigned\s+)?(uint8_t|int8_t|char|uint16_t|int16_t|uint32_t|int32_t|int|unsigned|uint64_t|int64_t)\s+(\w+)((?:\s*\[[^\]]*\])+)\s*="
                    r"\s*(\{.*?\}|\"(?:[^\"\\]|\\.)*\")\s*;", re.S)
_ITEM = re.compile(r"'(?:\\.|[^'\\])'|-?0[xX][0-9a-fA-F]+|-?\d+")
_ESC = {"n": 10, "t": 9, "r": 13, "0": 0, "\\": 92, "'": 39, '"': 34, "a": 7, "b": 8, "f": 12, "v": 11}


def _c_string(lit: str) -> bytes:
    """The bytes of a C string literal body (without the quotes): \\n \\t \\\\ \\\" \\xHH and octal escapes, anything else as itself."""
    out, i = bytearray(), 0
    while i < len(lit):
        c = lit[i]
        if c != "\\":
            out += c.encode("latin-1")
            i += 1
            continue
        i += 1
        if lit[i] == "x":
            m = re.match(r"[0-9a-fA-F]+", lit[i + 1:])
            out.append(int(m.group(0), 16) & 0xff)
            i += 1 + len(m.group(0))
        elif lit[i] in "01234567":
            m = re.match(r"[0-7]{1,3}", lit[i:])
            out.append(int(m.group(0), 8) & 0xff)
            i += len(m.group(0))
        else:
            out.append(_ESC.get(lit[i], ord(lit[i])))
            i += 1
    return bytes(out)


def _item(tok: str) -> int:
    if tok.startswith("'"):
        body = tok[1:-1]
        return _c_string(body)[0] if body else 0
    return int(tok, 16) if "0x" in tok.lower() else int(tok)


def static_arrays(body: str) -> list:
    """[(name, bytes)] of every `static const <integer type> name[..] = {..} | "..";` in the C text, as the compiler lays it out in memory (little endian, row-major,
    a string with its NUL, zero padding up to a declared size). Arrays of fewer than MIN_ARRAY bytes are left out."""
    out = []
    for m in _ARRAY.finditer(body):
        typ, name, dims, init = m.groups()
        width = _WIDTH[typ]
        if init.startswith('"'):
            raw = _c_string(init[1:-1]) + b"\0"
        else:
            raw = b"".join((_item(tok) & ((1 << 8 * width) - 1)).to_bytes(width, "little") for tok in _ITEM.findall(init))
        sizes = [int(x) for x in re.findall(r"\[\s*(\d+)\s*\]", dims)]
        if sizes and len(sizes) == dims.count("["):
            total = width
            for s in sizes:
                total *= s
            raw = raw[:total].ljust(total, b"\0")
        if len(raw) >= MIN_ARRAY:
            out.append((name, raw))
    return out


# ---- the linked binary ------------------------------------------------------------------------------------------------------------

class Exe:
    """What the checks need of a linked ELF64 little-endian executable: sections, function symbols, .rodata."""

    def __init__(self, path):
        self.path = str(path)
        data = pathlib.Path(path).read_bytes()
        if data[:4] != b"\x7fELF" or data[4] != 2 or data[5] != 1:
            raise RuntimeError("the visibility check reads ELF64 little-endian executables only")
        self.etype, = struct.unpack_from("<H", data, 0x10)
        shoff, = struct.unpack_from("<Q", data, 0x28)
        shentsize, shnum, shstrndx = struct.unpack_from("<HHH", data, 0x3A)
        heads = [struct.unpack_from("<IIQQQQII", data, shoff + i * shentsize) for i in range(shnum)]     # name, type, flags, addr, offset, size, link, info
        names_at = heads[shstrndx][4]
        self.sections = {}
        self.symtab = None
        for name, typ, _flags, addr, offset, size, link, _info in heads:
            nm = data[names_at + name:data.index(b"\0", names_at + name)].decode()
            self.sections.setdefault(nm, (addr, offset, size))
            if typ == 2:
                self.symtab = (offset, size, heads[link][4])
        self.data = data
        if ".rodata" in self.sections:
            vma, off, size = self.sections[".rodata"]
            self.rodata_vma, self.rodata = vma, data[off:off + size]
        else:
            self.rodata_vma, self.rodata = 0, b""

    def functions(self, bases) -> dict:
        """{symbol: (address, size)} of the function symbols named `base` or `base.<suffix>` (proc.part.0, proc.cold, helper.constprop.0) for a base in `bases`."""
        out = {}
        if not self.symtab:
            return out
        offset, size, strtab = self.symtab
        for i in range(size // 24):
            st_name, info, _other, shndx, value, sz = struct.unpack_from("<IBBHQQ", self.data, offset + 24 * i)
            if info & 0xf != 2 or not shndx or not sz:
                continue
            nm = self.data[strtab + st_name:self.data.index(b"\0", strtab + st_name)].decode()
            if any(nm == b or nm.startswith(b + ".") for b in bases) and (nm not in out or value < out[nm][0]):      # a name defined twice (the scanner links a harness copy): the lower address was linked first
                out[nm] = (value, sz)
        return out


def _disassemble(exe: Exe, addr: int, size: int) -> list:
    text = subprocess.run([OBJDUMP, "-d", "--no-show-raw-insn", "-M", "intel", f"--start-address={addr:#x}", f"--stop-address={addr + size:#x}", exe.path],
                          capture_output=True, text=True, timeout=60).stdout
    out = []
    for line in text.splitlines():
        m = _INSN.match(line)
        if m:
            out.append((int(m.group(1), 16), m.group(2).strip()))
    return out


class Model:
    """What is shown: `refs` (addresses the functions reference in .rodata, jump tables apart), `jump_tables`, `shown` (the set of addresses printed)."""

    def __init__(self, exe: Exe, bases):
        self.exe = exe
        self.refs, self.jump_tables, self.functions = [], [], exe.functions(bases)
        for name, (addr, size) in sorted(self.functions.items(), key=lambda kv: kv[1][0]):
            self._scan(_disassemble(exe, addr, size), addr, addr + size)
        self.shown = set()
        for a in self.refs:
            self.shown |= set(range(a, a + WINDOW))
            run = self.printable_run(a)
            if run:
                self.shown |= set(range(a, a + run + 1))
        for table, lo, hi in self.jump_tables:      # Ghidra turns the table into the cases of a switch: its entries (offsets that land in the function) are what the pseudocode shows
            n = 0
            while self.in_rodata(table + 4 * n + 3):
                o = table - exe.rodata_vma + 4 * n
                if not lo <= table + int.from_bytes(exe.rodata[o:o + 4], "little", signed=True) < hi:
                    break
                n += 1
            self.shown |= set(range(table, table + 4 * n))

    def in_rodata(self, a: int) -> bool:
        return self.exe.rodata_vma <= a < self.exe.rodata_vma + len(self.exe.rodata)

    def printable_run(self, a: int) -> int:
        """Length of the printable ASCII run that starts at address `a` of .rodata, up to the first NUL (0 when it is not one: Ghidra shows no string there)."""
        o = a - self.exe.rodata_vma
        run = self.exe.rodata[o:o + 4096].split(b"\0", 1)[0]
        return len(run) if run and all(32 <= c < 127 or c in (9, 10) for c in run) else 0

    def _scan(self, ins: list, lo: int, hi: int) -> None:
        for k, (_addr, text) in enumerate(ins):
            m = _REF.search(text)
            if not m or not self.in_rodata(int(m.group(1), 16)):
                continue
            target = int(m.group(1), 16)
            reg = _LEA.match(text)
            rest = " ".join(re.sub(r"#.*", "", x) for _, x in ins[k + 1:])
            if reg and re.search(r"movsxd\s+\w+,DWORD PTR \[" + reg.group(1) + r"\+\w+\*4\]", rest) and re.search(r"jmp\s+r[a-z0-9]+\b", rest):
                self.jump_tables.append((target, lo, hi))
            else:
                self.refs.append(target)

    def hidden(self, lo: int, hi: int) -> list:
        """The addresses of [lo, hi) that are not shown."""
        return [a for a in range(lo, hi) if a not in self.shown]


def locate(arr: bytes, rod: bytes) -> list:
    """[(offset in rod, length)] where the bytes of `arr` are in .rodata: the whole array when it is there contiguously, else its longest runs of at least MIN_ARRAY
    bytes (the compiler may split an array into vector constants). A byte found nowhere is not in .rodata: it is an immediate of the code, which the pseudocode shows."""
    o = rod.find(arr)
    if o >= 0:
        return [(o, len(arr))]
    runs, i = [], 0
    while i < len(arr):
        for ln in range(len(arr) - i, MIN_ARRAY - 1, -1):
            o = rod.find(arr[i:i + ln])
            if o >= 0:
                runs.append((o, ln))
                i += ln
                break
        else:
            i += 1
    return runs


def hidden_bytes(model: Model, arrays: list) -> list:
    """[(name, array length, [addresses not shown])] for the arrays with a byte outside every window."""
    out = []
    for name, arr in arrays:
        gone = []
        for off, ln in locate(arr, model.exe.rodata):
            gone += model.hidden(model.exe.rodata_vma + off, model.exe.rodata_vma + off + ln)
        if gone:
            out.append((name, len(arr), gone))
    return out


def check_exe(exe_path, arrays: list, bases=("proc",)) -> tuple:
    """(ok, why) for a linked binary: every byte of every `arrays` entry that .rodata holds is inside a window refdata prints for a data reference of `proc` (or a helper
    in `bases`, or a piece the compiler split off them)."""
    if not arrays:
        return True, ""
    if not OBJDUMP:
        raise RuntimeError("the visibility check needs objdump on PATH")
    exe = Exe(exe_path)
    if exe.etype != 3:
        raise RuntimeError("the visibility check models a PIE (Ghidra at 0x100000, rip-relative data references): link with -fPIE -pie")
    model = Model(exe, bases)
    bad = hidden_bytes(model, arrays)
    if not bad:
        return True, ""
    name, n, gone = bad[0]
    starts = sorted(a - exe.rodata_vma for a in model.refs)
    return False, (f"{len(gone)} of the {n} bytes of `{name}` are outside every window refdata prints ({WINDOW} bytes forward from a data reference of the function; "
                   f"references at .rodata+{', +'.join(hex(s) for s in starts[:4]) or 'none'}, bytes at +{hex(gone[0] - exe.rodata_vma)}...)")


def link(src: str, cc: str, opt: str, out) -> subprocess.CompletedProcess:
    """The binary the sample stage decompiles (tools/modal/modal_rea_data.py: `{cc} {opt} -fno-lto -o t t.c`), as a PIE with the symbols kept. `-g` does not change the
    code of either compiler; it is what keeps the symbol table when the `clang` is the zig-cc wrapper, which strips every optimised link otherwise.
    `-fno-sanitize=undefined`: the wrapper adds UBSan checks (calls and source-location records in .rodata) to its -O0 code, which the clang of the Modal image does not."""
    return subprocess.run([cc, opt, "-fno-lto", "-w", "-g", "-fno-sanitize=undefined", "-fPIE", "-pie", "-o", str(out), "-x", "c", "-", "-lm"], input=src, capture_output=True, text=True, timeout=60)
