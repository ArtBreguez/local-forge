"""The byte-flip scan: which bytes of .rodata does the program's BEHAVIOUR depend on, and does the model see each of them?  (round 9, B1)

The reviewer's scanner of round 9, adapted: instead of asking whether the stored array of a task is shown (survive.check_visible, a static question about the arrays of the
source), it asks it of the program. The pipeline's `proc` is compiled as the sample stage builds it (`cc opt -fPIE`, the task's own compiler and -O level), linked with a
harness (the family's own `check` main, gcc -O0, which calls THAT `proc`), and every byte of the .rodata that the pipeline object contributes is flipped in the linked binary
(xor 0xff, and xor 0x01) one at a time; a flip that changes the harness's exit status or output is a byte the behaviour needs. `visible.Model` says what is shown; a needed byte
that is not shown is an UNCOVERED byte: the answer is read from data the model never gets.

    python3 tools/rea-bench bytescan --family strobf --seed 9091 --per-variant 24 --gated --own

A task with an uncovered byte in an array the SOURCE wrote (a `static const` table, key, alphabet or encoded string), an error, or a harness that disagrees with the truth is a FINDING
(one JSON line each). A needed byte that is in none of those arrays is data the compiler made out of the code (CRC tables it builds from a bit loop, vector constants of immediates):
the line says NOTE, and the table counts those tasks in their own column: such a byte is not the answer unless the question asks for what the table is made of (the polynomial; the gate's
`no_table` rule keeps those tasks out), so it is reported, not failed. The summary table is per variant. The scan is a check, not part of the gate: it runs the program once per byte
(a few hundred runs per task). Legacy tasks have no harness of their own: `legacy_main` feeds each kind inputs for which its .rodata strings decide the output.
"""
from __future__ import annotations

import json
import os
import pathlib
import re
import resource
import subprocess
import tempfile

from . import families, survive, visible

FLIPS = (0xff, 0x01)
COMBOS = [("gcc", "-O2"), ("clang", "-O2"), ("gcc", "-O3"), ("clang", "-O3"), ("gcc", "-Os"), ("clang", "-Os"), ("gcc", "-O1"), ("clang", "-O1"), ("gcc", "-O0"), ("clang", "-O0")]


def legacy_main(t) -> str:
    """A `main` that calls the frozen legacy `proc` with inputs for which the strings of its .rodata decide what is printed."""
    acc = t.truth["accept"]
    cases = {
        "xor-key": 'int main(void) { uint8_t b[48]; memset(b, 0, sizeof b); proc(b, sizeof b); for (size_t i = 0; i < sizeof b; i++) printf("%02x", b[i]); puts(""); return 0; }',
        "password": f'int main(void) {{ printf("%d %d %d\\n", proc("{acc[0]}"), proc("{acc[0]}x"), proc("")); return 0; }}',
        "magic": 'int main(void) { uint8_t h[8] = {1, 2, 3, 4, 1, 0, 0, 0}; printf("%d\\n", proc(h, 8)); return 0; }',
        "lcg": 'int main(void) { printf("%u %u\\n", proc(1u), proc(12345u)); return 0; }',
        "hash": 'int main(void) { printf("%u %u\\n", proc("hello"), proc("")); return 0; }',
        "skip": 'int main(void) { uint8_t p[6] = {0, 1, 2, 10, 32, 255}; printf("%u\\n", proc(p, 6)); return 0; }',
        "keycount": f'int main(void) {{ printf("%d\\n", proc("{acc[0]}=1\\n#{acc[0]}=2\\nother=3\\n{acc[0]}=4\\n")); return 0; }}',
        "alphabet": 'int main(void) { for (int c = 0; c < 256; c++) printf("%d ", proc((char)c)); puts(""); return 0; }',
        "rotate": 'int main(void) { printf("%u %u\\n", proc(0x12345678u), proc(0u)); return 0; }',
        "threshold": 'int main(void) { for (int v = -600; v < 5200; v += 7) printf("%d", proc(v)); puts(""); return 0; }',
    }
    return cases[t.kind]


GUARD = """
#include <signal.h>
#include <unistd.h>
static void bscan_die(int s) { _exit(100 + (s & 31)); }
__attribute__((constructor)) static void bscan_arm(void) {
    signal(SIGSEGV, bscan_die); signal(SIGBUS, bscan_die); signal(SIGFPE, bscan_die); signal(SIGILL, bscan_die); signal(SIGABRT, bscan_die); signal(SIGALRM, bscan_die);
    alarm(2);
}
"""      # a flipped byte that crashes or loops ends the program at once with a status of its own: without it every crash costs about a second in the kernel's core handling


def harness(t) -> tuple:
    """(main C text, expected stdout or None, preamble) of a task."""
    if t.check:
        main_c, want, *pre = t.check
        return main_c, want, (pre[0] if pre else "")
    return legacy_main(t), None, ""


def _sh(cmd, work, inp=None, env=None, check=True):
    p = subprocess.run(cmd, input=inp, capture_output=True, env=env or work["env"], timeout=120)
    if check and p.returncode:
        raise RuntimeError((p.stderr or p.stdout).decode(errors="replace")[:300])
    return p


def _run(exe, timeout=6):
    p = subprocess.run([exe], capture_output=True, timeout=timeout, preexec_fn=lambda: resource.setrlimit(resource.RLIMIT_CORE, (0, 0)))
    return p.returncode, p.stdout


def _a_ranges(map_path: str) -> list:
    """The (lo, hi) address ranges of .rodata that the pipeline object `a.o` contributes, from the linker map."""
    out, in_ro, pend = [], False, None
    with open(map_path, errors="replace") as fh:
        lines = fh.readlines()
    for ln in lines:
        if re.match(r"^\.\S+", ln):
            in_ro = ln.startswith(".rodata ") or ln.startswith(".rodata\t")
            pend = None
            continue
        if not in_ro:
            continue
        m1 = re.match(r"^ (\.\S+)\s+0x([0-9a-f]+)\s+0x([0-9a-f]+)\s+(\S+)\s*$", ln)
        if m1:
            _nm, ad, sz, fl = m1.groups()
        else:
            m0 = re.match(r"^ (\.\S+)\s*$", ln)
            if m0:
                pend = m0.group(1)
                continue
            m2 = re.match(r"^\s+0x([0-9a-f]+)\s+0x([0-9a-f]+)\s+(\S+)\s*$", ln)
            if not (m2 and pend):
                continue
            ad, sz, fl = m2.groups()
            pend = None
        if fl.endswith("a.o"):
            out.append((int(ad, 16), int(ad, 16) + int(sz, 16)))
    return out


def scan_task(t, cc: str, opt: str, work: dict) -> dict:
    """Build the pipeline `proc` with the task's harness and flip the bytes of its .rodata. Keys: base_ok, needed (bytes), uncovered (needed, not shown), the first few of them."""
    main_c, want, pre = harness(t)
    pre = re.sub(r"\bstatic ", "", pre)     # antidebug: the preamble's fakes (`static long fake_rv; static long fake_ptrace(...)`) must be ONE object for the two translation units, or the harness sets a variable the pipeline's proc never reads
    d = work["dir"]
    a, h, exe, mp, flipped = (os.path.join(d, n) for n in ("a.o", "h.o", "x.out", "x.map", "p.out"))
    _sh([cc, opt, "-w", "-fno-sanitize=undefined", "-fPIE", "-Dmain=stub_main", "-c", "-o", a, "-x", "c", "-"], work, t.source(preamble=pre).encode())
    _sh(["/usr/bin/gcc", "-O0", "-w", "-fPIE", "-fno-merge-constants", "-c", "-o", h, "-x", "c", "-"], work, (t.source(main_c, pre) + GUARD).encode(), env=os.environ)
    _sh(["/usr/bin/gcc", "-pie", "-g", "-o", exe, a, h, f"-Wl,-Map={mp}", "-Wl,--allow-multiple-definition", "-lm"], work, env=os.environ)
    names = [ln.split()[2] for ln in _sh(["nm", a], work).stdout.decode().splitlines() if len(ln.split()) == 3 and ln.split()[1] in "Tt" and ln.split()[2] != "stub_main"]
    rc, base = _run(exe)
    base_ok = rc == 0 and (want is None or base.decode() == want)
    model = visible.Model(visible.Exe(exe), names)
    ex = model.exe
    ranges = _a_ranges(mp)
    needed = []
    if base_ok and ex.rodata:
        _vma, foff, _size = ex.sections[".rodata"]
        data = bytearray(ex.data)
        for lo, hi in ranges:
            for addr in range(max(lo, ex.rodata_vma), min(hi, ex.rodata_vma + len(ex.rodata))):
                o = foff + addr - ex.rodata_vma
                old = data[o]
                for mask in FLIPS:
                    data[o] = old ^ mask
                    pathlib.Path(flipped).write_bytes(data)
                    os.chmod(flipped, 0o755)
                    try:
                        rc2, out2 = _run(flipped)
                    except Exception:
                        rc2, out2 = -1, b"CRASH"
                    if (rc2, out2) != (rc, base):
                        needed.append(addr)
                        break
                data[o] = old
    miss = [x for x in needed if x not in model.shown]
    written = set()                      # the bytes of the `static const` arrays the SOURCE has: data the author wrote, as against data the compiler made out of the code
    for _name, arr in visible.static_arrays(t.body):
        for off, ln in visible.locate(arr, ex.rodata):
            written |= set(range(ex.rodata_vma + off, ex.rodata_vma + off + ln))
    miss_made = [x for x in miss if x not in written]
    return {"base_ok": base_ok, "uncovered_made": len(miss_made), "uncovered_written": len(miss) - len(miss_made), "rodata": len(ex.rodata), "a_rodata": sum(hi - lo for lo, hi in ranges), "refs": len(model.refs), "jump_tables": len(model.jump_tables),
            "needed": len(needed), "uncovered": len(miss), "uncovered_first": [hex(x - ex.rodata_vma) for x in miss[:8]],
            "needed_span": [hex(min(needed) - ex.rodata_vma), hex(max(needed) - ex.rodata_vma)] if needed else None,
            "reference_offsets": sorted({hex(x - ex.rodata_vma) for x in model.refs})[:12]}


def scan(family: str, seed: int, per_variant: int, own: bool = True, gated: bool = True, combos=None, variants=None, out=print) -> dict:
    """Scan `per_variant` tasks of every variant of the family (drawn through the gate when `gated`), each at its own cc/-O (own) or cycling through `combos`.
    Prints one FINDING line per task with an uncovered byte, an error, or a harness that disagrees with the truth; returns the per-variant table."""
    mods = families.modules()
    nv = len(mods[family].VARIANTS)
    gate = survive.Gate() if gated else None
    tasks = families.generate(family, per_variant * nv * (3 if gate and family == "legacy" else 1), seed, "mix", gate)
    if gate and family == "legacy":         # the frozen family is never redrawn: the population the data path trains on is what families.untrainable keeps
        tasks = [t for t in tasks if not families.untrainable(t, seed, gate)]
    combos = combos or COMBOS
    env = {**os.environ, "PATH": os.path.expanduser("~/work/tmp-rea/bin") + ":" + os.environ["PATH"]}
    table, seen = {}, {}
    for t in tasks:
        v = t.variant or t.kind
        if variants and v not in variants:
            continue
        i = seen.get(v, 0)
        if i >= per_variant:
            continue
        seen[v] = i + 1
        cc, opt = ((t.cc or "gcc"), t.opt) if own else combos[i % len(combos)]
        row = table.setdefault(v, {"tasks": 0, "needed": 0, "uncovered_tasks": 0, "uncovered_bytes": 0, "written_tasks": 0, "made_only_tasks": 0, "base_bad": 0, "errors": 0, "no_rodata": 0, "ccopt": {}})
        row["tasks"] += 1
        row["ccopt"][f"{cc}{opt}"] = row["ccopt"].get(f"{cc}{opt}", 0) + 1
        with tempfile.TemporaryDirectory(prefix="bscan-", dir=os.environ.get("TMPDIR")) as d:
            try:
                r = scan_task(t, cc, opt, {"dir": d, "env": env})
            except Exception as e:
                r = {"error": repr(e)[:200]}
        r.update({"id": t.id, "variant": v, "cc": cc, "opt": opt})
        if r.get("error"):
            row["errors"] += 1
        else:
            row["base_bad"] += not r["base_ok"]
            row["needed"] += r["needed"]
            row["no_rodata"] += not r["rodata"]
            row["uncovered_tasks"] += bool(r["uncovered"])
            row["uncovered_bytes"] += r["uncovered"]
            row["written_tasks"] += bool(r["uncovered_written"])
            row["made_only_tasks"] += bool(r["uncovered"]) and not r["uncovered_written"]
        if r.get("error") or r.get("uncovered") or not r.get("base_ok", True):
            out(("FINDING " if r.get("error") or r.get("uncovered_written") or not r.get("base_ok", True) else "NOTE (data the compiler made, not an array of the source) ") + json.dumps(r))
    return table


def format_table(family: str, table: dict) -> list:
    lines = [f"{'family/variant':30s} {'tasks':>5s} {'needed bytes':>12s} {'uncovered tasks':>15s} {'in written arrays':>17s} {'compiler-made only':>18s} {'harness bad':>11s} {'errors':>6s}"]
    for v, r in table.items():
        lines.append(f"{family + '/' + v:30s} {r['tasks']:5d} {r['needed']:12d} {r['uncovered_tasks']:15d} {r['written_tasks']:17d} {r['made_only_tasks']:18d} {r['base_bad']:11d} {r['errors']:6d}")
    return lines
