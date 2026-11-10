"""Command line: `python3 tools/rea-bench gen ...` (or `PYTHONPATH=tools/rea-bench python3 -m rea_bench gen ...`).

    gen  --family crc --n 60 --seed 7 --out /tmp/rea-crc        regenerate one family (writes <id>.c and <id>.json per task)
    gen  --family all --n 600 --seed 7 --out /tmp/rea-all       every family, n split evenly (legacy included)
    gen  --family crc,cipher --n 120 --seed 1001 --cc clang     a list; --cc gcc|clang|mix (default mix)
    list                                                       families and their variants
    check --family all --n 330 --seed 1001                     compile every task at -O0..-O3/-Os with -Wall -Werror (gcc, and
                                                               clang if installed) and run its harness; exit 1 on any problem

The old invocation `gen_rea_tasks.py --n N --seed S --out D` still works (it means `gen --family legacy`).
"""
from __future__ import annotations

import argparse
import pathlib
import sys

from . import families
from .core import write_tasks


def cmd_gen(a) -> None:
    mods = families.modules()
    names = list(mods) if a.family == "all" else a.family.split(",")
    for nm in names:
        if nm not in mods:
            raise SystemExit(f"unknown family {nm!r}; known: {', '.join(mods)} (or 'all')")
    counts = families.split_counts(a.n, len(names))
    total = 0
    for nm, c in zip(names, counts):
        tasks = families.generate(nm, c, a.seed, a.cc)
        write_tasks(tasks, a.out)
        total += len(tasks)
        print(f"  {nm}: {len(tasks)} tasks")
    print(f"{total} tasks -> {a.out}")


def cmd_check(a) -> None:
    from . import verify
    mods = families.modules()
    names = list(mods) if a.family == "all" else a.family.split(",")
    for nm in names:
        if nm not in mods:
            raise SystemExit(f"unknown family {nm!r}; known: {', '.join(mods)} (or 'all')")
    ubsan = verify.ubsan_available() and not a.no_ubsan
    print(f"checking with gcc{' + clang' if verify.CLANG else ' (clang not installed)'}, UBSan harness {'on' if ubsan else 'off'}")
    bad = 0
    for nm, c in zip(names, families.split_counts(a.n, len(names))):
        problems = []
        tasks = families.generate(nm, c, a.seed, "mix")
        for t in tasks:
            problems += verify.check_task(t, ubsan)
        print(f"  {nm}: {len(tasks)} tasks, {len(problems)} problems", flush=True)
        for pr in problems[:20]:
            print("    " + pr)
        bad += len(problems)
    if bad:
        raise SystemExit(f"{bad} problems")
    print("all tasks compile clean and their harnesses prove the truth")


def cmd_list(_a) -> None:
    for nm, m in families.modules().items():
        print(f"{nm:10s} {len(m.VARIANTS):2d} variants: {', '.join(m.VARIANTS)}")


def main(argv=None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0].startswith("--"):      # old gen_rea_tasks.py invocation
        argv = ["gen", "--family", "legacy", *argv]
    ap = argparse.ArgumentParser(prog="rea-bench", description="REA task generator with automatic ground truth")
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen", help="generate tasks")
    g.add_argument("--family", default="all", help="family name, comma list, or 'all'")
    g.add_argument("--n", type=int, default=200)
    g.add_argument("--seed", type=int, default=1)
    g.add_argument("--cc", choices=["gcc", "clang", "mix"], default="mix", help="compiler written to the task JSON (new families)")
    g.add_argument("--out", type=pathlib.Path, required=True)
    g.set_defaults(fn=cmd_gen)
    s = sub.add_parser("list", help="list families and variants")
    s.set_defaults(fn=cmd_list)
    c = sub.add_parser("check", help="compile and run the harness of generated tasks (any seed)")
    c.add_argument("--family", default="all")
    c.add_argument("--n", type=int, default=110)
    c.add_argument("--seed", type=int, default=1)
    c.add_argument("--no-ubsan", action="store_true")
    c.set_defaults(fn=cmd_check)
    a = ap.parse_args(argv)
    a.fn(a)
