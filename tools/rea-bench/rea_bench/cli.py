"""Command line: `python3 tools/rea-bench gen ...` (or `PYTHONPATH=tools/rea-bench python3 -m rea_bench gen ...`).

    gen  --family crc --n 60 --seed 7 --out /tmp/rea-crc        regenerate one family (writes <id>.c and <id>.json per task)
    gen  --family all --n 600 --seed 7 --out /tmp/rea-all       every family, n split evenly (legacy included)
    gen  --family crc,cipher --n 120 --seed 1001 --cc clang     a list; --cc gcc|clang|mix (default mix)
    list                                                       families and their variants
    check --family all --n 330 --seed 1001                     compile every task at -O0..-O3/-Os with -Wall -Werror (gcc, and
                                                               clang if installed), run its harness, report duplicate tasks inside
                                                               the seed; exit 1 on any problem
    heldout [--write]                                          verify (or regenerate) the pinned list of held-out tasks (seed 1001)
    chance [--write]                                           estimated chance level per variant (JSON field `chance`)
    survival --family all --n 120                              share of tasks whose asked constant is in the compiled code, per variant
    stats --seed 7 --n 3000                                    per family: distinct programs and overlap with the held-out list

The old invocation `gen_rea_tasks.py --n N --seed S --out D` still works (it means `gen --family legacy`).
"""
from __future__ import annotations

import argparse
import pathlib
import re
import sys

from . import dedupe, families
from . import core
from .core import write_tasks


TASK_FILE = re.compile(r"^[a-z]+-\d+-\d{4}\.(c|json)$")


def pick_families(a, allow_small: bool = False) -> list:
    mods = families.modules()
    names = list(mods) if a.family == "all" else a.family.split(",")
    for nm in names:
        if nm not in mods:
            raise SystemExit(f"unknown family {nm!r}; known: {', '.join(mods)} (or 'all')")
    if a.n < len(names) and not allow_small:
        raise SystemExit(f"--n {a.n} is smaller than the {len(names)} families asked for: some would get 0 tasks (use --n >= {len(names)})")
    return names


def make_gate(a):
    """The fact gate (survive.Gate) unless --no-survive. It needs gcc, objdump and, for tasks drawn with cc=clang, clang."""
    if a.no_survive:
        return None
    from . import survive
    need = {"gcc": "gcc", "mix": "gcc and clang", "clang": "clang"}.get(getattr(a, "cc", "mix"), "gcc and clang")
    missing = [x for x in need.split(" and ") if not survive.toolchain(x)] + ([] if survive.OBJDUMP else ["objdump"])
    if missing:
        raise SystemExit(f"the fact gate needs {', '.join(missing)} on PATH; install it, or pass --no-survive (tasks are then not checked "
                         f"against the compiled code and the survival rate of their asked constant is unknown)")
    return survive.Gate()


def cmd_gen(a) -> None:
    names = pick_families(a)
    stale = sorted(p.name for p in a.out.glob("*") if TASK_FILE.match(p.name)) if a.out.is_dir() else []
    if stale and not a.force:
        raise SystemExit(f"{a.out} already holds {len(stale)} task files ({stale[0]}, ...); use an empty directory or --force to replace them")
    for name in stale:
        (a.out / name).unlink()
    counts = families.split_counts(a.n, len(names))
    gate = make_gate(a)
    total = 0
    for nm, c in zip(names, counts):
        tasks = families.generate(nm, c, a.seed, a.cc, gate)
        write_tasks(tasks, a.out)
        total += len(tasks)
        print(f"  {nm}: {len(tasks)} tasks")
    print(f"{total} tasks -> {a.out}" + (f"   ({gate.summary()})" if gate else ""))


def cmd_check(a) -> None:
    from . import verify
    names = pick_families(a)
    ubsan = verify.ubsan_available() and not a.no_ubsan
    a.cc = "mix"
    gate = make_gate(a)
    print(f"checking with gcc{' + clang' if verify.CLANG else ' (clang not installed)'}, UBSan harness {'on' if ubsan else 'off'}, "
          f"fact gate {'off' if gate is None else 'on'}")
    bad = 0
    for nm, c in zip(names, families.split_counts(a.n, len(names))):
        problems = []
        tasks = families.generate(nm, c, a.seed, "mix", gate)
        problems += [f"{b} duplicates {first} (same program and question)" for first, b in dedupe.duplicates(tasks)]
        for t in tasks:
            problems += verify.check_task(t, ubsan, survive_check=gate is not None)
        print(f"  {nm}: {len(tasks)} tasks, {len(problems)} problems" + (f"; {sum(1 for t in tasks if t.facts)} with a fact checked at their own cc/opt" if gate else ""), flush=True)
        for pr in problems[:20]:
            print("    " + pr)
        bad += len(problems)
    if bad:
        raise SystemExit(f"{bad} problems")
    print("all tasks compile clean and their harnesses prove the truth")


def cmd_heldout(a) -> None:
    if a.write:
        print(f"{dedupe.write_heldout()} held-out keys -> {dedupe.HELDOUT_FILE}")
        return
    now = [line.split()[1] for line in dedupe.heldout_lines()]
    pinned = dedupe.load_heldout()
    gone = len(pinned - set(now))
    print(f"{len(pinned)} pinned keys, {len(set(now))} regenerated; {gone} pinned keys are no longer produced by seed {dedupe.HELDOUT_SEED}")
    if gone or len(pinned) != len(set(now)):
        raise SystemExit("the generator no longer reproduces the pinned held-out list; if the change is intended run: "
                         "python3 tools/rea-bench heldout --write (and say so in the commit)")


def cmd_stats(a) -> None:
    held = dedupe.heldout_keys()
    print(f"seed {a.seed}, n={a.n} per family; held-out list: {len(held)} keys ({dedupe.HELDOUT_N} per family at seed {dedupe.HELDOUT_SEED})")
    print(f"{'family':10s} {'bodies':>7s} {'programs':>8s} {'overlap':>8s}   (bodies: distinct C text; programs: distinct normalized body+question;"
          f" overlap: tasks whose program is held out)")
    for nm in pick_families(a, allow_small=True):
        tasks = families.generate(nm, a.n, a.seed, "mix")
        keys = [dedupe.task_key(t) for t in tasks]
        print(f"{nm:10s} {len({t.body for t in tasks}):7d} {len(set(keys)):8d} {sum(k in held for k in keys):8d}")


def cmd_survival(a) -> None:
    from . import survive
    ccs = [cc for cc in ("gcc", "clang") if survive.toolchain(cc)]
    print(f"fact survival per variant: share of (task, -O level) pairs in which the asked constant is in the compiled code; "
          f"{', '.join(ccs)}; n={a.n} tasks per family, seed {a.seed}, generated without the gate")
    print(f"{'family/variant':26s} {'tasks':>5s}" + "".join(f" {cc:>7s}" for cc in ccs) + "    by -O level (" + ", ".join(core.OPTS) + ") per compiler")
    for nm in pick_families(a, allow_small=True):
        tasks = [t for t in families.generate(nm, a.n, a.seed, "mix") if t.facts]
        for variant in dict.fromkeys(t.variant for t in tasks):
            group = [t for t in tasks if t.variant == variant]
            cells, detail = [], []
            for cc in ccs:
                per_opt = {o: [0, 0] for o in core.OPTS}
                for t in group:
                    for o in core.OPTS:
                        per_opt[o][0] += survive.check(t.facts, t.source(), cc, o)[0]
                        per_opt[o][1] += 1
                tot = sum(v[0] for v in per_opt.values()), sum(v[1] for v in per_opt.values())
                cells.append(f"{100 * tot[0] / tot[1]:6.1f}%")
                detail.append("/".join(f"{100 * per_opt[o][0] / per_opt[o][1]:.0f}" for o in core.OPTS))
            print(f"{nm + '/' + variant:26s} {len(group):5d} " + " ".join(cells) + "    " + "  ".join(f"{cc}: {d}" for cc, d in zip(ccs, detail)), flush=True)


def cmd_chance(a) -> None:
    from . import chance
    table = chance.estimate()
    if a.write:
        chance.write_table(table)
        print(f"{len(table)} chance values -> {chance.__file__}")
        return
    for k, v in sorted(table.items()):
        print(f"{k:28s} {v:7.4f}   {'(table: ' + str(chance.CHANCE.get(k)) + ')' if chance.CHANCE.get(k) != v else ''}")


def cmd_list(_a) -> None:
    for nm, m in families.modules().items():
        print(f"{nm:10s} {len(m.VARIANTS):2d} variants: {', '.join(m.VARIANTS)}")


def main(argv=None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0].startswith("--") and argv[0] != "--help":      # old gen_rea_tasks.py invocation
        argv = ["gen", "--family", "legacy", *argv]
    ap = argparse.ArgumentParser(prog="rea-bench", description="REA task generator with automatic ground truth")
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen", help="generate tasks")
    g.add_argument("--family", default="all", help="family name, comma list, or 'all'")
    g.add_argument("--n", type=int, default=200)
    g.add_argument("--seed", type=int, default=1)
    g.add_argument("--cc", choices=["gcc", "clang", "mix"], default="mix", help="compiler written to the task JSON (new families)")
    g.add_argument("--out", type=pathlib.Path, required=True)
    g.add_argument("--force", action="store_true", help="replace task files already in --out (default: refuse)")
    g.add_argument("--no-survive", action="store_true", help="skip the fact gate (needs gcc/clang + objdump): do not resample tasks whose "
                   "asked constant is missing from the compiled code")
    g.set_defaults(fn=cmd_gen)
    s = sub.add_parser("list", help="list families and variants")
    s.set_defaults(fn=cmd_list)
    c = sub.add_parser("check", help="compile and run the harness of generated tasks (any seed)")
    c.add_argument("--family", default="all")
    c.add_argument("--n", type=int, default=110)
    c.add_argument("--seed", type=int, default=1)
    c.add_argument("--no-ubsan", action="store_true")
    c.add_argument("--no-survive", action="store_true", help="do not check that the asked constant is in the compiled code at the task's cc/opt")
    c.set_defaults(fn=cmd_check)
    h = sub.add_parser("heldout", help="verify or regenerate the pinned held-out list")
    h.add_argument("--write", action="store_true")
    h.set_defaults(fn=cmd_heldout)
    sv = sub.add_parser("survival", help="share of tasks whose asked constant is still in the compiled code, per variant, gcc and clang")
    sv.add_argument("--family", default="all")
    sv.add_argument("--n", type=int, default=120, help="tasks PER FAMILY")
    sv.add_argument("--seed", type=int, default=5)
    sv.set_defaults(fn=cmd_survival)
    ch = sub.add_parser("chance", help="estimated chance level per variant (the `chance` field of the task JSON)")
    ch.add_argument("--write", action="store_true")
    ch.set_defaults(fn=cmd_chance)
    st = sub.add_parser("stats", help="distinct programs and held-out overlap of a generated seed")
    st.add_argument("--family", default="all")
    st.add_argument("--n", type=int, default=3000, help="tasks PER FAMILY")
    st.add_argument("--seed", type=int, default=7)
    st.set_defaults(fn=cmd_stats)
    a = ap.parse_args(argv)
    a.fn(a)
