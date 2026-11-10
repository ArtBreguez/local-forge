"""Task identity, duplicates inside a seed, and the pinned held-out list.

A task is identified by `task_key`: sha256 of its C `body` and its `question`, with the identifiers the generator invents (locals,
fields, struct names) replaced by v0, v1, ... in order of first appearance and whitespace collapsed. The compiler strips those names,
so two tasks that differ only in them are the same program to the model; keeping them apart would let a renamed copy of a held-out
program into training. Library names (memset, uint8_t, ptrace, ...), keywords, numbers and string literals stay as written, so
every real difference (a constant, a form, an order) is a different key.

The held-out split is seed 1001. `heldout-1001.txt` pins the keys of its first HELDOUT_N tasks per family (a superset of every
standard held-out run: `--family all --n 330` uses 30 per family, one family at n=60 uses 60): the file records what was actually held
out even if a generator edit later changes seed 1001. The data pipeline drops every generated task whose key is in the list
(`modal_rea_data.py --exclude-seed 1001`, default for any other seed); `check` reports duplicates within the generated seed.

    python3 tools/rea-bench heldout --write      regenerate the list (after a generator change; commit the result)
    python3 tools/rea-bench heldout              verify the list against the generator
    python3 tools/rea-bench stats --seed 7 --n 3000   per family: distinct programs, overlap with the held-out list
"""
from __future__ import annotations

import hashlib
import pathlib
import re

HELDOUT_SEED = 1001
HELDOUT_N = 120
HELDOUT_FILE = pathlib.Path(__file__).with_name("heldout-1001.txt")

_TOKEN = re.compile(r'"(?:[^"\\]|\\.)*"|\'(?:[^\'\\]|\\.)+\'|[A-Za-z_]\w*|0[xX][0-9a-fA-F]+\w*|\d\w*|\S')
KNOWN = set("""
auto break case char const continue default do double else enum extern float for goto if inline int long register restrict return
short signed sizeof static struct switch typedef union unsigned void volatile while
uint8_t uint16_t uint32_t uint64_t int8_t int16_t int32_t int64_t size_t ssize_t intptr_t uintptr_t clockid_t time_t suseconds_t FILE va_list
proc helper helper2 NULL main
memcpy memset memcmp strlen strcmp strncmp strchr strstr strcpy atoi malloc calloc free puts printf snprintf sscanf fgets fopen fclose fmemopen
getenv setenv unsetenv ptrace syscall getpid clock_gettime gettimeofday va_start va_arg va_end offsetof
SYS_ptrace PTRACE_TRACEME CLOCK_MONOTONIC timespec timeval tv_sec tv_nsec tv_usec
""".split())


def normalize(body: str) -> str:
    """The body with invented identifiers replaced by v0, v1, ... and whitespace collapsed."""
    names: dict = {}
    out = []
    for tok in _TOKEN.findall(body):
        if re.fullmatch(r"[A-Za-z_]\w*", tok) and tok not in KNOWN:
            tok = names.setdefault(tok, f"v{len(names)}")
        out.append(tok)
    return " ".join(out)


def task_key(t) -> str:
    return hashlib.sha256((normalize(t.body) + "\0" + t.question).encode()).hexdigest()


def raw_key(t) -> str:
    return hashlib.sha256((t.body + "\0" + t.question).encode()).hexdigest()


def duplicates(tasks) -> list:
    """[(first id, later id)] for tasks with the same key."""
    seen, out = {}, []
    for t in tasks:
        k = task_key(t)
        if k in seen:
            out.append((seen[k], t.id))
        else:
            seen[k] = t.id
    return out


# ---- the held-out list ---------------------------------------------------------------------------------------------------

def heldout_lines() -> list:
    """`family key` lines for the first HELDOUT_N tasks of every family at seed 1001, as the generator makes them now (no
    compiler needed)."""
    from . import families
    return [f"{fam} {task_key(t)}" for fam in families.modules() for t in families.generate(fam, HELDOUT_N, HELDOUT_SEED)]


def load_heldout(path: pathlib.Path = HELDOUT_FILE) -> set:
    if not path.exists():
        return set()
    return {line.split()[1] for line in path.read_text().splitlines() if line.strip() and not line.startswith("#")}


def write_heldout(path: pathlib.Path = HELDOUT_FILE) -> int:
    lines = heldout_lines()
    path.write_text(f"# rea-bench held-out list: normalized (body, question) sha256 of the first {HELDOUT_N} tasks per family at seed "
                    f"{HELDOUT_SEED}. Regenerate: python3 tools/rea-bench heldout --write\n" + "\n".join(lines) + "\n")
    return len(lines)


def heldout_keys(seed: int = HELDOUT_SEED) -> set:
    """What a training run must avoid: for seed 1001 the pinned list plus the held-out tasks the generator makes today; for any other
    seed only what the generator makes today (first HELDOUT_N tasks per family)."""
    from . import families
    now = {task_key(t) for fam in families.modules() for t in families.generate(fam, HELDOUT_N, seed)}
    return now | load_heldout() if seed == HELDOUT_SEED else now


def exclusion_seed(seed: int, exclude_seed: int = -1):
    """The seed whose tasks `seed` must not contain: -1 (default) is the held-out seed for any other seed, 0 means none."""
    if exclude_seed < 0:
        return None if seed == HELDOUT_SEED else HELDOUT_SEED
    return exclude_seed or None


def training_tasks(names: list, n: int, seed: int, cc: str = "mix", exclude_seed: int = -1) -> tuple:
    """(kept, dropped): n tasks over the named families without any task of the excluded (held-out) seed."""
    from . import families
    tasks = families.generate_many(names, n, seed, cc)
    ex = exclusion_seed(seed, exclude_seed)
    return drop_heldout(tasks, heldout_keys(ex)) if ex else (tasks, [])


def drop_heldout(tasks, keys: set) -> tuple:
    """(kept, dropped) - tasks whose key is held out go to `dropped`."""
    kept, dropped = [], []
    for t in tasks:
        (dropped if task_key(t) in keys else kept).append(t)
    return kept, dropped
