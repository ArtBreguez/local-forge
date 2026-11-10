"""Task identity, duplicates inside a seed, and the pinned held-out list.

A task is identified by `task_key`: sha256 of its C `body` WITHOUT the decoy statements (core.decoys marks every one of them) and its
`question`, with the identifiers the generator invents (locals, fields, struct names) replaced by v0, v1, ... in order of first
appearance and whitespace collapsed. The compiler strips those names, and the decoys are unrelated volatile locals that never reach the
result, so two tasks that differ only in them are the same program to the model; keeping them apart let a held-out program into training
under a different set of decoys (round-2 review, M2). Library names (memset, uint8_t, ptrace, ...), keywords, numbers and string
literals stay as written, so every real difference (a constant, a form, an order) is a different key. `body_key` keeps the decoys: two
tasks of one seed with the same `body_key` are verbatim repeats (reported by `check`), the same `task_key` only the same program.

The held-out split is seed 1001. `heldout-1001.txt` pins the keys of its first HELDOUT_N tasks per family (a superset of every
standard held-out run: `--family all --n 330` uses 30 per family, one family at n=60 uses 60): the file records what was actually held
out even if a generator edit later changes seed 1001. The data pipeline drops every generated task whose key is in the list
(`modal_rea_data.py --exclude-seed 1001`, default for any other seed); `check` reports duplicates within the generated seed. The
exclusion also covers the later draws a fact gate may have kept in a held-out slot. Where a family has too few programs for that to be
enough (standard alphabets, standard CRC algorithms, ChaCha/Salsa round counts, return codes) the held-out seed draws from reserved
parameter values (core.pool), so its programs are not ones a training seed can make at all. A seed-1001 run with more than HELDOUT_N
tasks per family has tasks the list does not cover: `check_heldout_range` refuses it.

    python3 tools/rea-bench heldout --write      regenerate the list (after a generator change; commit the result)
    python3 tools/rea-bench heldout              verify the list against the generator
    python3 tools/rea-bench stats --seed 7 --n 3000   per family: distinct programs, overlap with the held-out list
"""
from __future__ import annotations

import functools
import hashlib
import pathlib
import re

from .core import HELDOUT_SEED, strip_decoys

HELDOUT_N = 120
HELDOUT_ATTEMPTS = 24     # a gated seed-1001 task may be a later draw of its slot (core.draw); all of these stay out of training
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
    """The program a task asks about: normalized body without decoys + question (what the held-out list and the exclusion use)."""
    return hashlib.sha256((normalize(strip_decoys(t.body)) + "\0" + t.question).encode()).hexdigest()


def body_key(t) -> str:
    """The task as generated: normalized body WITH its decoys + question (two tasks of one seed with the same key are verbatim repeats)."""
    return hashlib.sha256((normalize(t.body) + "\0" + t.question).encode()).hexdigest()


def raw_key(t) -> str:
    return hashlib.sha256((t.body + "\0" + t.question).encode()).hexdigest()


def duplicates(tasks) -> list:
    """[(first id, later id)] for tasks with the same body_key (verbatim repeats inside one seed)."""
    seen, out = {}, []
    for t in tasks:
        k = body_key(t)
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


@functools.lru_cache(maxsize=None)
def _held_out_draws(seed: int) -> frozenset:
    from . import families
    return frozenset(task_key(t) for fam in families.modules() for t in families.draws(fam, HELDOUT_N, seed, HELDOUT_ATTEMPTS))


def heldout_keys(seed: int = HELDOUT_SEED) -> set:
    """What a training run must avoid: every draw of the first HELDOUT_N tasks per family of `seed` (the first HELDOUT_ATTEMPTS attempts
    of each slot: which one the fact gate kept depends on the compilers that generated the held-out set, the candidates do not) and,
    for seed 1001, the pinned list."""
    return set(_held_out_draws(seed)) | (load_heldout() if seed == HELDOUT_SEED else set())


def exclusion_seed(seed: int, exclude_seed: int = -1):
    """The seed whose tasks `seed` must not contain: -1 (default) is the held-out seed for any other seed, 0 means none."""
    if exclude_seed < 0:
        return None if seed == HELDOUT_SEED else HELDOUT_SEED
    return exclude_seed or None


def check_heldout_range(names: list, n: int, seed: int) -> None:
    """SystemExit when a run of the held-out seed asks for more than HELDOUT_N tasks of one family: the pinned list and the exclusion
    cover the first HELDOUT_N slots only, so the rest would be tasks of the held-out seed that no training run knows to avoid (and
    that modal_rea.py would score as held-out)."""
    from . import families
    if seed != HELDOUT_SEED:
        return
    over = {nm: c for nm, c in zip(names, families.split_counts(n, len(names))) if c > HELDOUT_N}
    if over:
        raise SystemExit(f"seed {HELDOUT_SEED} is the held-out split and only its first {HELDOUT_N} tasks per family are listed and excluded from "
                         f"training: {', '.join(f'{nm} would get {c}' for nm, c in over.items())}. Use --n <= {HELDOUT_N * len(names)} for "
                         f"{len(names)} famil{'y' if len(names) == 1 else 'ies'}, or another --seed")


def family_of_kind(kind: str) -> str:
    """The family a task record belongs to: its `kind` for the new families, `legacy` for the ten original templates (one kind each)."""
    from . import families
    return kind if kind in families.modules() and kind != "legacy" else "legacy"


def heldout_overflow(tasks) -> dict:
    """{family: count} of the held-out records (dicts with `kind`) that exceed HELDOUT_N per family."""
    counts: dict = {}
    for t in tasks:
        fam = family_of_kind(t.get("kind", ""))
        counts[fam] = counts.get(fam, 0) + 1
    return {f: c for f, c in counts.items() if c > HELDOUT_N}


def training_tasks(names: list, n: int, seed: int, cc: str = "mix", exclude_seed: int = -1, gate=None) -> tuple:
    """(kept, dropped): n tasks over the named families without any task of the excluded (held-out) seed. `gate`: survive.Gate."""
    from . import families
    check_heldout_range(names, n, seed)
    tasks = families.generate_many(names, n, seed, cc, gate)
    ex = exclusion_seed(seed, exclude_seed)
    return drop_heldout(tasks, heldout_keys(ex)) if ex else (tasks, [])


def drop_heldout(tasks, keys: set) -> tuple:
    """(kept, dropped) - tasks whose key is held out go to `dropped`."""
    kept, dropped = [], []
    for t in tasks:
        (dropped if task_key(t) in keys else kept).append(t)
    return kept, dropped
