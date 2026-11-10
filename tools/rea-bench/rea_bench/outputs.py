"""File names for the data pipeline's outputs, keyed by seed AND family, and the rule that nothing is overwritten without --force.

The tag of a run is `seed<seed>-<family>` (a family list joins with `+`: `seed7-crc+cipher`, `--family all` is `seed7-all`).
  results/phase6/rea-data/decompiled-<tag>.jsonl         (and /decompiled-<tag>/ on the Modal volume)
  results/phase6/rea-data/sft-<model>-<tag>.jsonl        (and .stats.json)
Before this a second family or seed overwrote the first (`sft-<model>.jsonl`, `decompiled-seed<seed>.jsonl`) and the decompile stage
wiped the previous run's volume directory.
"""
from __future__ import annotations

import collections
import json
import os
import pathlib
import re
import subprocess


def run_tag(seed: int, family: str) -> str:
    return f"seed{seed}-{re.sub(r'[^a-z0-9+]+', '+', family.lower()).strip('+')}"


def decompiled_stem(tag: str) -> str:
    return f"decompiled-{tag}"


def sft_stem(model: str, tag: str) -> str:
    return f"sft-{model}-{tag}"


def refuse_overwrite(paths, force: bool) -> None:
    """SystemExit naming the files that already exist, unless `force`."""
    have = [str(p) for p in paths if pathlib.Path(p).exists()]
    if have and not force:
        raise SystemExit("refusing to overwrite " + ", ".join(have) + " (pass --force, or use another --seed/--family)")


def heldout_files(rea_data: pathlib.Path, seed: int = 1001) -> list:
    """Every decompiled held-out file: the old `decompiled-seed1001.jsonl` and each `decompiled-seed1001-<family>.jsonl`."""
    return sorted(rea_data.glob(f"decompiled-seed{seed}.jsonl")) + sorted(rea_data.glob(f"decompiled-seed{seed}-*.jsonl"))


def load_tasks(files) -> list:
    """Records with pseudocode from the given jsonl files; a task id that occurs twice (the same task in an `all` run and a single-family
    run) is kept once."""
    seen, out = set(), []
    for f in files:
        for line in pathlib.Path(f).read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if "pseudocode" in rec and rec["id"] not in seen:
                seen.add(rec["id"])
                out.append(rec)
    return out


# ---- robustness of the Modal stages (round 2, m6) ----------------------------------------------------------------------------------

_ABSENT = re.compile(r"not found|no such|does not exist|doesn't exist|no files|file ?not ?found|not exist", re.I)


def _listing_says_absent(volume: str, path: str, run) -> bool:
    """Decide from a listing, not from an error text: True only when `modal volume ls <volume> <parent>` succeeds, does not show the
    entry's name, and shows no truncation marker (a table cell cut with an ellipsis could hide the name)."""
    parent, _, name = path.rstrip("/").rpartition("/")
    r = run(["modal", "volume", "ls", volume, parent or "/"], capture_output=True, text=True)
    if r.returncode not in (0, None):
        return False
    listing = f"{getattr(r, 'stdout', '')}"
    return bool(name) and name not in listing and "\u2026" not in listing and "..." not in listing


def rm_volume_dir(volume: str, path: str, run=subprocess.run) -> None:
    """Remove `path` on a Modal volume before a decompile run writes there. A directory that is not there is fine (first run); anything
    else stops the run (fail closed): stale records of an older generator under the same tag would otherwise be sampled with the new ones.
    "Not there" is recognised from the error text of `rm` (several spellings, `FileNotFound` included) or, when that text is not one of
    them, from a successful `ls` of the parent directory that does not list the name: the CLI's message is not a contract (round 3, m8)."""
    r = run(["modal", "volume", "rm", "-r", volume, path], capture_output=True, text=True)
    if r.returncode in (0, None):
        return
    msg = f"{getattr(r, 'stdout', '')} {getattr(r, 'stderr', '')}"
    if _ABSENT.search(msg) or _listing_says_absent(volume, path, run):
        return
    raise SystemExit(f"could not clear {path} on the Modal volume {volume} (exit {r.returncode}): {msg.strip()[:300]}\n"
                     f"stale records of an earlier run would be sampled with the new ones; fix the volume or use another --seed/--family")


def as_record(task: dict, result) -> dict:
    """A decompile result as a record: an exception that `starmap(return_exceptions=True)` returned becomes an error record."""
    if isinstance(result, BaseException):
        return {**task, "error": f"decompile crashed: {type(result).__name__}: {result}"[:300]}
    return result


def error_reason(rec: dict) -> str:
    """A short reason class for an error record (fact gate, decompile timeout, ...), for the summary of a decompile run."""
    e = rec.get("error", "")
    if e.startswith("fact not in the compiled code"):
        return "fact gate (Modal compilers)"
    if e.startswith("gate failed"):
        return "gate could not run"
    if e == "no pseudocode":
        return "decompile timeout" if "[timeout" in rec.get("raw", "") else "decompile: no pseudocode"
    if e.endswith("not found"):
        return "symbol not found"
    if e.startswith(("decompile crashed", "decompile failed")):
        return "decompile crashed"
    return "other"


def summarize_errors(records) -> dict:
    """{reason: count} and {reason: {family kind: count}} over the error records of a decompile run."""
    by, kinds = collections.Counter(), collections.defaultdict(collections.Counter)
    for rec in records:
        if "error" in rec:
            why = error_reason(rec)
            by[why] += 1
            kinds[why][rec.get("kind", "?")] += 1
    return {"total": sum(by.values()), "by_reason": dict(by), "by_reason_kind": {w: dict(c) for w, c in kinds.items()}}


def write_atomic(path: pathlib.Path, lines) -> int:
    """Write the jsonl through `<path>.part` and move it into place at the end: a run that dies half way leaves no `path` for
    refuse_overwrite to block the next run with (the .part file is replaced by the next run)."""
    part = path.with_name(path.name + ".part")
    n = 0
    with open(part, "w") as out:
        for line in lines:
            out.write(line + "\n")
            n += 1
    os.replace(part, path)
    return n


def load_partial(path: pathlib.Path, resume: bool) -> dict:
    """{task id: record} of a checkpoint the sample stage wrote (one json line per finished task). Without `resume` an old checkpoint
    is moved aside, never reused: its replies come from an earlier run that may have used other tasks or another scorer."""
    path = pathlib.Path(path)
    if not path.exists():
        return {}
    if not resume:
        path.rename(path.with_name(path.name + f".old-{int(path.stat().st_mtime)}"))
        return {}
    out = {}
    for line in path.read_text().splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue                  # a line cut by a timeout
        out[rec["id"]] = rec
    return out


def heldout_summary(records) -> dict:
    """The held-out arm, kept apart from the fixed questions: per family (and in total) the strict score (headline only), the
    format-insensitive control (any line of the reply) and the error count. The gap between the two is formatting, not reading."""
    rows = collections.defaultdict(lambda: {"n": 0, "strict": 0, "control": 0, "errors": 0})
    for r in records:
        if r.get("arm") != "heldout":
            continue
        for row in (rows["ALL"], rows[r.get("kind", "?")]):
            row["n"] += 1
            row["strict"] += bool(r.get("pass"))
            row["control"] += bool(r.get("pass_any_line"))
            row["errors"] += "error" in r
    return dict(rows)
