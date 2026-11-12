"""What a run writes and checks: file names keyed by tag, the rule that nothing is overwritten without --force, the checkpoint of the sample stage, atomic writes, and the fingerprint of a prompt. Stdlib only.

The tag of a run names its files (REA: `seed<seed>-<family>`), and a reasoning run adds `-reasoning` (`-reasoning-think` with thinking) to its own:
  sft-<model>-<tag>.jsonl             the records (and .stats.json)
  sft-<model>-<tag>.partial.jsonl     the checkpoint of the sample stage, one line per finished task
Moved from `rea_bench/outputs.py` (which keeps the REA decompile stage's names and error summaries and re-exports these).
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib

from .traces import is_error_reply, is_truncated


REASONING_SUFFIX = "-reasoning"
THINKING_SUFFIX = "-reasoning-think"


def sample_tag(tag: str, reasoning: bool = False, thinking: bool = False) -> str:
    """The tag of a sample run's OWN files (checkpoint and sft output): the decompile tag, plus `-reasoning` for a teacher-reasoning run and `-reasoning-think` for a thinking run
    (`--reasoning --thinking`), so that none of them ever writes into, resumes or moves aside the checkpoint of another kind of run of the same seed and family. The decompiled tasks are
    always read by the decompile tag. A thinking run is a reasoning run (`thinking` alone is no tag: the stage refuses it)."""
    if thinking and not reasoning:
        raise ValueError("a thinking run is a reasoning run: --thinking needs --reasoning")
    return tag + (THINKING_SUFFIX if thinking else REASONING_SUFFIX) if reasoning else tag


def sft_stem(model: str, tag: str) -> str:
    return f"sft-{model}-{tag}"


def refuse_overwrite(paths, force: bool) -> None:
    """SystemExit naming the files that already exist, unless `force`."""
    have = [str(p) for p in paths if pathlib.Path(p).exists()]
    if have and not force:
        raise SystemExit("refusing to overwrite " + ", ".join(have) + " (pass --force, or use another --seed/--family)")


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


def move_aside(path: pathlib.Path) -> pathlib.Path:
    """Rename `path` to `<name>.old-<mtime>`, or `<name>.old-<mtime>-2`, `-3`, ... when that name is taken: two non-resume runs that each found a checkpoint in the
    same second used to share one `.old` name, and POSIX `rename` replaced the first with the second (round 8, m15)."""
    path = pathlib.Path(path)
    base = path.name + f".old-{int(path.stat().st_mtime)}"
    dest, n = path.with_name(base), 1
    while dest.exists():
        n += 1
        dest = path.with_name(f"{base}-{n}")
    path.rename(dest)
    return dest


def load_partial(path: pathlib.Path, resume: bool) -> dict:
    """{task id: record} of a checkpoint the sample stage wrote (one json line per finished task). Without `resume` an old checkpoint
    is moved aside, never reused: its replies come from an earlier run that may have used other tasks or another scorer."""
    path = pathlib.Path(path)
    if not path.exists():
        return {}
    if not resume:
        move_aside(path)
        return {}
    out = {}
    for line in path.read_text().splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue                  # a line cut by a timeout
        out[rec["id"]] = rec
    return out


def out_meta(out) -> dict:
    """What a reasoning run's checkpoint says about one reply beyond its text and flag: the third element of an out, `{"prompt_sha", "max_tokens"}` (the fingerprint of the messages it answered and the budget it
    was asked with); {} for the two-element outs of a default run and of an older checkpoint."""
    return out[2] if len(out) > 2 and isinstance(out[2], dict) else {}


def valid_budget(n) -> bool:
    """A token budget: an integer of at least 1 that is not a bool (anything else in a checkpoint is a hand edit: the budget is not known)."""
    return isinstance(n, int) and not isinstance(n, bool) and n >= 1


def assume_meta(rec: dict, prompt: str, budget: int) -> dict:
    """`rec` with the fingerprint and the budget that a reply of an older checkpoint lacks filled in: the owner's word (`trust_unfingerprinted`) that its replies were asked under `prompt` and `budget`. A reply
    that has its own keeps them; the others are written with the two fields from then on. Nothing is assumed unless the run was told to: without it such a reply is stale (`is_stale`)."""
    outs = []
    for o in rec["outs"]:
        meta = out_meta(o)
        meta = {"prompt_sha": meta.get("prompt_sha") or prompt, "max_tokens": meta["max_tokens"] if valid_budget(meta.get("max_tokens")) else budget}
        outs.append([o[0], o[1], meta, *o[3:]])
    return {**rec, "outs": outs}


def is_stale(out, prompt: str = None, budget: int = None) -> bool:
    """True when an answered reply no longer stands for its task. With `prompt` (the task's prompt now): it answered another prompt, or it has NO fingerprint (what it answered is not known: the two-element out
    of an older checkpoint). With `budget` (the task's now): it was cut off at `max_tokens` (it ends in the TRUNCATED marker) at a smaller budget, or at one it does not say. Without `prompt` and `budget`
    nothing is stale."""
    meta = out_meta(out)
    if prompt and meta.get("prompt_sha") != prompt:
        return True
    asked = meta.get("max_tokens")
    return bool(budget and is_truncated(out[0]) and (not valid_budget(asked) or asked < budget))


def current_outs(outs: list, prompt: str) -> list:
    """The replies of a checkpointed task that answered the task's prompt `prompt`: a reply to another prompt, or one that does not say which (an older checkpoint), is left out, so that it is never stored under
    this one."""
    return [o for o in outs if out_meta(o).get("prompt_sha") == prompt]


def samples_to_ask(rec, k: int, prompt: str = None, budget: int = None) -> list:
    """The sample indexes (0..k-1) of one task that a resumed sample stage must ask: all of them for a task that is not in the checkpoint (`rec` is None),
    otherwise the samples that failed (`[error ...]`, a dead server or a timeout), are missing, or are STALE (`is_stale`: a reply to another prompt than `prompt` or with none known, or one cut off at a smaller
    budget than the task's `budget` now or at an unknown one: a longer budget gives a different reply, not the same cut). A failed request is checkpointed like an answer so that the
    run can finish and count it, but it is not an answer: `--resume` is the remedy the failure warning prints, so it must ask again (round 4, M3)."""
    if rec is None:
        return list(range(k))
    outs = rec["outs"]
    return [i for i in range(k) if i >= len(outs) or is_error_reply(outs[i][0]) or is_stale(outs[i], prompt, budget)]


def prompt_sha(messages: list) -> str:
    """A short fingerprint of the messages a request carried. The Kaggle notebook writes it with every reply (it has no copy of this package) and `score_replies` compares it with the
    messages it rebuilds: a reply to another prompt is not stored under this one."""
    return hashlib.sha256(json.dumps(messages, sort_keys=True, ensure_ascii=True).encode()).hexdigest()[:16]
