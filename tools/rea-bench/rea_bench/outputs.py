"""File names for the data pipeline's outputs, keyed by seed AND family, and the rule that nothing is overwritten without --force.

The tag of a run is `seed<seed>-<family>` (a family list joins with `+`: `seed7-crc+cipher`, `--family all` is `seed7-all`).
  results/phase6/rea-data/decompiled-<tag>.jsonl         (and /decompiled-<tag>/ on the Modal volume)
  results/phase6/rea-data/sft-<model>-<tag>.jsonl        (and .stats.json)
Before this a second family or seed overwrote the first (`sft-<model>.jsonl`, `decompiled-seed<seed>.jsonl`) and the decompile stage
wiped the previous run's volume directory.
"""
from __future__ import annotations

import json
import pathlib
import re


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
