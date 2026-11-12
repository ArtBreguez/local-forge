"""WRAPPER: `python3 tools/kaggle/build_tasks.py ...` is `distill/backends/kaggle/build_tasks.py` with the REA domain bound (`distill/domains/rea`): the tasks are the decompiled tasks of the Modal decompile stage,
the messages are REA's. Nothing changed in what it writes; the kernel script it copies is `distill/backends/kaggle/run_teacher.py`. See tools/kaggle/README.md.

    python3 tools/kaggle/build_tasks.py --tasks results/phase6/rea-data/decompiled-seed9-all.jsonl --owner <kaggle-user> --out ~/work/kaggle-rea \\
        --quota structs=220,crc=220,antidebug=220,bounds=220,fsm=220,pktsum=80,encoding=80 [--k 4] [--max-tokens 4096] [--no-thinking] [--force]
"""
from __future__ import annotations

import argparse
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from distill.backends.kaggle import build_tasks as _generic  # noqa: E402  (the module the generic command runs; it finds its sibling run_teacher.py itself)
from distill.domains.rea import DOMAIN  # noqa: E402

SLUG = _generic.SLUG
_slug = _generic._slug


def pick_tasks(files, quota: str = "") -> list:
    return _generic.pick_tasks(files, quota, domain=DOMAIN)


def make_config(tasks: list, files, quota: str, k: int, thinking: bool, max_tokens, sampling_overrides: dict, deadline_h: float, model_repo, model_file) -> dict:
    return _generic.make_config(tasks, files, quota, k, thinking, max_tokens, sampling_overrides, deadline_h, model_repo, model_file, domain=DOMAIN)


def build(files, out, owner: str, quota: str = "", k: int = 4, thinking: bool = True, max_tokens=None, sampling_overrides=None, deadline_h: float = None,
          dataset_slug: str = None, kernel_slug: str = None, resume_from: str = "", model_repo=None, model_file=None, force: bool = False, teacher: str = None) -> dict:
    return _generic.build(files, out, owner, quota, k, thinking, max_tokens, sampling_overrides, deadline_h, dataset_slug, kernel_slug, resume_from, model_repo, model_file, force, teacher, domain=DOMAIN)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Build the Kaggle inputs of a teacher run (tasks, config, metadata, kernel script)")
    _generic.add_arguments(ap, "decompiled-seed<seed>-<family>.jsonl of the decompile stage (never seed 1001)")
    _generic.run(ap.parse_args(argv), DOMAIN)


if __name__ == "__main__":
    main()
