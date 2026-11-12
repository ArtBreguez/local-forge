"""WRAPPER: `python3 tools/rea-bench kaggle-score` is `distill/score_replies.py` with the REA domain bound (`distill/domains/rea`). Stdlib only, unit-tested without Modal.

The scorer of the replies of a Kaggle sampling run moved to `distill/score_replies.py`, where it does not know the subject (the replies, the stats, the records and the checks are described there). This module
keeps the REA command line and the old names working: `score`, `run` and `main` call it with REA, `default_out` and `add_arguments` know REA's file names (`decompiled-<tag>.jsonl`, the default tasks file).

    python3 tools/rea-bench kaggle-score --replies replies.jsonl --model codeforge-27b-heretic-q4 \\
        [--tasks results/phase6/rea-data/decompiled-seed7-all.jsonl] [--max-traces 3] [--thinking] [--keep-unchecked] [--out sft-....jsonl] [--force]
"""
from __future__ import annotations

import argparse
import pathlib

from distill import score_replies as _generic
from distill.domains.rea import DOMAIN
from distill.score_replies import (REPLY_DROPS, check_run_info, families_report, family_table, projected_fit, projected_fit_line, read_replies, read_run_info,  # noqa: F401
                                   reply_of, token_stats)
from distill.traces import DEFAULT_MAX_TRACES

DEFAULT_TASKS = pathlib.Path("results/phase6/rea-data/decompiled-seed7-all.jsonl")


def score(replies_path, tasks_path, model: str, max_traces: int = DEFAULT_MAX_TRACES, thinking: bool = False, run_info: dict = None, keep_unchecked: bool = False) -> tuple:
    """(SFT records, stats dict) of a Kaggle replies.jsonl against the decompiled tasks: `distill.score_replies.score` with REA."""
    return _generic.score(replies_path, tasks_path, model, max_traces, thinking, run_info, keep_unchecked, domain=DOMAIN)


def default_out(tasks_path, model: str, thinking: bool) -> pathlib.Path:
    return _generic.default_out(tasks_path, model, thinking, DOMAIN.tasks_prefix)


def add_arguments(ap: argparse.ArgumentParser) -> None:
    _generic.add_arguments(ap, DEFAULT_TASKS)


def run(a: argparse.Namespace) -> None:
    _generic.run(a, DOMAIN)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="kaggle-score", description="SFT records and stats from the replies.jsonl of a Kaggle sampling run (teacher-reasoning mode, no Modal)")
    add_arguments(ap)
    run(ap.parse_args(argv))


if __name__ == "__main__":
    main()
