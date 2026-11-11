"""Score the replies of a Kaggle sampling run, without Modal (stdlib only; unit-tested on a small fixture).

The teacher-reasoning sample stage (`modal_rea_data.py --stage sample --reasoning`) asks a model k times per task on a rented GPU. A Kaggle notebook can ask the same questions for free; this module
turns what it wrote into the SAME SFT records and stats, through the SAME functions the Modal stage ends in: the reply text is built by `reasoning.reply_text` (think blocks stripped, or kept as
the trace with `--thinking`; a reply the server cut off ends in the TRUNCATED marker), the acceptance is `reasoning.score_reasoning` (the last `Answer:` line, `score_strict`), the agreement rule, the
trace quality filters, the duplicate rule and the cap are `sft.assemble` -> `sft.select_traces` -> `reasoning.pick_traces`, and the messages are `reasoning.messages` (SYSTEM_REASONING + the user message),
the ones the sampler asks. Nothing is re-implemented here, so a record cannot differ from one the Modal stage would have made from the same replies.

    python3 tools/rea-bench kaggle-score --replies replies.jsonl --model codeforge-27b-heretic-q4 \\
        [--tasks results/phase6/rea-data/decompiled-seed7-all.jsonl] [--max-traces 3] [--thinking] [--out sft-....jsonl] [--force]

Inputs. `replies.jsonl`: one line per task, `{"id", "kind", "outs": [{"text", "finish", "tokens", "secs"}, ...]}`; `finish` "length" is a reply cut off at max_tokens (rejected), "error" or a text that starts with
`[error` is a failed request (never scored, counted); the last line of an id wins (a resumed run). The decompiled tasks (`--tasks`, a `decompiled-seed<seed>-<family>.jsonl` of the decompile stage) carry the
pseudocode, question and truth; a replies line whose id is not in it is counted and skipped, the held-out seed 1001 is refused. The notebook must have asked with `reasoning.SYSTEM_REASONING` and
`reasoning.user_message(task)`: the records are rebuilt with those messages, so a reply to another prompt would be stored under a prompt it never saw.

Outputs: the SFT records (`sft-<model>-<tag>-reasoning[-think].jsonl` next to the tasks file, or `--out`) and `<out>.stats.json`: the stats of `sft.assemble` (`reasoning` block with the drops by reason,
in total and per family) and `families`, per family and in total: tasks, answers, accepted, kept tasks, traces, `drops` by reason (every reply that is not a trace is under exactly one), truncated
share, mean and median tokens per reply (the `tokens` the notebook recorded). NOT applied: the over-length rule of the Modal stage (it needs the sampler's tokenizer); `modal_lora.py` drops a pair longer than
its window at training time, never cuts it.
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import re
import statistics
import sys

from . import dedupe, outputs, reasoning as rsn, sft
from .scoring import check_mode, check_target, is_error_reply

DEFAULT_TASKS = pathlib.Path("results/phase6/rea-data/decompiled-seed7-all.jsonl")
REPLY_DROPS = ("error", "truncated", "no_answer_line", "wrong")        # a reply that is not accepted: its verdict (reasoning.verdict)


def read_replies(path) -> tuple:
    """({task id: line}, {"lines", "malformed", "duplicate_ids"}) from a replies.jsonl. The last line of an id wins (a resumed notebook appends); a line that is not JSON (cut by a timeout), or
    has no `id` or `outs`, is counted and skipped."""
    out, info = {}, collections.Counter()
    for line in pathlib.Path(path).read_text().splitlines():
        if not line.strip():
            continue
        info["lines"] += 1
        try:
            rec = json.loads(line)
            assert isinstance(rec["outs"], list) and isinstance(rec["id"], str)
        except (json.JSONDecodeError, KeyError, AssertionError, TypeError):
            info["malformed"] += 1
            continue
        if rec["id"] in out:
            info["duplicate_ids"] += 1
        out[rec["id"]] = rec
    return out, {k: info[k] for k in ("lines", "malformed", "duplicate_ids")}


def reply_of(out, thinking: bool = False) -> str:
    """The text of one `outs` entry as the sampler's checkpoint would hold it (`reasoning.reply_text`). A failed request (`finish: "error"`, or a text that is the `[error ...]` of the sampler) becomes
    the sampler's `[error ...]` text. `reasoning_content`, when the notebook kept it apart from `text`, is the reasoning of a thinking run."""
    if not isinstance(out, dict):
        return f"[error malformed reply entry {type(out).__name__}]"
    text, finish = out.get("text") or "", out.get("finish")
    if finish == "error" or is_error_reply(text):
        return text if is_error_reply(text) else f"[error finish={finish!r}]"
    return rsn.reply_text(text, finish, out.get("reasoning_content") or "", thinking)


def token_stats(counts: list) -> dict:
    return {"n": len(counts), "mean": round(statistics.fmean(counts), 1) if counts else None, "median": statistics.median(counts) if counts else None}


def family_table(stats: dict, tokens: dict) -> dict:
    """{family or "ALL": row} from the stats of `sft.assemble` and the token counts per family. A row: tasks, answers, accepted, kept_tasks, traces, `drops` by reason (the replies that were not
    stored: failed, cut off, no final Answer line, wrong answer, then the trace filters, duplicates, over the cap, and the accepted replies of a task lost to the agreement rule),
    truncated_share, tokens {n, mean, median}. replies = traces + sum(drops)."""
    def row(block: dict, tasks: int, answers: int, kept_tasks: int, toks: list) -> dict:
        drops = {**{k: block[k] for k in REPLY_DROPS}, **block["dropped"], "in_lost_tasks": block["in_lost_tasks"]}
        return {"tasks": tasks, "answers": answers, "accepted": block["accepted"], "kept_tasks": kept_tasks, "traces": block["teacher_records"], "drops": drops,
                "truncated_share": round(block["truncated"] / answers, 4) if answers else 0.0, "tokens": token_stats(toks)}
    r = stats["reasoning"]
    out = {"ALL": row(r, stats["tasks"], stats["answers"], stats["kept_tasks"], [c for v in tokens.values() for c in v])}
    for kind, block in r["by_kind"].items():
        s = stats["by_kind"][kind]
        out[kind] = row(block, s["tasks"], s["answers"], s["kept_tasks"], tokens.get(kind, []))
    return out


def score(replies_path, tasks_path, model: str, max_traces: int = rsn.DEFAULT_MAX_TRACES, thinking: bool = False) -> tuple:
    """(SFT records, stats dict) of a Kaggle replies.jsonl against the decompiled tasks. Records and stats are `sft.assemble`'s, the same call as the Modal sample stage (`reasoning=True`); the stats
    also hold `families` (family_table), `kaggle` (what was read: lines, malformed, duplicate ids, unknown ids, tasks without replies) and `thinking`, `max_traces`."""
    if max_traces < 1:
        raise SystemExit(f"--max-traces {max_traces}: at least 1 teacher record per kept task")
    seed = re.search(r"seed(\d+)", pathlib.Path(tasks_path).name)
    if seed:
        dedupe.check_sample_seed(int(seed.group(1)))             # the held-out split is never turned into SFT pairs
    tasks = sorted(outputs.load_tasks([tasks_path]), key=lambda t: f"{t['id']}.json")       # the order the sample stage reads the volume in (sorted file names), so the records come out in its order
    if not tasks:
        raise SystemExit(f"no decompiled task (a record with pseudocode) in {tasks_path}")
    replies, info = read_replies(replies_path)
    known = {t["id"] for t in tasks}
    unknown = sorted(set(replies) - known)
    if unknown:
        sys.stderr.write(f"warning: {len(unknown)} replies line(s) have an id that is not in {pathlib.Path(tasks_path).name} and were skipped: {', '.join(unknown[:5])}{' ...' if len(unknown) > 5 else ''}\n")
    tasks = [t for t in tasks if t["id"] in replies]
    if not tasks:
        raise SystemExit(f"none of the {len(replies)} task ids of {replies_path} is in {tasks_path}: is it the file of the decompile run the notebook read?")
    if any(re.search(r"-1001-\d+$", t["id"]) for t in tasks):
        dedupe.check_sample_seed(dedupe.HELDOUT_SEED)
    for t in tasks:
        check_mode(t)
        check_target(t)                                           # as the sample stage does before it asks anything
    results, tokens = {}, collections.defaultdict(list)
    for t in tasks:
        outs = replies[t["id"]]["outs"]
        results[t["id"]] = {"outs": [[reply_of(o, thinking), False] for o in outs]}
        tokens[t["kind"]] += [o["tokens"] for o in outs if isinstance(o, dict) and isinstance(o.get("tokens"), (int, float)) and not isinstance(o.get("tokens"), bool)]
    kept, stats = sft.assemble(tasks, results, model, rsn.messages, True, thinking, max_traces)
    stats["families"] = family_table(stats, tokens)
    stats["thinking"] = thinking
    stats["kaggle"] = {**info, "unknown_ids": len(unknown), "tasks_in_file": len(known), "tasks_without_replies": len(known - set(replies)), "replies_file": pathlib.Path(replies_path).name,
                       "tasks_file": pathlib.Path(tasks_path).name}
    return kept, stats


def default_out(tasks_path, model: str, thinking: bool) -> pathlib.Path:
    """`sft-<model>-<tag>-reasoning[-think].jsonl` next to the tasks file, `<tag>` the one of `decompiled-<tag>.jsonl`: the name the Modal stage would have written."""
    tasks_path = pathlib.Path(tasks_path)
    m = re.fullmatch(r"decompiled-(.+)\.jsonl", tasks_path.name)
    tag = m.group(1) if m else tasks_path.stem
    return tasks_path.with_name(outputs.sft_stem(model, outputs.sample_tag(tag, True, thinking)) + ".jsonl")


def families_report(stats: dict) -> list:
    """The per-family lines printed after a run."""
    lines = [f"  {'family':10s} {'tasks':>5s} {'answers':>7s} {'accepted':>8s} {'kept':>5s} {'traces':>6s} {'cut off':>8s} {'tokens mean/median':>19s}  drops"]
    for kind, r in sorted(stats["families"].items(), key=lambda kv: (kv[0] != "ALL", kv[0])):
        tk = r["tokens"]
        drops = ", ".join(f"{k} {v}" for k, v in r["drops"].items() if v)
        toks = "-" if tk["mean"] is None else "%s/%s" % (tk["mean"], tk["median"])
        lines.append(f"  {kind:10s} {r['tasks']:5d} {r['answers']:7d} {r['accepted']:8d} {r['kept_tasks']:5d} {r['traces']:6d} {100 * r['truncated_share']:7.1f}% {toks:>19s}  {drops or '-'}")
    return lines


def add_arguments(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--replies", type=pathlib.Path, required=True, help="the Kaggle replies.jsonl: {id, kind, outs: [{text, finish, tokens, secs}]} per line")
    ap.add_argument("--tasks", type=pathlib.Path, default=DEFAULT_TASKS, help=f"the decompiled tasks the notebook asked (default {DEFAULT_TASKS})")
    ap.add_argument("--model", required=True, help="the model that wrote the replies: the `model` field of the records (and the file name)")
    ap.add_argument("--max-traces", type=int, default=rsn.DEFAULT_MAX_TRACES, help="teacher-reasoning records per kept task (default %(default)s)")
    ap.add_argument("--thinking", action="store_true", help="the replies hold the model's reasoning (<think>...</think>, or `reasoning_content`): keep it as the trace; files say -reasoning-think")
    ap.add_argument("--out", type=pathlib.Path, default=None, help="the SFT records (default: sft-<model>-<tag>-reasoning[-think].jsonl next to --tasks); the stats go to <out>.stats.json")
    ap.add_argument("--force", action="store_true", help="overwrite an existing output (default: refuse)")


def run(a: argparse.Namespace) -> None:
    out = a.out or default_out(a.tasks, a.model, a.thinking)
    stats_path = out.with_name(out.name[:-len(out.suffix)] + ".stats.json") if out.suffix else out.with_name(out.name + ".stats.json")
    outputs.refuse_overwrite([out, stats_path], a.force)
    kept, stats = score(a.replies, a.tasks, a.model, a.max_traces, a.thinking)
    outputs.write_atomic(out, (json.dumps(r) for r in kept))
    stats_path.write_text(json.dumps(stats, indent=1))
    s = stats
    print(f"{a.model} {pathlib.Path(out).name}: {s['tasks']} tasks, {s['answers']} replies, {s['kept_tasks']} kept tasks, {s['kept_records']} records "
          f"({s['reasoning']['teacher_records']} teacher-reasoning) -> {out}; stats -> {stats_path}")
    k = s["kaggle"]
    if k["malformed"] or k["unknown_ids"] or k["duplicate_ids"]:
        print(f"replies file: {k['lines']} lines, {k['malformed']} malformed, {k['duplicate_ids']} repeated ids (the last wins), {k['unknown_ids']} ids not in the tasks file")
    warning = sft.error_warning(s)
    if warning:
        print("\n" + "!" * 100 + f"\n{warning}\n" + "!" * 100 + "\n")
    for line in sft.reasoning_report(s):
        print(line)
    for line in families_report(s):
        print(line)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="kaggle-score", description="SFT records and stats from the replies.jsonl of a Kaggle sampling run (teacher-reasoning mode, no Modal)")
    add_arguments(ap)
    run(ap.parse_args(argv))


if __name__ == "__main__":
    main()
