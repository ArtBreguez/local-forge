"""Score the replies of a sampling run that happened elsewhere (a Kaggle notebook, a local server), without Modal. Stdlib only, no subject in it (moved from `rea_bench/kaggle_score.py`).

The sample stage on Modal asks a teacher k times per task on a rented GPU. A Kaggle notebook can ask the same questions for free; this module turns what it wrote into the SAME SFT records and stats, through the
SAME functions the Modal stage ends in: the reply text is built by `traces.reply_text` (think blocks stripped, or kept as the trace with `--thinking`; a reply the server cut off ends in the TRUNCATED marker), the
acceptance is `traces.score_reasoning` (the last `Answer:` line, the domain's `verify`), the agreement rule, the trace quality filters, the duplicate rule and the cap are `records.assemble` ->
`records.select_traces` -> `traces.pick_traces`, and the messages are the domain's (`domain.messages(task, "reasoning" | "thinking")`), the ones the sampler asks. Nothing is re-implemented here, so a record cannot
differ from one the Modal stage would have made from the same replies.

    python3 -m distill.score_replies --domain rea --replies replies.jsonl --tasks tasks.jsonl --model codeforge-27b-heretic-q4 \\
        [--max-traces 3] [--thinking] [--keep-unchecked] [--out sft-....jsonl] [--force]        (REA: `python3 tools/rea-bench kaggle-score`, the same code)

Inputs. `replies.jsonl`: one line per task, `{"id", "kind", "outs": [{"text", "finish", "tokens", "secs"}, ...]}`; `finish` "length" is a reply cut off at max_tokens (rejected), "error" or a text that starts with
`[error` is a failed request (never scored, counted); the last line of an id wins (a resumed run). The tasks (`--tasks`, whatever `domain.load_tasks` reads) carry the question and the truth; a replies line whose id is
not among them is counted and skipped, and a task in the domain's held-out set is refused. The notebook must have asked with the domain's messages: the records are rebuilt with them, so a reply to another prompt would
be stored under a prompt it never saw.

`tools/kaggle/run_teacher.py` (`distill/backends/kaggle/`) is the notebook that writes `replies.jsonl` (README of `tools/kaggle`). It stores a thinking reply as `<think>\\nreasoning\\n</think>\\n\\nvisible` (what
`--thinking` expects), the server's `tokens` and `prompt_tokens` per reply, a `prompt_sha` per line and per reply (`outputs.prompt_sha` of the messages it sent) and a `run.json` next to the replies with the settings it
used. `--run-info run.json` (default: the `run.json` beside `--replies`, when there is one): its sampling settings go into the stats (`sampling`, the block the Modal stage writes), and `--thinking` must
agree with it (SystemExit otherwise: a thinking run scored as visible reasoning would drop the reasoning, and the other way round would keep none). A reply whose `prompt_sha` (its own, else its line's: `reply_fingerprints`)
is not the one of the messages rebuilt here is skipped and counted (`prompts`: lines that lost every reply in `mismatch`, a resumed line that lost some in `partial`, {id: replies}): a reply is never stored under a prompt it did not answer.
A reply with NO fingerprint (neither its own nor its line's: a file of an older notebook, the thinking pilot's `out2`) cannot be shown to have answered it: it is counted (`prompts.unchecked` lines, `unchecked_replies`,
`unchecked_ids`) and, by default, skipped like a mismatch; `--keep-unchecked` keeps them (the owner's word that the file was asked under this prompt), still counted, and the stats say so (`prompts.kept_unchecked`). `projected_fit` (stats and a printed line): of the teacher-reasoning records, how many fit a
training window of 4096 and 8192 tokens (a thinking run; 2048, 3072, 4096 otherwise), counted as the prompt tokens plus the completion tokens the server reported for the reply that became the record.

Outputs: the SFT records (`sft-<model>-<tag>-reasoning[-think].jsonl` next to the tasks file, or `--out`) and `<out>.stats.json`: the stats of `records.assemble` (`reasoning` block with the drops by reason,
in total and per family) and `families`, per family and in total: tasks, answers, accepted, kept tasks, traces, `drops` by reason (every reply that is not a trace is under exactly one), truncated
share, mean and median tokens per reply (the `tokens` the notebook recorded). NOT applied: the over-length rule of the Modal stage (it needs the sampler's tokenizer); training drops a pair longer than
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

from . import config, outputs, records, teachers, traces as tr
from .domain import mode_of
from .traces import is_error_reply

REPLY_DROPS = ("error", "truncated", "no_answer_line", "wrong")        # a reply that is not accepted: its verdict (reasoning.verdict)


def _count(v) -> bool:
    """A token count: a number that is not a bool."""
    return isinstance(v, (int, float)) and not isinstance(v, bool)


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
    return tr.reply_text(text, finish, out.get("reasoning_content") or "", thinking)


def reply_fingerprints(rec: dict) -> list:
    """One prompt fingerprint per reply of a replies line, None where there is none to compare: a reply's own `prompt_sha` (the kernel writes one with every reply, because a resumed line holds replies of several
    sessions), else the line's (an older kernel wrote only that one). A line without replies has its own fingerprint as the only one."""
    own = [(o.get("prompt_sha") if isinstance(o, dict) else None) or rec.get("prompt_sha") for o in rec["outs"]]
    return own or [rec.get("prompt_sha")]


def read_run_info(path):
    """The `run.json` of a notebook run (a dict), None when the file is not there; SystemExit when it is not a JSON object."""
    p = pathlib.Path(path)
    if not p.exists():
        return None
    try:
        info = json.loads(p.read_text())
        assert isinstance(info, dict)
    except (json.JSONDecodeError, AssertionError):
        raise SystemExit(f"{p} is not the JSON object run_teacher.py writes (run.json)")
    return info


def check_run_info(info, thinking: bool) -> None:
    """SystemExit when the notebook's own record of its run (`thinking`) is not what `--thinking` says."""
    if info and "thinking" in info and bool(info["thinking"]) != bool(thinking):
        raise SystemExit(f"the run asked with thinking={bool(info['thinking'])} (run.json), but kaggle-score was run {'with' if thinking else 'without'} --thinking: "
                         f"{'pass --thinking' if info['thinking'] else 'leave --thinking out'}")


def check_teacher(info, model: str) -> None:
    """SystemExit when the notebook's own record of the GGUF it asked (`model_repo` and `model_file` of run.json, when it has them) is a teacher whose text may not be stored: the rule of
    `teachers.require_reasoning` where the kit is built, applied again where the replies become records (round 12, M2: a replies file does not say who wrote it, and `--model` is only a label).
    The label is judged too: one that is a catalogue name must be the GGUF run.json names, and one that says gpt-oss or claude is refused. With both repo and file the GGUF must be a catalogue entry."""
    if info and (info.get("model_repo") or info.get("model_file")):
        repo, file = info.get("model_repo") or "", info.get("model_file") or ""
        teachers.require_reasoning(model, repo, file, strict=bool(repo and file), why="kaggle-score")


def projected_fit(kept: list, lengths: dict, windows: tuple) -> dict:
    """{"windows": {w: records that fit}, "records": teacher records, "measured": how many of them have a length, "basis"} for the teacher-reasoning records of `kept`. `lengths` maps (task id, record text)
    to the tokens of the reply that became the record: the server's prompt tokens (chat template, system + user, generation prompt) plus its completion tokens (reasoning, answer, end of turn). A record whose
    reply has no recorded length is not counted as fitting or not (`measured` says how many are)."""
    rows = [lengths.get((r["task"], r["messages"][-1]["content"])) for r in kept if r["target_source"] == tr.TARGET_SOURCE]
    known = [n for n in rows if n is not None]
    return {"windows": {str(w): sum(n <= w for n in known) for w in windows}, "records": len(rows), "measured": len(known),
            "basis": "prompt_tokens + completion_tokens of the server, per reply"}


def projected_fit_line(stats: dict):
    """The line printed about `projected_fit` (None when there is no teacher record)."""
    f = stats.get("projected_fit")
    if not f or not f["records"]:
        return None
    if not f["measured"]:
        return f"projected fit: the replies carry no token counts (prompt_tokens, tokens): {f['records']} teacher-reasoning records, none measured"
    parts = ", ".join(f"{w} tokens: {n} of {f['measured']} ({100 * n // f['measured']}%)" for w, n in f["windows"].items())
    return (f"projected fit of the teacher-reasoning records to a training window (prompt + reasoning + answer, the server's token counts): {parts}"
            + (f"; {f['records'] - f['measured']} records have no length" if f["measured"] != f["records"] else ""))


def token_stats(counts: list) -> dict:
    return {"n": len(counts), "mean": round(statistics.fmean(counts), 1) if counts else None, "median": statistics.median(counts) if counts else None}


def family_table(stats: dict, tokens: dict) -> dict:
    """{family or "ALL": row} from the stats of `records.assemble` and the token counts per family. A row: tasks, answers, accepted, kept_tasks, traces, `drops` by reason (the replies that were not
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


def score(replies_path, tasks_path, model: str, max_traces: int = tr.DEFAULT_MAX_TRACES, thinking: bool = False, run_info: dict = None, keep_unchecked: bool = False, *, domain) -> tuple:
    """(SFT records, stats dict) of a replies.jsonl against the tasks of `domain.load_tasks(tasks_path)`. Records and stats are `records.assemble`'s, the same call as the Modal sample stage (`reasoning=True`); the stats
    also hold `families` (family_table), `kaggle` (what was read: lines, malformed, duplicate ids, unknown ids, tasks without replies), `prompts` (lines whose prompt fingerprint is not the rebuilt messages', and
    the ones that have none to compare: `keep_unchecked` keeps the replies of the latter, else they are skipped), `projected_fit` and `thinking`, `max_traces`; with `run_info` (the notebook's run.json) also `sampling`
    and `run`."""
    if max_traces < 1:
        raise SystemExit(f"--max-traces {max_traces}: at least 1 teacher record per kept task")
    check_run_info(run_info, thinking)
    check_teacher(run_info, model)
    mode = mode_of(True, thinking)
    messages_for = lambda t: domain.messages(t, mode)
    tasks = domain.load_tasks(tasks_path)            # the order the sample stage reads them in, so the records come out in its order
    if not tasks:
        raise SystemExit(f"no task in {tasks_path}")
    replies, info = read_replies(replies_path)
    known = {t["id"] for t in tasks}
    unknown = sorted(set(replies) - known)
    if unknown:
        sys.stderr.write(f"warning: {len(unknown)} replies line(s) have an id that is not in {pathlib.Path(tasks_path).name} and were skipped: {', '.join(unknown[:5])}{' ...' if len(unknown) > 5 else ''}\n")
    tasks = [t for t in tasks if t["id"] in replies]
    if not tasks:
        raise SystemExit(f"none of the {len(replies)} task ids of {replies_path} is in {tasks_path}: is it the file of the task stage run the notebook read?")
    prompts = {"lines": len(tasks), "checked": 0, "mismatch": 0, "mismatched_ids": [], "unchecked": 0, "unchecked_replies": 0, "unchecked_ids": [], "kept_unchecked": bool(keep_unchecked)}
    own_outs = {}                                                 # id -> the replies of a line that answered the prompt rebuilt here, for a line that lost only some
    for t in list(tasks):
        rec = replies[t["id"]]
        fingerprints = reply_fingerprints(rec)
        want = outputs.prompt_sha(messages_for(t))
        wrong = [i for i, sha in enumerate(fingerprints) if sha is not None and sha != want]
        no_sha = [i for i, sha in enumerate(fingerprints) if sha is None]       # no fingerprint of their own, none on the line: nothing to compare
        prompts["checked"] += len(no_sha) < len(fingerprints)
        if no_sha:
            prompts["unchecked"] += 1
            prompts["unchecked_replies"] += len(no_sha)
            prompts["unchecked_ids"].append(t["id"])
        skipped = {*wrong, *([] if keep_unchecked else no_sha)}
        if not skipped:
            continue
        if len(skipped) == len(fingerprints):                     # every reply of the line (or the line, when it has none) answered another prompt or cannot be checked
            if wrong:
                prompts["mismatch"] += 1
                prompts["mismatched_ids"].append(t["id"])
            tasks.remove(t)
        else:                                                     # a resumed line mixes replies of several sessions: only the ones that answered another prompt, or cannot be checked, go
            own_outs[t["id"]] = [o for i, o in enumerate(rec["outs"]) if i not in skipped]
            if wrong:
                prompts.setdefault("partial", {})[t["id"]] = len(wrong)
    if prompts["mismatch"]:
        sys.stderr.write(f"warning: {prompts['mismatch']} replies line(s) answered a prompt other than the one rebuilt here (domain.messages) and were skipped: {', '.join(prompts['mismatched_ids'][:5])}"
                         f"{' ...' if prompts['mismatch'] > 5 else ''}\n")
    if prompts.get("partial"):
        sys.stderr.write(f"warning: {sum(prompts['partial'].values())} repl(ies) of {len(prompts['partial'])} replies line(s) answered a prompt other than the one rebuilt here and were skipped, the line's other replies were kept: "
                         f"{', '.join(list(prompts['partial'])[:5])}{' ...' if len(prompts['partial']) > 5 else ''}\n")
    if prompts["unchecked"] and not keep_unchecked:
        sys.stderr.write(f"warning: {prompts['unchecked']} replies line(s) hold {prompts['unchecked_replies']} repl(ies) with no prompt fingerprint (an older notebook): they cannot be shown to have answered the prompt rebuilt "
                         f"here and were skipped; --keep-unchecked keeps them\n")
    if not tasks:
        hint = (f" ({prompts['unchecked']} line(s) have no fingerprint at all, a file of an older notebook: --keep-unchecked scores them as they are, on your word that the notebook asked this prompt)"
                if prompts["unchecked"] and not keep_unchecked else "")
        raise SystemExit(f"no replies line of {replies_path} answered the prompt this version of the code asks: the notebook and score_replies disagree on reasoning.messages (domain.messages of {domain.name!r}){hint}")
    heldout = domain.heldout_ids()
    if any(t["id"] in heldout for t in tasks):
        raise SystemExit(f"held-out tasks of domain {domain.name!r} are never turned into SFT records: {', '.join(t['id'] for t in tasks if t['id'] in heldout)[:200]}")
    for t in tasks:
        domain.check_task(t)                                      # as the sample stage does before it asks anything
    results, tokens, lengths = {}, collections.defaultdict(list), {}
    for t in tasks:
        outs = own_outs.get(t["id"], replies[t["id"]]["outs"])
        results[t["id"]] = {"outs": [[reply_of(o, thinking), False] for o in outs]}
        tokens[t["kind"]] += [o["tokens"] for o in outs if isinstance(o, dict) and _count(o.get("tokens"))]
        for o, (reply, _flag) in zip(outs, results[t["id"]]["outs"]):
            if isinstance(o, dict) and _count(o.get("tokens")) and _count(o.get("prompt_tokens")) and tr.score_reasoning(reply, t, domain=domain):
                lengths[(t["id"], tr.trace_text(reply, thinking))] = o["prompt_tokens"] + o["tokens"]
    kept, stats = records.assemble(tasks, results, model, messages_for, True, thinking, max_traces, domain=domain)
    stats["prompts"] = prompts
    stats["projected_fit"] = projected_fit(kept, lengths, config.THINKING_REPORT_WINDOWS if thinking else config.REPORT_WINDOWS)
    by_family = (run_info or {}).get("max_tokens_by_family") or {}
    if run_info:
        stats["sampling"] = config.sampling_record(run_info.get("thinking", thinking), run_info.get("sampling") or {}, run_info.get("max_tokens"), seed_base=run_info.get("seed_base"),
                                                   slot_context=run_info.get("ctx_per_slot"), parallel=run_info.get("slots"), **({"max_tokens_by_family": by_family} if by_family else {}))
        stats["run"] = run_info
    stats["families"] = family_table(stats, tokens)
    if by_family:               # a family with a budget of its own: its cut-off share is read against that budget, so the row says which
        for kind, row in stats["families"].items():
            if kind != "ALL":
                row["max_tokens"] = config.task_max_tokens(kind, run_info.get("max_tokens"), by_family)
    stats["thinking"] = thinking
    stats["kaggle"] = {**info, "unknown_ids": len(unknown), "tasks_in_file": len(known), "tasks_without_replies": len(known - set(replies)), "replies_file": pathlib.Path(replies_path).name,
                       "tasks_file": pathlib.Path(tasks_path).name}
    return kept, stats


def default_out(tasks_path, model: str, thinking: bool, prefix: str = "") -> pathlib.Path:
    """`sft-<model>-<tag>-reasoning[-think].jsonl` next to the tasks file, `<tag>` the name of the tasks file without its `prefix` (`decompiled-` for REA: `decompiled-<tag>.jsonl`) and its suffix: the name the
    Modal stage would have written."""
    tasks_path = pathlib.Path(tasks_path)
    m = re.fullmatch(re.escape(prefix) + r"(.+)\.jsonl", tasks_path.name) if prefix else None
    tag = m.group(1) if m else tasks_path.stem
    return tasks_path.with_name(outputs.sft_stem(model, outputs.sample_tag(tag, True, thinking)) + ".jsonl")


def families_report(stats: dict) -> list:
    """The per-family lines printed after a run."""
    budgets = any("max_tokens" in r for r in stats["families"].values())          # a run with a budget per family says which budget each family had
    lines = [f"  {'family':10s} {'tasks':>5s} {'answers':>7s} {'accepted':>8s} {'kept':>5s} {'traces':>6s} {'cut off':>8s} {'tokens mean/median':>19s}" + (f" {'max_tokens':>10s}" if budgets else "") + "  drops"]
    for kind, r in sorted(stats["families"].items(), key=lambda kv: (kv[0] != "ALL", kv[0])):
        tk = r["tokens"]
        drops = ", ".join(f"{k} {v}" for k, v in r["drops"].items() if v)
        toks = "-" if tk["mean"] is None else "%s/%s" % (tk["mean"], tk["median"])
        lines.append(f"  {kind:10s} {r['tasks']:5d} {r['answers']:7d} {r['accepted']:8d} {r['kept_tasks']:5d} {r['traces']:6d} {100 * r['truncated_share']:7.1f}% {toks:>19s}"
                     + (f" {r.get('max_tokens', '-'):>10}" if budgets else "") + f"  {drops or '-'}")
    return lines


def add_arguments(ap: argparse.ArgumentParser, default_tasks=None) -> None:
    ap.add_argument("--replies", type=pathlib.Path, required=True, help="the replies.jsonl of the sampling run: {id, kind, outs: [{text, finish, tokens, secs}]} per line")
    ap.add_argument("--tasks", type=pathlib.Path, default=default_tasks, required=default_tasks is None, help="the tasks the notebook asked" + (f" (default {default_tasks})" if default_tasks else ""))
    ap.add_argument("--model", required=True, help="the model that wrote the replies: the `model` field of the records (and the file name)")
    ap.add_argument("--max-traces", type=int, default=tr.DEFAULT_MAX_TRACES, help="teacher-reasoning records per kept task (default %(default)s)")
    ap.add_argument("--thinking", action="store_true", help="the replies hold the model's reasoning (<think>...</think>, or `reasoning_content`): keep it as the trace; files say -reasoning-think")
    ap.add_argument("--out", type=pathlib.Path, default=None, help="the SFT records (default: sft-<model>-<tag>-reasoning[-think].jsonl next to --tasks); the stats go to <out>.stats.json")
    ap.add_argument("--run-info", type=pathlib.Path, default=None, help="the notebook's run.json (default: the one beside --replies, if there is one): its sampling settings go into the stats, and it must agree with --thinking")
    ap.add_argument("--keep-unchecked", action="store_true", help="keep the replies that carry no prompt fingerprint (a replies.jsonl of an older notebook, like the thinking pilot's out2): your word that they were asked "
                    "under this prompt. Default: they cannot be shown to have answered it, so they are skipped and counted (prompts.unchecked)")
    ap.add_argument("--force", action="store_true", help="overwrite an existing output (default: refuse)")


def run(a: argparse.Namespace, domain) -> None:
    out = a.out or default_out(a.tasks, a.model, a.thinking, domain.tasks_prefix)
    stats_path = out.with_name(out.name[:-len(out.suffix)] + ".stats.json") if out.suffix else out.with_name(out.name + ".stats.json")
    outputs.refuse_overwrite([out, stats_path], a.force)
    info_path = a.run_info or a.replies.with_name("run.json")
    if a.run_info and not a.run_info.exists():
        raise SystemExit(f"--run-info {a.run_info}: no such file")
    kept, stats = score(a.replies, a.tasks, a.model, a.max_traces, a.thinking, read_run_info(info_path), a.keep_unchecked, domain=domain)
    outputs.write_atomic(out, (json.dumps(r) for r in kept))
    stats_path.write_text(json.dumps(stats, indent=1))
    s = stats
    print(f"{a.model} {pathlib.Path(out).name}: {s['tasks']} tasks, {s['answers']} replies, {s['kept_tasks']} kept tasks, {s['kept_records']} records "
          f"({s['reasoning']['teacher_records']} teacher-reasoning) -> {out}; stats -> {stats_path}")
    k = s["kaggle"]
    if k["malformed"] or k["unknown_ids"] or k["duplicate_ids"]:
        print(f"replies file: {k['lines']} lines, {k['malformed']} malformed, {k['duplicate_ids']} repeated ids (the last wins), {k['unknown_ids']} ids not in the tasks file")
    warning = records.error_warning(s)
    if warning:
        print("\n" + "!" * 100 + f"\n{warning}\n" + "!" * 100 + "\n")
    if s["prompts"]["mismatch"]:
        print(f"prompts: {s['prompts']['mismatch']} of {s['prompts']['checked']} lines answered another prompt and were skipped")
    if s["prompts"]["unchecked"]:
        p = s["prompts"]
        print(f"prompts: {p['unchecked']} of {p['lines']} lines hold {p['unchecked_replies']} replies with no fingerprint (an older notebook), so unchecked: "
              + ("kept (--keep-unchecked), on your word that they answered this prompt" if p["kept_unchecked"] else "skipped (--keep-unchecked keeps them)"))
    if s["prompts"].get("partial"):
        print(f"prompts: {sum(s['prompts']['partial'].values())} replies of {len(s['prompts']['partial'])} lines answered another prompt and were skipped (the rest of those lines was kept)")
    for line in records.reasoning_report(s):
        print(line)
    if projected_fit_line(s):
        print(projected_fit_line(s))
    for line in families_report(s):
        print(line)


def main(argv=None) -> None:
    from .domains import load
    ap = argparse.ArgumentParser(prog="score_replies", description="SFT records and stats from the replies.jsonl of a sampling run (teacher-reasoning mode, no Modal)")
    ap.add_argument("--domain", default="rea", help="the subject: a package of distill/domains (default %(default)s)")
    add_arguments(ap)
    a = ap.parse_args(argv)
    run(a, load(a.domain))


if __name__ == "__main__":
    main()
