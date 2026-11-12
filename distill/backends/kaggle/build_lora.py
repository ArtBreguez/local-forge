"""Build what a Kaggle LoRA + held-out evaluation run needs, for any subject: the training records, the held-out tasks (with their exact messages), the run's config, the dataset and kernel metadata and the
kernel's files; and print what the run will cost. The kernel is `lora_eval.py`; `score_lora.py` scores what it brings back. Nothing is pushed to Kaggle and nothing runs on a GPU.

    python3 -m distill.backends.kaggle.build_lora --domain rea --owner <kaggle-user> --out ~/work/kaggle-lora \\
        --sft results/phase6/rea-data/sft-<teacher>-<tag>-reasoning-think.jsonl --train-tasks results/phase6/rea-data/decompiled-seed7-all.jsonl \\
        --heldout results/phase6/rea-data/decompiled-seed1001-all.jsonl [--quota structs=10,crc=10] [--per-family 10] [--max-len 4096] [--mix] [--k 1] [--export-gguf] [--smoke] [--measured smoke-run.json]
        [--gpu-hours 8] [--min-family 5] [--strict-families] [--resume-from <user>/<kernel>] [--force]

    python3 -m distill.backends.kaggle.build_lora overlap --train-tasks <decompiled-seed7-all.jsonl> --heldout <decompiled-seed1001-all.jsonl>      # how much of the held-out file the training tasks cover

Writes under --out:
    data/train.jsonl             the SFT records, as given (checked: see below)
    data/eval_tasks.jsonl        {"id", "kind", "messages"} per held-out task: `domain.messages(task, "thinking")`, the prompt the training records have; nothing of the task's truth
    data/lora-config.json        the run (`lora_eval.DEFAULTS` with this build's values) and what it was built from (record counts, teacher labels, the held-out tasks and the ones that were
                                 excluded and why, the estimate)
    data/lora_eval.py            the kernel's code (`run.py` below imports it; the GPU workers start it by path), data/run_teacher.py (reused by it)
    data/distill.<module>.py     the stdlib-only core it runs (`distill/sft.py`, the Modal trainer's code, and the three modules it needs); flat, because the Kaggle CLI skips the sub-folders of a dataset
    data/dataset-metadata.json   for `kaggle datasets create -p <out>/data`
    kernel/run.py                a launcher of ten lines
    kernel/kernel-metadata.json  for `kaggle kernels push -p <out>/kernel` (private, GPU, internet on, the dataset attached; --resume-from also attaches an earlier run's output)

What decides which held-out tasks are asked. A held-out id that is not a training id says nothing about the CONTENT: the generator draws the same problem under two seeds, and the student that trained on one
would be tested on the other. A held-out task is therefore EXCLUDED from the evaluation (a training record never is) when EITHER rule matches a task of --train-tasks (ALL of its tasks, the ones that
made it into the SFT file or not; `domain.task_keys`):
  code      its normalised code and its question are a training task's (REA: the pseudocode with Ghidra's FUN_/DAT_/LAB_/PTR_/SUB_ addresses collapsed, whitespace collapsed),
  answer    its question and its answer key (kind, question, accepted answers) are a training task's: with the same question and answer the evaluation can be passed by memory.
The exclusion comes BEFORE the quota, so the quota fills from the tasks that remain. A family the exclusion left with fewer than --min-family (5) tasks, or none, is left out of the evaluation, with a printed
note (--strict-families: SystemExit). What was excluded, per family and rule, and the pairs, is in the config's `built_from.eval.excluded`; `overlap` prints the table without building anything.

What it REFUSES (SystemExit, before anything is written): a training record of a held-out task (`domain.heldout_ids`), a training record of a task that is in the evaluation subset or whose user message
is an evaluation task's (normalised: `domain.normalize_prompt`), a training record of a task that --train-tasks does not have, a held-out file that holds a task that is not held-out, a subject that gives no
content keys, a record with think text of a teacher that may only filter (`teachers.require_reasoning`: gpt-oss, claude, ... the rule of kaggle-score; whatever the record's `target_source`), a truth-template
record that has think text or whose text is not its task's `truth_target`, a file with no thinking record (this kit trains and tests the thinking format), and a build over an earlier one without --force.
The evaluation subset is `--quota kind=N,...`, by default `--per-family` (10) tasks of each family that the training records have, the first ones by id (`config.apply_quota`). The tasks are read by
`domain.load_heldout`. Stdlib only.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import pathlib
import re
import shutil
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
from distill import config, outputs, records, sft, teachers  # noqa: E402
from distill.domain import KEY_RULES  # noqa: E402
import lora_eval  # noqa: E402  (its DEFAULTS, LIB_MODULES and resolve_config: what the kernel is built to run)

SLUG = re.compile(r"[a-z0-9][a-z0-9-]*")
OWNER_KERNEL = re.compile(r"[a-z0-9-]+/[a-z0-9-]+")
LAUNCHER = '''"""Kaggle kernel: launches lora_eval.py of the attached dataset (distill/backends/kaggle/lora_eval.py). A script kernel ships one file, and the GPU workers start lora_eval.py by path."""
import pathlib
import sys

lib = next(pathlib.Path("/kaggle/input").rglob("lora_eval.py")).parent
sys.path.insert(0, str(lib))
import lora_eval  # noqa: E402

lora_eval.main()
'''

# What the estimate ASSUMES until a smoke run measures it (`--measured`). Nothing here has been measured on a T4; the two that matter most are the review's (PR 34, M3), not the first guesses.
RATES = {
    "train_tok_per_s_per_gpu": 150.0,       # QLoRA of a 9B on a T4: 100-180 tok/s expected. 6 x 9e9 FLOP per token at ~12 TFLOP/s of the T4's 65 is 220, but without flash-linear-attention the 24 linear-attention layers run a PyTorch fallback
    "prefill_tok_per_s_per_gpu": 450.0,     # forward only: three times the training rate
    "gen_step_s": 0.3,                      # one decode step of a batch of 8, HF generate on the 4-bit model: bitsandbytes' fused 4-bit gemv is for ONE token, a batch of 8 dequantizes and runs a GEMM (~34 GB of traffic per step, 0.14 s at the T4's ~250 GB/s), then the 248,320-logit head and sampling and HF's per-step Python: 0.2-0.3 s
    "setup_s": 1200.0,                      # pip install and the 18 GB download
    "load_s": 300.0,                        # one load + 4-bit quantization of the student (training and evaluation each load it)
    "export_s": 1800.0,                     # merge on the CPU, convert, quantize
    "typical_gen_share": 0.75,              # the longest reply of a batch is rarely the budget (the pilot cut every cipher reply off at 4096, so a batch of 8 usually has one that is)
}
DEADLINE_MARGIN_H = 0.25                    # what the kernel's deadline leaves of the quota that is left: the merge of the shards and run.json are written after the last batch
MIN_FAMILY = 5                              # a family that the exclusion leaves with fewer held-out tasks than this is left out of the evaluation
CHARS_PER_TOKEN = 3.0                       # no tokenizer on this machine: pseudocode and reasoning, ~3 characters per token (+-25%; the kernel counts the real tokens and prints them)


def _slug(what: str, value: str) -> str:
    if not SLUG.fullmatch(value or ""):
        raise SystemExit(f"{what} {value!r}: lower-case letters, digits and `-` (a Kaggle owner or slug)")
    return value


# ---- the records and the held-out tasks ------------------------------------------------------------------------------------------------------

def read_sft(path) -> list:
    """The records of an SFT jsonl (`kaggle-score`'s `sft-<model>-<tag>-reasoning-think.jsonl`), each with `records.RECORD_KEYS` and a system, user, assistant message. SystemExit otherwise."""
    rows = []
    for n, line in enumerate(pathlib.Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            raise SystemExit(f"{path}:{n} is not JSON")
    if not rows:
        raise SystemExit(f"no record in {path}")
    try:
        records.check_records(rows)
    except ValueError as e:
        raise SystemExit(f"{path}: {e}")
    for r in rows:
        m = r["messages"]
        if not (isinstance(m, list) and [x.get("role") for x in m if isinstance(x, dict)] == ["system", "user", "assistant"] and all(isinstance(x.get("content"), str) for x in m)):
            raise SystemExit(f"{path}: record of {r['task']} does not have a system, a user and an assistant message")
    return rows


def read_train_tasks(files, *, domain) -> dict:
    """{id: task} of EVERY task of the training task files (`domain.load_tasks`: the tasks the run was made from, not only the ones that became SFT records; REA: `decompiled-seed7-*.jsonl`, whose loader
    refuses the files of the held-out seed). SystemExit when one of them is a held-out id (the file would be the evaluation's own) or when there is no task."""
    tasks = domain.load_tasks([pathlib.Path(f) for f in files])
    heldout = domain.heldout_ids()
    bad = [t["id"] for t in tasks if t["id"] in heldout]
    if bad:
        raise SystemExit(f"--train-tasks holds held-out tasks of domain {domain.name!r} ({', '.join(bad)[:200]}): it must be the tasks the training records were made from")
    if not tasks:
        raise SystemExit(f"no task in --train-tasks {', '.join(map(str, files))}")
    return {t["id"]: t for t in tasks}


def key_index(tasks, *, domain) -> dict:
    """{rule: {key: id of the first task (by id) that has it}} for `tasks` (a dict or a list), by `domain.task_keys`."""
    index = {rule: {} for rule in KEY_RULES}
    for t in sorted(tasks.values() if isinstance(tasks, dict) else tasks, key=lambda t: t["id"]):
        for rule, key in domain.task_keys(t).items():
            if rule in index:
                index[rule].setdefault(key, t["id"])
    return index


def overlap(train: dict, heldout: list, *, domain) -> dict:
    """What the held-out tasks share with the training tasks `train` ({id: task}), by `domain.task_keys`: {"train_tasks", "heldout", "unkeyed" (held-out ids that have no key at all: nothing can be said about them),
    "matches": {held-out id: {rule: id of the training task it matches}} (only the tasks that match), "by_family": {kind: {"heldout", "code", "answer", "union"}}, "total": the same summed}. `code` and `answer`
    count the tasks that match under that rule, `union` the tasks that match under either (the ones the builder excludes). SystemExit when the domain gives no key for any held-out task: the check
    would find nothing, whatever the data."""
    index = key_index(train, domain=domain)
    matches, unkeyed, fam = {}, [], collections.defaultdict(collections.Counter)
    for t in heldout:
        keys = {rule: key for rule, key in domain.task_keys(t).items() if rule in index}
        if not keys:
            unkeyed.append(t["id"])
        hit = {rule: index[rule][key] for rule, key in keys.items() if key in index[rule]}
        if hit:
            matches[t["id"]] = hit
        c = fam[t["kind"]]
        c["heldout"] += 1
        for rule in hit:
            c[rule] += 1
        c["union"] += bool(hit)
    if len(unkeyed) == len(heldout):
        raise SystemExit(f"domain {domain.name!r} gives no content key (`Domain.task_keys`) for any held-out task: a held-out task cannot be told from a training task by anything but its id")
    cols = ("heldout", "code", "answer", "union")
    by_family = {kind: {k: c[k] for k in cols} for kind, c in sorted(fam.items())}
    return {"train_tasks": len(train), "heldout": len(heldout), "unkeyed": sorted(unkeyed), "matches": matches, "by_family": by_family, "total": {k: sum(v[k] for v in by_family.values()) for k in cols}}


def overlap_lines(report: dict) -> list:
    """The per-family table of an `overlap` report: held-out tasks, how many match a training task by code+question, by question+answer, either (the union: what the builder excludes), and how many remain."""
    out = [f"{'family':<12}{'held-out':>9}{'code+question':>15}{'question+answer':>17}{'either':>8}{'left':>6}"]
    for kind, c in [*report["by_family"].items(), ("TOTAL", report["total"])]:
        out.append(f"{kind:<12}{c['heldout']:>9}{c['code']:>15}{c['answer']:>17}{c['union']:>8}{c['heldout'] - c['union']:>6}")
    out.append(f"{report['train_tasks']} training tasks compared; a held-out task is excluded when its code+question OR its question+answer matches one of them"
               + (f"; {len(report['unkeyed'])} held-out task(s) have no key at all and are not compared" if report["unkeyed"] else ""))
    return out


def pick_eval(files, quota: str, per_family: int, families, *, domain, train_tasks: dict, rules=KEY_RULES, min_family: int = 5, strict_families: bool = False) -> tuple:
    """(the held-out tasks the evaluation asks, what was excluded). `domain.load_heldout(files)`, every one of them a held-out id (SystemExit otherwise: a file of tasks that a training set may hold would make the
    evaluation meaningless); then every task that matches a task of `train_tasks` under one of `rules` is EXCLUDED (`overlap`); then `--quota` (`kind=N,...`) or `per_family` of each of `families` (the kinds of the
    training records), the first ones by id round robin (`config.apply_quota`) of the tasks that REMAIN, each checked by `domain.check_task`. A family that the exclusion touched and left with fewer than
    `min_family` tasks (or none) is left out of the evaluation with a printed note, or SystemExit with `strict_families`. SystemExit for no task at all; a note for a family that has fewer than asked.
    The second value: {"rules", "min_family", "train_tasks", "by_family": {kind: {"heldout", "code", "answer", "union", "excluded", "left", "chosen"}}, "total", "pairs": {id: {rule: training id}} of the excluded
    tasks, "left_out": {kind: why}, "unkeyed"}."""
    tasks = domain.load_heldout([pathlib.Path(f) for f in files])
    heldout = domain.heldout_ids()
    leaked = [t["id"] for t in tasks if t["id"] not in heldout]
    if leaked:
        raise SystemExit(f"--heldout holds tasks that are not held-out ids of domain {domain.name!r} ({', '.join(leaked)[:200]}): the evaluation would be on tasks that a training set may hold")
    report = overlap(train_tasks, tasks, domain=domain)
    pairs = {hid: {r: tid for r, tid in m.items() if r in rules} for hid, m in report["matches"].items() if set(m) & set(rules)}
    left = [t for t in tasks if t["id"] not in pairs]
    all_n, gone_n, left_n = (collections.Counter(t["kind"] for t in group) for group in (tasks, [t for t in tasks if t["id"] in pairs], left))
    want = config.parse_quota(quota, domain.kinds()) if quota else {kind: per_family for kind in sorted(families)}
    left_out = {kind: f"{left_n[kind]} of {all_n[kind]} held-out tasks remain after excluding {gone_n[kind]} that share content with a training task (fewer than {min_family})"
                for kind in want if gone_n[kind] and left_n[kind] < min_family}
    if left_out and strict_families:
        raise SystemExit("--strict-families: " + "; ".join(f"{kind}: {why}" for kind, why in left_out.items()))
    for kind, why in left_out.items():
        print(f"note: family {kind} is left out of the evaluation: {why}")
    want = {kind: n for kind, n in want.items() if kind not in left_out}
    chosen = config.apply_quota(left, want) if want else []          # `apply_quota` with no quota means every task: with every family left out, none
    have = collections.Counter(t["kind"] for t in chosen)
    short = [f"{kind} {have[kind]}/{n}" for kind, n in want.items() if have[kind] < n]
    if not chosen:
        raise SystemExit(f"no held-out task of {', '.join(sorted({*want, *left_out}))} in {', '.join(map(str, files))}" + (f" after excluding {len(pairs)} that share content with a training task" if pairs else ""))
    if short:
        print(f"note: fewer held-out tasks than asked for: {', '.join(short)}")
    for t in chosen:
        domain.check_task(t)
    by_family = {kind: {**c, "excluded": gone_n[kind], "left": left_n[kind], "chosen": have[kind]} for kind, c in report["by_family"].items()}
    info = {"rules": list(rules), "min_family": min_family, "train_tasks": report["train_tasks"], "by_family": by_family, "total": {**report["total"], "excluded": len(pairs), "chosen": len(chosen)},
            "pairs": dict(sorted(pairs.items())), "left_out": left_out, "unkeyed": report["unkeyed"]}
    return chosen, info


def check_training(rows: list, eval_tasks: list, *, domain, mix: bool = False, train_tasks: dict = None) -> dict:
    """SystemExit unless the records may be trained on and evaluated against `eval_tasks`: none of a held-out task, none of a task of the evaluation subset, none whose user message is an evaluation task's
    (both through `domain.normalize_prompt`: the check does not depend on `train_tasks`), every record that holds a teacher's text (a teacher-reasoning record, or ANY record with think text, whatever its
    `target_source`) of a teacher that may store it (`teachers.require_reasoning`, the rule `kaggle-score` applies to its `--model`), every truth-template record plain (no think tag: it is rendered from the
    truth), and at least one thinking record. With `train_tasks` ({id: task}, `read_train_tasks`): every record's task is in it and every truth-template record's text IS `domain.truth_target` of its task.
    Returns {"records", "by_source", "labels", "families", "fates"}."""
    heldout, tasks = domain.heldout_ids(), sorted({r["task"] for r in rows})
    bad = [t for t in tasks if t in heldout]
    if bad:
        raise SystemExit(f"held-out tasks of domain {domain.name!r} are in the training records, and are never trained on: {', '.join(bad)[:200]}")
    missing = [t for t in tasks if train_tasks is not None and t not in train_tasks]
    if missing:
        raise SystemExit(f"{len(missing)} training record task(s) are not in --train-tasks ({', '.join(missing)[:200]}): it is not the tasks file the records were made from")
    overlap_ids = sorted(set(tasks) & {t["id"] for t in eval_tasks})
    if overlap_ids:
        raise SystemExit(f"{len(overlap_ids)} training record task(s) are in the evaluation subset ({', '.join(overlap_ids)[:200]}): the student would be tested on what it was trained on")
    asked = {domain.normalize_prompt(domain.messages(t, "thinking")[-1]["content"]): t["id"] for t in eval_tasks}
    same = sorted({f"{r['task']} = {asked[key]}" for r in rows if (key := domain.normalize_prompt(r["messages"][1]["content"])) in asked})
    if same:
        raise SystemExit(f"{len(same)} training record(s) whose user message is an evaluation task's ({'; '.join(same)[:300]}): the student would be asked a prompt it trained on")
    texts = [r["messages"][-1]["content"] for r in rows]
    thinks = [sft.parse_assistant(t)[0] != "plain" for t in texts]
    labels = sorted({r["model"] for r, think in zip(rows, thinks) if r["target_source"] == records.TEACHER_SOURCE or think})
    for label in labels:
        teachers.require_reasoning(label, why="the Kaggle LoRA kit (it trains on the teacher's stored text)")
    mislabelled = sorted({r["task"] for r, think in zip(rows, thinks) if think and r["target_source"] == records.TRUTH_SOURCE})
    if mislabelled:
        raise SystemExit(f"{len(mislabelled)} {records.TRUTH_SOURCE} record(s) hold think text ({', '.join(mislabelled)[:200]}): a truth template is rendered from the truth and has no `<think>` tag, so this text is a model's under another label")
    if train_tasks is not None:
        off = sorted({r["task"] for r in rows if r["target_source"] == records.TRUTH_SOURCE and r["messages"][-1]["content"] != domain.truth_target(train_tasks[r["task"]])})
        if off:
            raise SystemExit(f"{len(off)} {records.TRUTH_SOURCE} record(s) whose text is not `domain.truth_target` of their task ({', '.join(off)[:200]}): this text did not come from the truth")
    if not sft.is_thinking_config(texts):
        raise SystemExit("no thinking record (`<think>reasoning</think>` then an answer) in the SFT file: this kit trains and tests the thinking format; a plain file is `tools/modal/modal_lora.py`'s")
    fates = collections.Counter(sft.record_fate(t, True, mix) for t in texts)
    return {"records": len(rows), "by_source": dict(collections.Counter(r["target_source"] for r in rows)), "labels": labels, "families": sorted({r["kind"] for r in rows}), "fates": dict(fates)}


# ---- what it costs ---------------------------------------------------------------------------------------------------------------------------

def est_tokens(row_or_task: dict) -> int:
    """Tokens of a record or task by characters (`CHARS_PER_TOKEN`) plus the template's margin (`records.LENGTH_OVERHEAD`)."""
    return math.ceil(sum(len(m["content"]) for m in row_or_task["messages"]) / CHARS_PER_TOKEN) + records.LENGTH_OVERHEAD


def trainable_lengths(rows: list, max_len: int, mix: bool) -> list:
    """The estimated token lengths of the records the kernel would train on: thinking records (and the plain ones with `mix`) that fit `max_len`."""
    keep = ("think", "plain") if mix else ("think",)
    return [n for r in rows if sft.record_fate(r["messages"][-1]["content"], True, mix) in keep and (n := est_tokens(r)) <= max_len]


def rates_from(path) -> dict:
    """`RATES` with the numbers a smoke run measured (`measured` of its run.json): `train_tok_per_s_per_gpu` and `gen_step_s`; what the run did not measure stays assumed. SystemExit for a file that is
    not a run.json."""
    try:
        data = json.loads(pathlib.Path(path).read_text())
    except (OSError, json.JSONDecodeError):
        data = None
    measured = data.get("measured") if isinstance(data, dict) else None
    if not isinstance(measured, dict):
        raise SystemExit(f"{path} is not the run.json of a lora_eval kernel (a JSON object with a `measured` block)")
    return {**RATES, **{k: float(measured[k]) for k in ("train_tok_per_s_per_gpu", "gen_step_s") if isinstance(measured.get(k), (int, float)) and measured[k] > 0}}


def plan(lengths: list, eval_prompts: list, *, epochs: float, gpus: int, global_batch: int, train_hours: float, k: int, max_new_tokens: int, eval_batch: int, export: bool, rates: dict) -> dict:
    """What the run costs, in seconds, from the estimated token `lengths` of the training examples and of the evaluation prompts. Training: `steps_full` = the epochs asked, `step_s` = a step's tokens at
    `train_tok_per_s_per_gpu` over `gpus`; when `train_hours` binds, `max_steps` caps the run (0 = it does not). Evaluation: `k` replies per task and arm on `gpus` processes, batches of `eval_batch`; a batch costs its prefill and
    `max_new_tokens` decode steps at most (the UPPER bound, which every figure of the verdict uses) or `typical_gen_share` of them. Plus the setup, two model loads and the export when asked."""
    n, mean = len(lengths), sum(lengths) / len(lengths)
    per_epoch = math.ceil(n / global_batch)
    full = max(1, math.ceil(epochs * per_epoch))
    step_s = global_batch * mean / (gpus * rates["train_tok_per_s_per_gpu"])
    budget = max(1, int(train_hours * 3600 // step_s))
    steps = min(full, budget)
    items = len(eval_prompts) * k
    batches = math.ceil(math.ceil(items / gpus) / eval_batch)
    prefill = eval_batch * (sum(eval_prompts) / len(eval_prompts)) / rates["prefill_tok_per_s_per_gpu"]
    upper_batch, typical_batch = prefill + max_new_tokens * rates["gen_step_s"], prefill + rates["typical_gen_share"] * max_new_tokens * rates["gen_step_s"]
    overhead = rates["setup_s"] + 2 * rates["load_s"]
    train_s, eval_upper, eval_typical = steps * step_s, 2 * batches * upper_batch, 2 * batches * typical_batch
    export_s = rates["export_s"] if export else 0.0
    return {"examples": n, "mean_tokens": round(mean), "steps_per_epoch": per_epoch, "steps_full": full, "step_s": round(step_s, 1), "steps": steps, "epochs_effective": round(steps / per_epoch, 2),
            "max_steps": budget if budget < full else 0, "train_s": round(train_s), "eval_items_per_arm": items, "eval_batches_per_process": batches, "eval_upper_s": round(eval_upper),
            "eval_typical_s": round(eval_typical), "overhead_s": round(overhead), "export_s": round(export_s), "total_upper_s": round(overhead + train_s + eval_upper + export_s),
            "total_typical_s": round(overhead + train_s + eval_typical + export_s), "eval_reserve_s": round(eval_upper + rates["load_s"]), "export_reserve_s": round(export_s), "gpus": gpus,
            "global_batch": global_batch, "k": k, "max_new_tokens": max_new_tokens, "eval_batch": eval_batch, "rates": rates}


def smoke_plan(longest: int, *, gpus: int, steps: int, max_new_tokens: int, eval_batch: int, rates: dict) -> dict:
    """The cost of a smoke run: the setup, two loads, `steps` steps of two of the longest records per process, one batch of `eval_batch` replies per process and arm of `max_new_tokens` tokens (a batch of the
    size the full run uses, so that the `gen_step_s` it measures is the one the estimate needs)."""
    step_s = 2 * longest / rates["train_tok_per_s_per_gpu"]
    gen = 2 * (max_new_tokens * rates["gen_step_s"])
    total = rates["setup_s"] + 2 * rates["load_s"] + steps * step_s + gen
    return {"smoke": True, "steps": steps, "step_s": round(step_s, 1), "eval_batch": eval_batch, "total_upper_s": round(total), "total_typical_s": round(total), "gpus": gpus, "rates": rates}


def describe(p: dict, gpu_hours: float, deadline_h: float) -> list:
    """The lines the builder prints about a plan (`plan` or `smoke_plan`): what is assumed, what each stage costs, whether it fits `gpu_hours` (the quota that is left when the kernel starts, with the smoke run
    that has still to be run added: `p["smoke_s"]`), and what the kernel does at the deadline."""
    h = lambda s: f"{s / 3600:.1f} h"
    r = p["rates"]
    where = "assumed" if r == RATES else "assumed, with the numbers of --measured"
    share = f"{100 * r['typical_gen_share']:.0f}%"
    out = []
    if p.get("smoke"):
        out.append(f"smoke run: {p['steps']} steps of the longest records at {p['step_s']} s, one batch of {p['eval_batch']} replies per process and arm; {h(p['total_upper_s'])} in all ({where}); it measures what the full estimate assumes")
    else:
        out.append(f"training: {p['examples']} examples of ~{p['mean_tokens']} tokens (estimated), {p['steps_per_epoch']} steps per epoch, {p['steps_full']} steps for the epochs asked, {p['global_batch']} sequences per step on {p['gpus']} GPU(s): "
                   f"{p['step_s']} s per step at {r['train_tok_per_s_per_gpu']:.0f} tok/s per GPU ({where})")
        out.append(f"          -> {p['steps']} of the {p['steps_full']} steps = {p['epochs_effective']} epochs, {h(p['train_s'])}" + (f"; the training budget BINDS: --max-steps {p['max_steps']} in the config sizes the cosine schedule to these steps, "
                   "and a step slower than assumed stops the clock before the schedule ends (a higher last learning rate): measure it with the smoke run and --measured" if p["max_steps"] else "; the training budget does not bind"))
        out.append(f"eval:     {p['eval_items_per_arm']} replies per arm (k={p['k']}), batches of {p['eval_batch']}, {p['eval_batches_per_process']} per process and arm, up to {p['max_new_tokens']} new tokens at {r['gen_step_s']} s per step ({where}): "
                   f"{h(p['eval_upper_s'])} assuming every batch runs to the budget, ~{h(p['eval_typical_s'])} assuming {share} of it")
        out.append(f"overhead: {h(p['overhead_s'])} (pip install, download, two model loads)" + (f"; GGUF export {h(p['export_s'])}" if p["export_s"] else ""))
        if p.get("smoke_s"):
            out.append(f"smoke:    + {h(p['smoke_s'])} for the smoke run that comes first (README step 1), out of the same quota")
        elif p.get("smoke_done"):
            out.append("smoke:    the smoke run is done (--measured): --gpu-hours is the quota that is left after it")
    extra = p.get("smoke_s", 0)
    total, typical = p["total_upper_s"] + extra, p["total_typical_s"] + extra
    fits = total <= gpu_hours * 3600
    out.append(f"total:    {h(total)} assuming every batch of the evaluation runs to the token budget, ~{h(typical)} assuming {share} of it" + (" (smoke run included)" if extra else "") + f", of the {gpu_hours:g} h of quota left: "
               + ("it fits" if fits else "IT DOES NOT FIT: lower --train-hours, --k, --per-family or --max-new-tokens"))
    out.append(f"the kernel starts nothing after {deadline_h:g} h" + ("" if p.get("smoke") else f", and training stops early enough to leave {h(p['eval_reserve_s'])} for the evaluation"))
    return out


# ---- the files -------------------------------------------------------------------------------------------------------------------------------

def kernel_files() -> dict:
    """{name in the dataset: source file} of the code the kernel runs: itself, the teacher kernel it reuses, and the stdlib-only core as flat `distill.<module>.py`."""
    files = {"lora_eval.py": HERE / "lora_eval.py", "run_teacher.py": HERE / "run_teacher.py"}
    files.update({f"distill.{m}.py": ROOT / "distill" / f"{m}.py" for m in lora_eval.LIB_MODULES})
    return files


def make_config(*, facts: dict, eval_tasks: list, heldout_files, excluded: dict, p: dict, smoke: bool, **settings) -> dict:
    """The config.json of a run: `lora_eval.DEFAULTS` with the build's settings, validated by the kernel's own `resolve_config`, and what it was built from, the held-out tasks that were excluded included."""
    cfg = {**{k: v for k, v in lora_eval.DEFAULTS.items() if k != "pins"}, **settings, "smoke": smoke}
    if not smoke:
        cfg.update(max_steps=p["max_steps"], eval_reserve_s=p["eval_reserve_s"], export_reserve_s=p["export_reserve_s"])
    try:
        lora_eval.resolve_config(cfg)
    except ValueError as e:
        raise SystemExit(f"the run config is refused by the kernel: {e}")
    return {**cfg, "built_from": {"train": facts, "eval": {"tasks": len(eval_tasks), "by_kind": dict(sorted(collections.Counter(t["kind"] for t in eval_tasks).items())),
                                                           "files": [pathlib.Path(f).name for f in heldout_files], "excluded": excluded}, "estimate": {k: v for k, v in p.items() if k != "rates"}, "rates": p["rates"]}}


def build(sft_path, heldout_files, out, owner: str, *, domain, train_tasks, quota: str = "", per_family: int = 10, min_family: int = MIN_FAMILY, strict_families: bool = False, student: str = None, max_len: int = 4096, rank: int = None, epochs: float = None, lr: float = None,
          global_batch: int = None, mix: bool = False, gpus: int = None, k: int = 1, max_new_tokens: int = None, eval_batch: int = None, train_hours: float = None, deadline_h: float = None,
          gpu_hours: float = 8.0, export_gguf: bool = False, fla: bool = False, smoke: bool = False, measured=None, dataset_slug: str = None, kernel_slug: str = None, resume_from: str = "",
          force: bool = False) -> dict:
    """Write the files described above under `out`; returns the config. `train_tasks`: the training task files (ALL the tasks the records were made from). Refuses to overwrite a file that exists, unless `force`.
    The settings that are None take `lora_eval.DEFAULTS`. `gpu_hours` is the quota left when the kernel starts: the smoke run that has still to be run (no `measured`) is taken out of it for the deadline."""
    base = domain.name.replace("_", "-")
    owner, dataset_slug, kernel_slug = _slug("--owner", owner), _slug("--dataset-slug", dataset_slug or f"{base}-lora-data"), _slug("--kernel-slug", kernel_slug or f"{base}-lora-eval")
    if resume_from and not OWNER_KERNEL.fullmatch(resume_from):
        raise SystemExit(f"--resume-from {resume_from!r}: owner/kernel-slug of the earlier run")
    out = pathlib.Path(out)
    files = kernel_files()
    paths = {"train": out / "data" / "train.jsonl", "eval": out / "data" / "eval_tasks.jsonl", "config": out / "data" / "lora-config.json", "dataset": out / "data" / "dataset-metadata.json",
             "run": out / "kernel" / "run.py", "kernel": out / "kernel" / "kernel-metadata.json", **{name: out / "data" / name for name in files}}
    outputs.refuse_overwrite(list(paths.values()), force)
    train = read_train_tasks(train_tasks, domain=domain)
    rows = read_sft(sft_path)
    families = sorted({r["kind"] for r in rows})
    eval_tasks, excluded = pick_eval(heldout_files, quota, per_family, families, domain=domain, train_tasks=train, min_family=min_family, strict_families=strict_families)
    excluded["train_task_files"] = [pathlib.Path(f).name for f in train_tasks]
    facts = check_training(rows, eval_tasks, domain=domain, mix=mix, train_tasks=train)
    d = lora_eval.DEFAULTS
    given = {"student": student or d["student"], "max_len": max_len, "rank": rank or d["rank"], "epochs": epochs or d["epochs"], "lr": lr or d["lr"], "global_batch": global_batch or d["global_batch"], "mix": mix,
             "gpus": gpus or d["gpus"], "k": k, "max_new_tokens": max_new_tokens or d["max_new_tokens"], "eval_batch": eval_batch or d["eval_batch"], "train_hours": train_hours or d["train_hours"],
             "export_gguf": export_gguf, "fla": fla}
    lengths = trainable_lengths(rows, given["max_len"], mix)
    if not lengths:
        raise SystemExit(f"every thinking record is over --max-len {given['max_len']} tokens (estimated): nothing to train on")
    rates = rates_from(measured) if measured else RATES
    prompts = [est_tokens({"messages": domain.messages(t, "thinking")}) for t in eval_tasks]
    smoke_kw = dict(gpus=given["gpus"], steps=d["smoke_steps"], max_new_tokens=d["smoke_new_tokens"], eval_batch=given["eval_batch"], rates=rates)
    p = smoke_plan(max(lengths), **smoke_kw) if smoke else plan(lengths, prompts, epochs=given["epochs"], gpus=given["gpus"], global_batch=given["global_batch"], train_hours=given["train_hours"], k=k,
                                                                   max_new_tokens=given["max_new_tokens"], eval_batch=given["eval_batch"], export=export_gguf, rates=rates)
    if not smoke:
        p = {**p, "smoke_s": 0 if measured else smoke_plan(max(lengths), **smoke_kw)["total_upper_s"], "smoke_done": bool(measured)}
    limit_h = gpu_hours - p.get("smoke_s", 0) / 3600 - DEADLINE_MARGIN_H          # the quota is spent by the smoke run that comes first, and the kernel needs a margin to write its last files
    if limit_h <= 0 or (deadline_h is not None and deadline_h > limit_h):
        raise SystemExit(f"--gpu-hours {gpu_hours:g}: the kernel's deadline can be at most {max(limit_h, 0):.2f} h (the quota left, less {p.get('smoke_s', 0) / 3600:.1f} h for the smoke run that is still to come, less {DEADLINE_MARGIN_H} h)"
                         + ("" if deadline_h is None else f", and --deadline-h is {deadline_h:g}"))
    given["deadline_h"] = deadline_h or min(d["deadline_h"], math.floor(limit_h * 100) / 100)          # never later than the limit, in hundredths of an hour
    facts = {**facts, "files": [pathlib.Path(sft_path).name], "sha256": hashlib.sha256(pathlib.Path(sft_path).read_bytes()).hexdigest()[:16], "est_examples": len(lengths)}
    cfg = make_config(facts=facts, eval_tasks=eval_tasks, heldout_files=heldout_files, excluded=excluded, p=p, smoke=smoke, **given)
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    outputs.write_atomic(paths["train"], (json.dumps(r) for r in rows))
    outputs.write_atomic(paths["eval"], (json.dumps({"id": t["id"], "kind": t["kind"], "messages": domain.messages(t, "thinking")}) for t in eval_tasks))
    paths["config"].write_text(json.dumps(cfg, indent=1))
    paths["dataset"].write_text(json.dumps({"title": dataset_slug, "id": f"{owner}/{dataset_slug}", "licenses": [{"name": "other"}]}, indent=1))
    for name, src in files.items():
        shutil.copyfile(src, paths[name])
    paths["run"].write_text(LAUNCHER)
    paths["kernel"].write_text(json.dumps({"id": f"{owner}/{kernel_slug}", "title": kernel_slug, "code_file": "run.py", "language": "python", "kernel_type": "script", "is_private": True, "enable_gpu": True,
                                           "enable_internet": True, "dataset_sources": [f"{owner}/{dataset_slug}"], "competition_sources": [], "kernel_sources": [resume_from] if resume_from else []}, indent=1))
    cfg["plan_lines"] = describe(p, gpu_hours, cfg["deadline_h"])
    return cfg


def add_arguments(ap: argparse.ArgumentParser) -> None:
    d = lora_eval.DEFAULTS
    ap.add_argument("--sft", type=pathlib.Path, required=True, help="the SFT records to train on: `kaggle-score`'s sft-<teacher>-<tag>-reasoning-think.jsonl (teacher-reasoning and truth-template records)")
    ap.add_argument("--train-tasks", type=pathlib.Path, nargs="+", required=True, help="ALL the tasks the --sft records were made from (REA: decompiled-seed7-all.jsonl of the decompile stage), not only the ones that became records: a held-out task that shares its code+question or its question+answer with any of them is left out of the evaluation")
    ap.add_argument("--heldout", type=pathlib.Path, nargs="+", required=True, help="the decompiled HELD-OUT tasks the student is tested on (REA: decompiled-seed1001-all.jsonl of the decompile stage); read by domain.load_heldout")
    ap.add_argument("--owner", required=True, help="your Kaggle user name (the owner of the dataset and the kernel)")
    ap.add_argument("--out", type=pathlib.Path, required=True, help="directory: data/ (the dataset) and kernel/ are written under it")
    ap.add_argument("--quota", default="", help="kind=N,...: the held-out tasks of each listed family and none of another; default --per-family of each family the training records have")
    ap.add_argument("--per-family", type=int, default=10, help="held-out tasks per family when --quota is not given (default %(default)s)")
    ap.add_argument("--min-family", type=int, default=MIN_FAMILY, help="a family that the exclusion leaves with fewer held-out tasks than this (or none) is left out of the evaluation, with a note (default %(default)s)")
    ap.add_argument("--strict-families", action="store_true", help="refuse (SystemExit) instead of leaving out a family that the exclusion left under --min-family")
    ap.add_argument("--student", default=None, help=f"Hugging Face model to fine-tune (default {d['student']}, the one modal_train.py trains)")
    ap.add_argument("--max-len", type=int, default=d["max_len"], help="training window in tokens; a longer record is dropped, never cut (default %(default)s: 8192 does not fit a T4)")
    ap.add_argument("--rank", type=int, default=None, help=f"LoRA rank (default {d['rank']}; modal_train.py uses 32)")
    ap.add_argument("--epochs", type=float, default=None, help=f"epochs (default {d['epochs']}, as modal_train.py); --train-hours may cut it")
    ap.add_argument("--lr", type=float, default=None, help=f"learning rate (default {d['lr']}, as modal_train.py)")
    ap.add_argument("--global-batch", type=int, default=None, help=f"sequences per optimizer step (default {d['global_batch']}, as modal_train.py)")
    ap.add_argument("--mix", action="store_true", help="also train the truth-template records, with an empty think block (modal_train's --mix); default: the thinking records only")
    ap.add_argument("--gpus", type=int, default=None, help=f"processes/GPUs to train and evaluate on (default {d['gpus']}; the kernel uses fewer when it finds fewer)")
    ap.add_argument("--k", type=int, default=1, help="replies per held-out task and arm (default %(default)s; the evaluation's time grows with it)")
    ap.add_argument("--max-new-tokens", type=int, default=None, help=f"reply budget of the evaluation, reasoning included (default {d['max_new_tokens']}); a reply cut off at it is counted")
    ap.add_argument("--eval-batch", type=int, default=None, help=f"replies generated together on one GPU (default {d['eval_batch']})")
    ap.add_argument("--train-hours", type=float, default=None, help=f"stop training after this many hours (default {d['train_hours']}); the number of steps is capped to fit it")
    ap.add_argument("--deadline-h", type=float, default=None, help=f"hours after the start at which the kernel starts nothing (default min({d['deadline_h']}, --gpu-hours less the smoke run still to come less {DEADLINE_MARGIN_H} h): the weekly quota, not the 12 h session, is the limit)")
    ap.add_argument("--gpu-hours", type=float, default=8.0, help="GPU hours of quota left when the kernel starts, what the estimate and the deadline are judged against (default %(default)s). Without --measured the smoke run that comes first is taken out of it; with --measured it has been run, so give what is left after it")
    ap.add_argument("--export-gguf", action="store_true", help="also merge the adapter into fp16 weights and write a Q4_K_M GGUF to /kaggle/working/gguf (about 5.6 GB; not what was evaluated; not tried on Kaggle)")
    ap.add_argument("--fla", action="store_true", help="install flash-linear-attention in the kernel (fast kernels for the student's linear-attention layers; not tried on a T4)")
    ap.add_argument("--smoke", action="store_true", help="build the SMOKE run: 4 steps on the longest records and one batch of 8 replies per process, files named smoke-*; it measures the step time, the generation speed (at the full run's batch) and the peak memory")
    ap.add_argument("--measured", type=pathlib.Path, default=None, help="the run.json of a smoke run: its measured step time and generation speed replace the assumed ones in the estimate")
    ap.add_argument("--dataset-slug", default=None, help="default <domain>-lora-data")
    ap.add_argument("--kernel-slug", default=None, help="default <domain>-lora-eval")
    ap.add_argument("--resume-from", default="", help="owner/kernel-slug of an earlier run: its output (checkpoints, adapter, replies) is attached to this kernel and continued")
    ap.add_argument("--force", action="store_true", help="overwrite the files of an earlier build (default: refuse)")


def run(a: argparse.Namespace, domain) -> None:
    cfg = build(a.sft, a.heldout, a.out, a.owner, domain=domain, train_tasks=a.train_tasks, quota=a.quota, per_family=a.per_family, min_family=a.min_family, strict_families=a.strict_families, student=a.student, max_len=a.max_len, rank=a.rank, epochs=a.epochs, lr=a.lr,
                global_batch=a.global_batch, mix=a.mix, gpus=a.gpus, k=a.k, max_new_tokens=a.max_new_tokens, eval_batch=a.eval_batch, train_hours=a.train_hours, deadline_h=a.deadline_h,
                gpu_hours=a.gpu_hours, export_gguf=a.export_gguf, fla=a.fla, smoke=a.smoke, measured=a.measured, dataset_slug=a.dataset_slug, kernel_slug=a.kernel_slug, resume_from=a.resume_from, force=a.force)
    t = cfg["built_from"]
    print(f"{t['train']['records']} training records {t['train']['by_source']} (teacher labels {t['train']['labels']}; {t['train']['fates']}), {t['eval']['tasks']} held-out tasks {t['eval']['by_kind']}, "
          f"{'SMOKE run' if cfg['smoke'] else 'full run'} -> {a.out}")
    print(excluded_line(t["eval"]["excluded"]))
    print("\n".join(cfg["plan_lines"]))
    print(f"next: kaggle datasets create -p {a.out}/data   then   kaggle kernels push -p {a.out}/kernel   (set the kernel's accelerator to GPU T4 x2; distill/backends/kaggle/README.md)")


def excluded_line(info: dict) -> str:
    """What the exclusion did, for the build's output: how many held-out tasks share content with a training task, by rule, and the families that lost the most."""
    t, per = info["total"], ", ".join(f"{kind} {c['excluded']}/{c['heldout']}" for kind, c in info["by_family"].items() if c["excluded"])
    left_out = f"; left out of the evaluation: {', '.join(info['left_out'])}" if info["left_out"] else ""
    return (f"excluded from the evaluation: {t['excluded']} of {t['heldout']} held-out tasks share content with a task of {info['train_tasks']} training tasks (code+question {t['code']}, question+answer {t['answer']}, "
            f"either {t['union']}; rule {' OR '.join(info['rules'])})" + (f": {per}" if per else "") + left_out)


def overlap_main(argv: list) -> None:
    """`overlap`: the table of how much of the held-out files the training tasks cover, with nothing built."""
    from distill.domains import load
    ap = argparse.ArgumentParser(prog="build_lora overlap", description="Print, per family, how many held-out tasks share their code+question or their question+answer with a training task")
    ap.add_argument("--domain", default="rea", help="the subject: a package of distill/domains (default %(default)s)")
    ap.add_argument("--train-tasks", type=pathlib.Path, nargs="+", required=True, help="ALL the training tasks (REA: decompiled-seed7-all.jsonl)")
    ap.add_argument("--heldout", type=pathlib.Path, nargs="+", required=True, help="the held-out tasks (REA: decompiled-seed1001-all.jsonl)")
    a = ap.parse_args(argv)
    domain = load(a.domain)
    held = domain.load_heldout([pathlib.Path(f) for f in a.heldout])
    leaked = [t["id"] for t in held if t["id"] not in domain.heldout_ids()]
    if leaked:
        raise SystemExit(f"--heldout holds tasks that are not held-out ids of domain {domain.name!r} ({', '.join(leaked)[:200]})")
    print("\n".join(overlap_lines(overlap(read_train_tasks(a.train_tasks, domain=domain), held, domain=domain))))


def main(argv=None) -> None:
    from distill.domains import load
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["overlap"]:
        return overlap_main(argv[1:])
    ap = argparse.ArgumentParser(description="Build the Kaggle inputs of a LoRA training + held-out evaluation run (records, held-out tasks, config, metadata, kernel files) and print its cost "
                                             "(`build_lora overlap ...` instead prints how much of the held-out tasks the training tasks cover)")
    ap.add_argument("--domain", default="rea", help="the subject: a package of distill/domains (default %(default)s)")
    add_arguments(ap)
    a = ap.parse_args(argv)
    run(a, load(a.domain))


if __name__ == "__main__":
    main()
