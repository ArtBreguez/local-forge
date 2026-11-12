"""Build what a Kaggle LoRA + held-out evaluation run needs, for any subject: the training records, the held-out tasks (with their exact messages), the run's config, the dataset and kernel metadata and the
kernel's files; and print what the run will cost. The kernel is `lora_eval.py`; `score_lora.py` scores what it brings back. Nothing is pushed to Kaggle and nothing runs on a GPU.

    python3 -m distill.backends.kaggle.build_lora --domain rea --owner <kaggle-user> --out ~/work/kaggle-lora \\
        --sft results/phase6/rea-data/sft-<teacher>-<tag>-reasoning-think.jsonl --heldout results/phase6/rea-data/decompiled-seed1001-all.jsonl \\
        [--quota structs=10,crc=10] [--per-family 10] [--max-len 4096] [--mix] [--k 1] [--export-gguf] [--smoke] [--measured smoke-run.json] [--resume-from <user>/<kernel>] [--force]

Writes under --out:
    data/train.jsonl             the SFT records, as given (checked: see below)
    data/eval_tasks.jsonl        {"id", "kind", "messages"} per held-out task: `domain.messages(task, "thinking")`, the prompt the training records have; nothing of the task's truth
    data/lora-config.json        the run (`lora_eval.DEFAULTS` with this build's values) and what it was built from (record counts, teacher labels, the held-out tasks, the estimate)
    data/lora_eval.py            the kernel's code (`run.py` below imports it; the GPU workers start it by path), data/run_teacher.py (reused by it)
    data/distill.<module>.py     the stdlib-only core it runs (`distill/sft.py`, the Modal trainer's code, and the three modules it needs); flat, because the Kaggle CLI skips the sub-folders of a dataset
    data/dataset-metadata.json   for `kaggle datasets create -p <out>/data`
    kernel/run.py                a launcher of ten lines
    kernel/kernel-metadata.json  for `kaggle kernels push -p <out>/kernel` (private, GPU, internet on, the dataset attached; --resume-from also attaches an earlier run's output)

What it REFUSES (SystemExit, before anything is written): a training record of a held-out task (`domain.heldout_ids`), a training record of a task that is in the evaluation subset, a held-out file that holds
a task that is not held-out, a teacher-reasoning record whose label is a teacher that may only filter (`teachers.require_reasoning`: gpt-oss, claude, ... the rule of kaggle-score), a file with no thinking
record (this kit trains and tests the thinking format), and a build over an earlier one without --force. The evaluation subset is `--quota kind=N,...`, by default `--per-family` (10) tasks of each family
that the training records have, the first ones by id (`config.apply_quota`). The tasks are read by `domain.load_heldout`. Stdlib only.
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

# What the estimate ASSUMES until a smoke run measures it (`--measured`). Nothing here has been measured on a T4.
RATES = {
    "train_tok_per_s_per_gpu": 220.0,       # QLoRA of a 9B on a T4: 6 x 9e9 FLOP per token (forward, recompute, backward) at ~12 TFLOP/s effective of the T4's 65 (fp16 tensor cores, 4-bit dequantization, no fused kernels)
    "prefill_tok_per_s_per_gpu": 660.0,     # forward only: three times the training rate
    "gen_step_s": 0.12,                     # one decode step of a batch of 8, HF generate on the 4-bit model (dominated by per-step overhead, so it grows little with the batch)
    "setup_s": 1200.0,                      # pip install and the 18 GB download
    "load_s": 300.0,                        # one load + 4-bit quantization of the student (training and evaluation each load it)
    "export_s": 1800.0,                     # merge on the CPU, convert, quantize
    "typical_gen_share": 0.75,              # the longest reply of a batch is rarely the budget
}
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


def pick_eval(files, quota: str, per_family: int, families, *, domain) -> list:
    """The held-out tasks the evaluation asks: `domain.load_heldout(files)`, every one of them a held-out id (SystemExit otherwise: a file of tasks that a training set may hold would make the evaluation
    meaningless), then `--quota` (`kind=N,...`) or `per_family` of each of `families` (the kinds of the training records), the first ones by id round robin (`config.apply_quota`), each checked by
    `domain.check_task`. SystemExit for no task at all; a note for a family that has fewer than asked."""
    tasks = domain.load_heldout([pathlib.Path(f) for f in files])
    heldout = domain.heldout_ids()
    leaked = [t["id"] for t in tasks if t["id"] not in heldout]
    if leaked:
        raise SystemExit(f"--heldout holds tasks that are not held-out ids of domain {domain.name!r} ({', '.join(leaked)[:200]}): the evaluation would be on tasks that a training set may hold")
    want = config.parse_quota(quota, domain.kinds()) if quota else {kind: per_family for kind in sorted(families)}
    chosen = config.apply_quota(tasks, want)
    have = collections.Counter(t["kind"] for t in chosen)
    short = [f"{kind} {have[kind]}/{n}" for kind, n in want.items() if have[kind] < n]
    if not chosen:
        raise SystemExit(f"no held-out task of {', '.join(want)} in {', '.join(map(str, files))}")
    if short:
        print(f"note: fewer held-out tasks than asked for: {', '.join(short)}")
    for t in chosen:
        domain.check_task(t)
    return chosen


def check_training(rows: list, eval_ids, *, domain, mix: bool = False) -> dict:
    """SystemExit unless the records may be trained on and evaluated against `eval_ids`: none of a held-out task, none of a task of the evaluation subset, every teacher-reasoning record of a teacher that may
    store its text (`teachers.require_reasoning`, the rule `kaggle-score` applies to its `--model`; a truth-template record is rendered from the truth and its label is only the filter), and at least one
    thinking record. Returns {"records", "by_source", "labels", "families", "fates"}."""
    heldout, tasks = domain.heldout_ids(), sorted({r["task"] for r in rows})
    bad = [t for t in tasks if t in heldout]
    if bad:
        raise SystemExit(f"held-out tasks of domain {domain.name!r} are in the training records, and are never trained on: {', '.join(bad)[:200]}")
    overlap = sorted(set(tasks) & set(eval_ids))
    if overlap:
        raise SystemExit(f"{len(overlap)} training record task(s) are in the evaluation subset ({', '.join(overlap)[:200]}): the student would be tested on what it was trained on")
    labels = sorted({r["model"] for r in rows if r["target_source"] == records.TEACHER_SOURCE})
    for label in labels:
        teachers.require_reasoning(label, why="the Kaggle LoRA kit (it trains on the teacher's stored text)")
    texts = [r["messages"][-1]["content"] for r in rows]
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


def smoke_plan(longest: int, *, gpus: int, steps: int, max_new_tokens: int, rates: dict) -> dict:
    """The cost of a smoke run: the setup, two loads, `steps` steps of two of the longest records per process, one batch of two replies per process and arm of `max_new_tokens` tokens."""
    step_s = 2 * longest / rates["train_tok_per_s_per_gpu"]
    gen = 2 * (max_new_tokens * rates["gen_step_s"])
    total = rates["setup_s"] + 2 * rates["load_s"] + steps * step_s + gen
    return {"smoke": True, "steps": steps, "step_s": round(step_s, 1), "total_upper_s": round(total), "total_typical_s": round(total), "gpus": gpus, "rates": rates}


def describe(p: dict, gpu_hours: float, deadline_h: float) -> list:
    """The lines the builder prints about a plan (`plan` or `smoke_plan`): what is assumed, what each stage costs, whether it fits `gpu_hours` (the quota that is left), and what the kernel does at the deadline."""
    h = lambda s: f"{s / 3600:.1f} h"
    r = p["rates"]
    where = "assumed" if r == RATES else "assumed, with the numbers of --measured"
    out = []
    if p.get("smoke"):
        out.append(f"smoke run: {p['steps']} steps of the longest records at {p['step_s']} s, two replies per process and arm; {h(p['total_upper_s'])} in all ({where}); it measures what the full estimate assumes")
    else:
        out.append(f"training: {p['examples']} examples of ~{p['mean_tokens']} tokens (estimated), {p['steps_per_epoch']} steps per epoch, {p['steps_full']} steps for the epochs asked, {p['global_batch']} sequences per step on {p['gpus']} GPU(s): "
                   f"{p['step_s']} s per step at {r['train_tok_per_s_per_gpu']:.0f} tok/s per GPU ({where})")
        out.append(f"          -> {p['steps']} steps = {p['epochs_effective']} epochs, {h(p['train_s'])}" + (f"; the training budget caps it (--max-steps {p['max_steps']} in the config)" if p["max_steps"] else "; the training budget does not bind"))
        out.append(f"eval:     {p['eval_items_per_arm']} replies per arm (k={p['k']}), batches of {p['eval_batch']}, {p['eval_batches_per_process']} per process and arm, up to {p['max_new_tokens']} new tokens at {r['gen_step_s']} s per step ({where}): "
                   f"{h(p['eval_upper_s'])} at most, ~{h(p['eval_typical_s'])} typically")
        out.append(f"overhead: {h(p['overhead_s'])} (pip install, download, two model loads)" + (f"; GGUF export {h(p['export_s'])}" if p["export_s"] else ""))
    fits = p["total_upper_s"] <= gpu_hours * 3600
    out.append(f"total:    {h(p['total_upper_s'])} at most, ~{h(p['total_typical_s'])} typically, of the {gpu_hours:g} h of quota left: " + ("it fits" if fits else "IT DOES NOT FIT: lower --train-hours, --k, --per-family or --max-new-tokens"))
    out.append(f"the kernel starts nothing after {deadline_h:g} h" + ("" if p.get("smoke") else f", and training stops early enough to leave {h(p['eval_reserve_s'])} for the evaluation"))
    return out


# ---- the files -------------------------------------------------------------------------------------------------------------------------------

def kernel_files() -> dict:
    """{name in the dataset: source file} of the code the kernel runs: itself, the teacher kernel it reuses, and the stdlib-only core as flat `distill.<module>.py`."""
    files = {"lora_eval.py": HERE / "lora_eval.py", "run_teacher.py": HERE / "run_teacher.py"}
    files.update({f"distill.{m}.py": ROOT / "distill" / f"{m}.py" for m in lora_eval.LIB_MODULES})
    return files


def make_config(*, facts: dict, eval_tasks: list, heldout_files, p: dict, smoke: bool, **settings) -> dict:
    """The config.json of a run: `lora_eval.DEFAULTS` with the build's settings, validated by the kernel's own `resolve_config`, and what it was built from."""
    cfg = {**{k: v for k, v in lora_eval.DEFAULTS.items() if k != "pins"}, **settings, "smoke": smoke}
    if not smoke:
        cfg.update(max_steps=p["max_steps"], eval_reserve_s=p["eval_reserve_s"], export_reserve_s=p["export_reserve_s"])
    try:
        lora_eval.resolve_config(cfg)
    except ValueError as e:
        raise SystemExit(f"the run config is refused by the kernel: {e}")
    return {**cfg, "built_from": {"train": facts, "eval": {"tasks": len(eval_tasks), "by_kind": dict(sorted(collections.Counter(t["kind"] for t in eval_tasks).items())),
                                                           "files": [pathlib.Path(f).name for f in heldout_files]}, "estimate": {k: v for k, v in p.items() if k != "rates"}, "rates": p["rates"]}}


def build(sft_path, heldout_files, out, owner: str, *, domain, quota: str = "", per_family: int = 10, student: str = None, max_len: int = 4096, rank: int = None, epochs: float = None, lr: float = None,
          global_batch: int = None, mix: bool = False, gpus: int = None, k: int = 1, max_new_tokens: int = None, eval_batch: int = None, train_hours: float = None, deadline_h: float = None,
          gpu_hours: float = 8.0, export_gguf: bool = False, fla: bool = False, smoke: bool = False, measured=None, dataset_slug: str = None, kernel_slug: str = None, resume_from: str = "",
          force: bool = False) -> dict:
    """Write the files described above under `out`; returns the config. Refuses to overwrite a file that exists, unless `force`. The settings that are None take `lora_eval.DEFAULTS`."""
    base = domain.name.replace("_", "-")
    owner, dataset_slug, kernel_slug = _slug("--owner", owner), _slug("--dataset-slug", dataset_slug or f"{base}-lora-data"), _slug("--kernel-slug", kernel_slug or f"{base}-lora-eval")
    if resume_from and not OWNER_KERNEL.fullmatch(resume_from):
        raise SystemExit(f"--resume-from {resume_from!r}: owner/kernel-slug of the earlier run")
    out = pathlib.Path(out)
    files = kernel_files()
    paths = {"train": out / "data" / "train.jsonl", "eval": out / "data" / "eval_tasks.jsonl", "config": out / "data" / "lora-config.json", "dataset": out / "data" / "dataset-metadata.json",
             "run": out / "kernel" / "run.py", "kernel": out / "kernel" / "kernel-metadata.json", **{name: out / "data" / name for name in files}}
    outputs.refuse_overwrite(list(paths.values()), force)
    rows = read_sft(sft_path)
    families = sorted({r["kind"] for r in rows})
    eval_tasks = pick_eval(heldout_files, quota, per_family, families, domain=domain)
    facts = check_training(rows, {t["id"] for t in eval_tasks}, domain=domain, mix=mix)
    d = lora_eval.DEFAULTS
    given = {"student": student or d["student"], "max_len": max_len, "rank": rank or d["rank"], "epochs": epochs or d["epochs"], "lr": lr or d["lr"], "global_batch": global_batch or d["global_batch"], "mix": mix,
             "gpus": gpus or d["gpus"], "k": k, "max_new_tokens": max_new_tokens or d["max_new_tokens"], "eval_batch": eval_batch or d["eval_batch"], "train_hours": train_hours or d["train_hours"],
             "deadline_h": deadline_h or d["deadline_h"], "export_gguf": export_gguf, "fla": fla}
    lengths = trainable_lengths(rows, given["max_len"], mix)
    if not lengths:
        raise SystemExit(f"every thinking record is over --max-len {given['max_len']} tokens (estimated): nothing to train on")
    rates = rates_from(measured) if measured else RATES
    prompts = [est_tokens({"messages": domain.messages(t, "thinking")}) for t in eval_tasks]
    p = plan(lengths, prompts, epochs=given["epochs"], gpus=given["gpus"], global_batch=given["global_batch"], train_hours=given["train_hours"], k=k, max_new_tokens=given["max_new_tokens"],
             eval_batch=given["eval_batch"], export=export_gguf, rates=rates)
    if smoke:
        p = smoke_plan(max(lengths), gpus=given["gpus"], steps=d["smoke_steps"], max_new_tokens=256, rates=rates)
    facts = {**facts, "files": [pathlib.Path(sft_path).name], "sha256": hashlib.sha256(pathlib.Path(sft_path).read_bytes()).hexdigest()[:16], "est_examples": len(lengths)}
    cfg = make_config(facts=facts, eval_tasks=eval_tasks, heldout_files=heldout_files, p=p, smoke=smoke, **given)
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
    ap.add_argument("--heldout", type=pathlib.Path, nargs="+", required=True, help="the decompiled HELD-OUT tasks the student is tested on (REA: decompiled-seed1001-all.jsonl of the decompile stage); read by domain.load_heldout")
    ap.add_argument("--owner", required=True, help="your Kaggle user name (the owner of the dataset and the kernel)")
    ap.add_argument("--out", type=pathlib.Path, required=True, help="directory: data/ (the dataset) and kernel/ are written under it")
    ap.add_argument("--quota", default="", help="kind=N,...: the held-out tasks of each listed family and none of another; default --per-family of each family the training records have")
    ap.add_argument("--per-family", type=int, default=10, help="held-out tasks per family when --quota is not given (default %(default)s)")
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
    ap.add_argument("--deadline-h", type=float, default=None, help=f"hours after the start at which the kernel starts nothing (default {d['deadline_h']}: the weekly quota, not the 12 h session, is the limit)")
    ap.add_argument("--gpu-hours", type=float, default=8.0, help="GPU hours of quota left, what the estimate is judged against (default %(default)s)")
    ap.add_argument("--export-gguf", action="store_true", help="also merge the adapter into fp16 weights and write a Q4_K_M GGUF to /kaggle/working/gguf (about 5.6 GB; not what was evaluated; not tried on Kaggle)")
    ap.add_argument("--fla", action="store_true", help="install flash-linear-attention in the kernel (fast kernels for the student's linear-attention layers; not tried on a T4)")
    ap.add_argument("--smoke", action="store_true", help="build the SMOKE run: 4 steps on the longest records and two replies per process, files named smoke-*; it measures the step time, the generation speed and the peak memory")
    ap.add_argument("--measured", type=pathlib.Path, default=None, help="the run.json of a smoke run: its measured step time and generation speed replace the assumed ones in the estimate")
    ap.add_argument("--dataset-slug", default=None, help="default <domain>-lora-data")
    ap.add_argument("--kernel-slug", default=None, help="default <domain>-lora-eval")
    ap.add_argument("--resume-from", default="", help="owner/kernel-slug of an earlier run: its output (checkpoints, adapter, replies) is attached to this kernel and continued")
    ap.add_argument("--force", action="store_true", help="overwrite the files of an earlier build (default: refuse)")


def run(a: argparse.Namespace, domain) -> None:
    cfg = build(a.sft, a.heldout, a.out, a.owner, domain=domain, quota=a.quota, per_family=a.per_family, student=a.student, max_len=a.max_len, rank=a.rank, epochs=a.epochs, lr=a.lr,
                global_batch=a.global_batch, mix=a.mix, gpus=a.gpus, k=a.k, max_new_tokens=a.max_new_tokens, eval_batch=a.eval_batch, train_hours=a.train_hours, deadline_h=a.deadline_h,
                gpu_hours=a.gpu_hours, export_gguf=a.export_gguf, fla=a.fla, smoke=a.smoke, measured=a.measured, dataset_slug=a.dataset_slug, kernel_slug=a.kernel_slug, resume_from=a.resume_from, force=a.force)
    t = cfg["built_from"]
    print(f"{t['train']['records']} training records {t['train']['by_source']} (teacher labels {t['train']['labels']}; {t['train']['fates']}), {t['eval']['tasks']} held-out tasks {t['eval']['by_kind']}, "
          f"{'SMOKE run' if cfg['smoke'] else 'full run'} -> {a.out}")
    print("\n".join(cfg["plan_lines"]))
    print(f"next: kaggle datasets create -p {a.out}/data   then   kaggle kernels push -p {a.out}/kernel   (set the kernel's accelerator to GPU T4 x2; distill/backends/kaggle/README.md)")


def main(argv=None) -> None:
    from distill.domains import load
    ap = argparse.ArgumentParser(description="Build the Kaggle inputs of a LoRA training + held-out evaluation run (records, held-out tasks, config, metadata, kernel files) and print its cost")
    ap.add_argument("--domain", default="rea", help="the subject: a package of distill/domains (default %(default)s)")
    add_arguments(ap)
    a = ap.parse_args(argv)
    run(a, load(a.domain))


if __name__ == "__main__":
    main()
