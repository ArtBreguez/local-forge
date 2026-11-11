"""Build what a Kaggle teacher run needs, for any subject: the tasks (with their exact messages), the run's config, the dataset and kernel metadata, and the kernel script. Moved from `tools/kaggle/build_tasks.py`,
which is now the REA wrapper (REA bound, the commands of the README unchanged); this one takes `--domain` (a package of `distill/domains`) and `--teacher` (a name of `distill/teachers.py`).

    python3 tools/kaggle/build_tasks.py --tasks results/phase6/rea-data/decompiled-seed9-all.jsonl --owner <kaggle-user> --out ~/work/kaggle-rea \\
        --quota structs=220,crc=220,antidebug=220,bounds=220,fsm=220,pktsum=80,encoding=80 [--k 4] [--max-tokens 4096] [--no-thinking] [--force]
    python3 -m distill.backends.kaggle.build_tasks --domain rea --tasks ...        (the same, any subject)

Writes under --out:
    data/tasks.jsonl             {"id", "kind", "messages"} per task: `domain.messages(task, mode)` (for REA SYSTEM_REASONING + the user message), nothing else of the task (the truth stays here)
    data/config.json             the run: model, k, thinking, sampling (Qwen3's thinking settings with --thinking, never greedy), max_tokens, slots, ctx_per_slot (prompt + max_tokens fit one slot),
                                 deadline, seed base, where the tasks came from
    data/dataset-metadata.json   for `kaggle datasets create -p <out>/data`
    kernel/run.py                tools/kaggle/run_teacher.py
    kernel/kernel-metadata.json  for `kaggle kernels push -p <out>/kernel` (private, GPU, internet on, the dataset attached; --resume-from also attaches a previous kernel's output)

The tasks are the first N per family by id, in the order `modal_rea_data.py --quota` takes them (`config.apply_quota`, round-robin over the families so a run that is stopped early is balanced), from
what `domain.load_tasks` reads (REA: a `decompiled-seed<seed>-<family>.jsonl` of the decompile stage); a held-out task is refused. Stdlib only. tools/kaggle/README.md has the commands that follow.
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import re
import shutil
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
from distill import config, outputs, teachers  # noqa: E402
import run_teacher  # noqa: E402  (its DEFAULTS: model, slots, deadline, seed base)

SLUG = re.compile(r"[a-z0-9][a-z0-9-]*")


def _slug(what: str, value: str) -> str:
    if not SLUG.fullmatch(value or ""):
        raise SystemExit(f"{what} {value!r}: lower-case letters, digits and `-` (a Kaggle owner or slug)")
    return value


def pick_tasks(files, quota: str = "", *, domain) -> list:
    """The tasks of `files` (`domain.load_tasks`: for REA the decompiled records sorted by id as the sample stage reads them), after `--quota` (`config.apply_quota`). SystemExit for a held-out task, for a quota
    that names a kind no task has or selects nothing, and for a task the sample stage would refuse (`domain.check_task`)."""
    files = [pathlib.Path(f) for f in files]
    tasks = domain.load_tasks(files)
    heldout = domain.heldout_ids()
    if any(t["id"] in heldout for t in tasks):
        raise SystemExit(f"held-out tasks of domain {domain.name!r} are never sampled for training: {', '.join(t['id'] for t in tasks if t['id'] in heldout)[:200]}")
    if not tasks:
        raise SystemExit(f"no task in {', '.join(map(str, files))}")
    if quota:
        want = config.parse_quota(quota, domain.kinds())
        tasks = config.apply_quota(tasks, want)
        have = collections.Counter(t["kind"] for t in tasks)
        short = [f"{kind} {have[kind]}/{n}" for kind, n in want.items() if have[kind] < n]
        if not tasks:
            raise SystemExit(f"--quota {quota} selects no task of {', '.join(map(str, files))}")
        if short:
            print(f"note: fewer tasks than the quota asks for: {', '.join(short)}")
    for t in tasks:
        domain.check_task(t)
    return tasks


def make_config(tasks: list, files, quota: str, k: int, thinking: bool, max_tokens, sampling_overrides: dict, deadline_h: float, model_repo, model_file, *, domain) -> dict:
    """The config.json of a run. Sampling and `max_tokens` follow `thinking` (`config.sampling_for`, `config.default_max_tokens`); `ctx_per_slot` is `config.slot_context(max_tokens)`: the prompt
    and the reply fit one slot."""
    if k < 1:
        raise SystemExit(f"--k {k}: at least one reply per task")
    max_tokens = max_tokens or config.default_max_tokens(True, thinking)
    if max_tokens < 1:
        raise SystemExit(f"--max-tokens {max_tokens}")
    return {"model_repo": model_repo or run_teacher.DEFAULTS["model_repo"], "model_file": model_file or run_teacher.DEFAULTS["model_file"], "k": k, "thinking": thinking,
            "sampling": config.sampling_for(thinking, **sampling_overrides), "max_tokens": max_tokens, "slots": run_teacher.DEFAULTS["slots"], "ctx_per_slot": config.slot_context(max_tokens),
            "deadline_h": deadline_h, "seed_base": run_teacher.DEFAULTS["seed_base"], "reasoning_format": run_teacher.DEFAULTS["reasoning_format"],
            "tasks": {"files": [pathlib.Path(f).name for f in files], "quota": quota, "n": len(tasks), "by_kind": dict(sorted(collections.Counter(t["kind"] for t in tasks).items())),
                      "system_sha": outputs.prompt_sha([domain.messages(tasks[0], "thinking" if thinking else "reasoning")[0]])}}


def build(files, out, owner: str, quota: str = "", k: int = 4, thinking: bool = True, max_tokens=None, sampling_overrides=None, deadline_h: float = None,
          dataset_slug: str = None, kernel_slug: str = None, resume_from: str = "", model_repo=None, model_file=None, force: bool = False, teacher: str = None, *, domain) -> dict:
    """Write the files described above under `out`; returns the config. Refuses to overwrite a file that exists, unless `force`. `teacher`: a name of `distill/teachers.py` whose repo and file are the model
    (`model_repo` and `model_file` still override); the slugs default to `<domain>-reasoning-tasks` and `<domain>-teacher`."""
    base = domain.name.replace("_", "-")
    owner, dataset_slug, kernel_slug = _slug("--owner", owner), _slug("--dataset-slug", dataset_slug or f"{base}-reasoning-tasks"), _slug("--kernel-slug", kernel_slug or f"{base}-teacher")
    if resume_from and not re.fullmatch(r"[a-z0-9-]+/[a-z0-9-]+", resume_from):
        raise SystemExit(f"--resume-from {resume_from!r}: owner/kernel-slug of the earlier run")
    if teacher:
        found = teachers.find(teacher)
        model_repo, model_file = model_repo or found.repo, model_file or found.file
    out = pathlib.Path(out)
    paths = {"tasks": out / "data" / "tasks.jsonl", "config": out / "data" / "config.json", "dataset": out / "data" / "dataset-metadata.json", "run": out / "kernel" / "run.py",
             "kernel": out / "kernel" / "kernel-metadata.json"}
    outputs.refuse_overwrite(list(paths.values()), force)
    tasks = pick_tasks(files, quota, domain=domain)
    cfg = make_config(tasks, files, quota, k, thinking, max_tokens, sampling_overrides or {}, deadline_h or run_teacher.DEFAULTS["deadline_h"], model_repo, model_file, domain=domain)
    for p in paths.values():
        p.parent.mkdir(parents=True, exist_ok=True)
    mode = "thinking" if thinking else "reasoning"
    outputs.write_atomic(paths["tasks"], (json.dumps({"id": t["id"], "kind": t["kind"], "messages": domain.messages(t, mode)}) for t in tasks))
    paths["config"].write_text(json.dumps(cfg, indent=1))
    paths["dataset"].write_text(json.dumps({"title": dataset_slug, "id": f"{owner}/{dataset_slug}", "licenses": [{"name": "other"}]}, indent=1))
    shutil.copyfile(HERE / "run_teacher.py", paths["run"])
    paths["kernel"].write_text(json.dumps({"id": f"{owner}/{kernel_slug}", "title": kernel_slug, "code_file": "run.py", "language": "python", "kernel_type": "script", "is_private": True,
                                           "enable_gpu": True, "enable_internet": True, "dataset_sources": [f"{owner}/{dataset_slug}"], "competition_sources": [],
                                           "kernel_sources": [resume_from] if resume_from else []}, indent=1))
    return cfg


def add_arguments(ap: argparse.ArgumentParser, tasks_help: str) -> None:
    ap.add_argument("--tasks", type=pathlib.Path, nargs="+", required=True, help=tasks_help)
    ap.add_argument("--owner", required=True, help="your Kaggle user name (the owner of the dataset and the kernel)")
    ap.add_argument("--out", type=pathlib.Path, required=True, help="directory: data/ (the dataset) and kernel/ are written under it")
    ap.add_argument("--quota", default="", help="kind=N,...: at most N tasks of each listed kind and none of another (the first N by id, round-robin); default every task")
    ap.add_argument("--k", type=int, default=4, help="replies per task (default %(default)s)")
    ap.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=True, help="the model's native reasoning (default; --no-thinking: visible reasoning, 768 tokens, temperature 0.7)")
    ap.add_argument("--max-tokens", type=int, default=None, help=f"per reply (default {config.DEFAULT_THINKING_MAX_TOKENS} thinking, {config.DEFAULT_MAX_TOKENS} otherwise)")
    ap.add_argument("--temperature", type=float, default=None, help="override the sampling (a thinking run is never greedy)")
    ap.add_argument("--top-p", type=float, default=None)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--min-p", type=float, default=None)
    ap.add_argument("--deadline-h", type=float, default=run_teacher.DEFAULTS["deadline_h"], help="hours after the start at which no new request is started (default %(default)s; a session ends at 12)")
    ap.add_argument("--dataset-slug", default=None, help="default <domain>-reasoning-tasks")
    ap.add_argument("--kernel-slug", default=None, help="default <domain>-teacher")
    ap.add_argument("--resume-from", default="", help="owner/kernel-slug of an earlier run: its output (replies.jsonl, and the built llama-server) is attached to this kernel")
    ap.add_argument("--teacher", default=None, help="a teacher of distill/teachers.py (its GGUF repo and file); default: the kernel's own default, CodeForge 27B heretic Q4")
    ap.add_argument("--model-repo", default=None, help=f"Hugging Face repo of the GGUF (default {run_teacher.DEFAULTS['model_repo']})")
    ap.add_argument("--model-file", default=None, help=f"the GGUF file (default {run_teacher.DEFAULTS['model_file']})")
    ap.add_argument("--force", action="store_true", help="overwrite the files of an earlier build (default: refuse)")


def run(a: argparse.Namespace, domain) -> None:
    cfg = build(a.tasks, a.out, a.owner, a.quota, a.k, a.thinking, a.max_tokens, {"temperature": a.temperature, "top_p": a.top_p, "top_k": a.top_k, "min_p": a.min_p}, a.deadline_h,
                a.dataset_slug, a.kernel_slug, a.resume_from, a.model_repo, a.model_file, a.force, a.teacher, domain=domain)
    t = cfg["tasks"]
    print(f"{t['n']} tasks {t['by_kind']}, k={cfg['k']}, {'thinking' if cfg['thinking'] else 'visible reasoning'}, sampling {cfg['sampling']}, max_tokens {cfg['max_tokens']}, "
          f"slot {cfg['ctx_per_slot']} tokens x {cfg['slots']}, deadline {cfg['deadline_h']} h -> {a.out}")
    print(f"next: kaggle datasets create -p {a.out}/data   then   kaggle kernels push -p {a.out}/kernel   (tools/kaggle/README.md)")


def main(argv=None) -> None:
    from distill.domains import load
    ap = argparse.ArgumentParser(description="Build the Kaggle inputs of a teacher run (tasks, config, metadata, kernel script)")
    ap.add_argument("--domain", default="rea", help="the subject: a package of distill/domains (default %(default)s)")
    add_arguments(ap, "the tasks the subject's task stage wrote (what `domain.load_tasks` reads; never held-out tasks)")
    a = ap.parse_args(argv)
    run(a, load(a.domain))


if __name__ == "__main__":
    main()
