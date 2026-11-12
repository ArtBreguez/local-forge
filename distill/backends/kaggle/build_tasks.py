"""Build what a Kaggle teacher run needs, for any subject: the tasks (with their exact messages), the run's config, the dataset and kernel metadata, and the kernel script. Moved from `tools/kaggle/build_tasks.py`,
which is now the REA wrapper (REA bound, the commands of the README unchanged); this one takes `--domain` (a package of `distill/domains`) and `--teacher` (a name of `distill/teachers.py`).

    python3 tools/kaggle/build_tasks.py --tasks results/phase6/rea-data/decompiled-seed9-all.jsonl --owner <kaggle-user> --out ~/work/kaggle-rea \\
        --quota structs=220,crc=220,antidebug=220,bounds=220,fsm=220,pktsum=80,encoding=80 [--k 4] [--max-tokens 4096] [--max-tokens-by-family cipher=8192,magic=8192] [--no-thinking] [--force]
    python3 tools/kaggle/build_tasks.py --run-config distill/configs/rea-thinking.json --tasks ... --owner ... --out ...        (k, quota, teacher, sampling and token budgets of a run config)
    python3 -m distill.backends.kaggle.build_tasks --domain rea --tasks ...        (the same, any subject)

Writes under --out:
    data/tasks.jsonl             {"id", "kind", "messages"} per task: `domain.messages(task, mode)` (for REA SYSTEM_REASONING + the user message), nothing else of the task (the truth stays here)
    data/config.json             the run: model, k, thinking, sampling (Qwen3's thinking settings with --thinking, never greedy), max_tokens, `max_tokens_by_family` (only when a family has a budget of
                                 its own), slots, ctx_per_slot (prompt + max_tokens fit one slot: the BASE; `run_teacher.server_geometry` makes the slot longer and the slots fewer when a family has a
                                 longer budget), deadline, seed base, where the tasks came from
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


def make_config(tasks: list, files, quota: str, k: int, thinking: bool, max_tokens, sampling_overrides: dict, deadline_h: float, model_repo, model_file, *, domain, max_tokens_by_family: dict = None) -> dict:
    """The config.json of a run. Sampling and `max_tokens` follow `thinking` (`config.sampling_for`, `config.default_max_tokens`); `ctx_per_slot` is `config.slot_context(max_tokens)`: the prompt
    and the reply fit one slot. `max_tokens_by_family`: {family: tokens} (`config.check_family_tokens`: the families must be the domain's), written only when there is one; the kernel gives those tasks that
    budget and sizes its server for it (`run_teacher.server_geometry`)."""
    if k < 1:
        raise SystemExit(f"--k {k}: at least one reply per task")
    max_tokens = max_tokens or config.default_max_tokens(True, thinking)
    if max_tokens < 1:
        raise SystemExit(f"--max-tokens {max_tokens}")
    by_family = config.check_family_tokens(max_tokens_by_family or {}, domain.kinds())
    return {"model_repo": model_repo or run_teacher.DEFAULTS["model_repo"], "model_file": model_file or run_teacher.DEFAULTS["model_file"], "k": k, "thinking": thinking,
            "sampling": config.sampling_for(thinking, **sampling_overrides), "max_tokens": max_tokens, **({"max_tokens_by_family": by_family} if by_family else {}),
            "slots": run_teacher.DEFAULTS["slots"], "ctx_per_slot": config.slot_context(max_tokens),
            "deadline_h": deadline_h, "seed_base": run_teacher.DEFAULTS["seed_base"], "reasoning_format": run_teacher.DEFAULTS["reasoning_format"],
            "tasks": {"files": [pathlib.Path(f).name for f in files], "quota": quota, "n": len(tasks), "by_kind": dict(sorted(collections.Counter(t["kind"] for t in tasks).items())),
                      "system_sha": outputs.prompt_sha([domain.messages(tasks[0], "thinking" if thinking else "reasoning")[0]])}}


def build(files, out, owner: str, quota: str = "", k: int = 4, thinking: bool = True, max_tokens=None, sampling_overrides=None, deadline_h: float = None,
          dataset_slug: str = None, kernel_slug: str = None, resume_from: str = "", model_repo=None, model_file=None, force: bool = False, teacher: str = None, max_tokens_by_family: dict = None, llama_dataset: str = "", *, domain) -> dict:
    """Write the files described above under `out`; returns the config. Refuses to overwrite a file that exists, unless `force`. `teacher`: a name of `distill/teachers.py` whose repo and file are the model
    (`model_repo` and `model_file` still override); the slugs default to `<domain>-reasoning-tasks` and `<domain>-teacher`. `max_tokens_by_family`: see `make_config`. `llama_dataset`: owner/slug of a dataset that holds a built llama-server (`llama-bin/`, see distill/backends/kaggle/README.md): attached to the
    kernel next to the tasks, so the kernel skips the 30-minute build."""
    base = domain.name.replace("_", "-")
    owner, dataset_slug, kernel_slug = _slug("--owner", owner), _slug("--dataset-slug", dataset_slug or f"{base}-reasoning-tasks"), _slug("--kernel-slug", kernel_slug or f"{base}-teacher")
    if resume_from and not re.fullmatch(r"[a-z0-9-]+/[a-z0-9-]+", resume_from):
        raise SystemExit(f"--resume-from {resume_from!r}: owner/kernel-slug of the earlier run")
    if llama_dataset and not re.fullmatch(r"[a-z0-9-]+/[a-z0-9-]+", llama_dataset):
        raise SystemExit(f"--llama-dataset {llama_dataset!r}: owner/dataset-slug of the dataset that holds llama-bin/")
    if teacher:
        found = teachers.find(teacher)
        model_repo, model_file = model_repo or found.repo, model_file or found.file
    # the kit asks a reasoning or thinking run, which stores the teacher's reply: only a teacher of the catalogue with `reasoning_ok` (the kernel's default is CodeForge, whose entry says so)
    teachers.require_reasoning(teacher or "", model_repo or run_teacher.DEFAULTS["model_repo"], model_file or run_teacher.DEFAULTS["model_file"], strict=True, why="the Kaggle kit (a reasoning or thinking run)")
    out = pathlib.Path(out)
    paths = {"tasks": out / "data" / "tasks.jsonl", "config": out / "data" / "config.json", "dataset": out / "data" / "dataset-metadata.json", "run": out / "kernel" / "run.py",
             "kernel": out / "kernel" / "kernel-metadata.json"}
    outputs.refuse_overwrite(list(paths.values()), force)
    tasks = pick_tasks(files, quota, domain=domain)
    cfg = make_config(tasks, files, quota, k, thinking, max_tokens, sampling_overrides or {}, deadline_h or run_teacher.DEFAULTS["deadline_h"], model_repo, model_file, domain=domain,
                      max_tokens_by_family=max_tokens_by_family)
    for p in paths.values():
        p.parent.mkdir(parents=True, exist_ok=True)
    mode = "thinking" if thinking else "reasoning"
    outputs.write_atomic(paths["tasks"], (json.dumps({"id": t["id"], "kind": t["kind"], "messages": domain.messages(t, mode)}) for t in tasks))
    paths["config"].write_text(json.dumps(cfg, indent=1))
    paths["dataset"].write_text(json.dumps({"title": dataset_slug, "id": f"{owner}/{dataset_slug}", "licenses": [{"name": "other"}]}, indent=1))
    shutil.copyfile(HERE / "run_teacher.py", paths["run"])
    paths["kernel"].write_text(json.dumps({"id": f"{owner}/{kernel_slug}", "title": kernel_slug, "code_file": "run.py", "language": "python", "kernel_type": "script", "is_private": True,
                                           "enable_gpu": True, "enable_internet": True, "dataset_sources": [f"{owner}/{dataset_slug}", *([llama_dataset] if llama_dataset else [])], "competition_sources": [],
                                           "kernel_sources": [resume_from] if resume_from else []}, indent=1))
    return cfg


def add_arguments(ap: argparse.ArgumentParser, tasks_help: str) -> None:
    ap.add_argument("--tasks", type=pathlib.Path, nargs="+", required=True, help=tasks_help)
    ap.add_argument("--owner", required=True, help="your Kaggle user name (the owner of the dataset and the kernel)")
    ap.add_argument("--out", type=pathlib.Path, required=True, help="directory: data/ (the dataset) and kernel/ are written under it")
    ap.add_argument("--quota", default="", help="kind=N,...: at most N tasks of each listed kind and none of another (the first N by id, round-robin); default every task")
    ap.add_argument("--run-config", type=pathlib.Path, default=None, help="a run config (distill/configs/rea-thinking.json) that sets --teacher, --k, --quota, --thinking, --max-tokens, --max-tokens-by-family and the sampling "
                    "instead of their flags (a flag that is also given is refused)")
    ap.add_argument("--k", type=int, default=None, help=f"replies per task (default {DEFAULT_K})")
    ap.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=None, help="the model's native reasoning (default; --no-thinking: visible reasoning, 768 tokens, temperature 0.7)")
    ap.add_argument("--max-tokens", type=int, default=None, help=f"per reply (default {config.DEFAULT_THINKING_MAX_TOKENS} thinking, {config.DEFAULT_MAX_TOKENS} otherwise)")
    ap.add_argument("--max-tokens-by-family", default=None, help="kind=N,...: the reply budget of the tasks of those families instead of --max-tokens, e.g. cipher=8192,magic=8192,password=8192; the kernel sizes its "
                    "server for the longest (2 slots of 12288 tokens instead of 4 of 8192: the same memory, half the requests in flight)")
    ap.add_argument("--temperature", type=float, default=None, help="override the sampling (a thinking run is never greedy)")
    ap.add_argument("--top-p", type=float, default=None)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--min-p", type=float, default=None)
    ap.add_argument("--deadline-h", type=float, default=run_teacher.DEFAULTS["deadline_h"], help="hours after the start at which no new request is started (default %(default)s; a session ends at 12)")
    ap.add_argument("--dataset-slug", default=None, help="default <domain>-reasoning-tasks")
    ap.add_argument("--kernel-slug", default=None, help="default <domain>-teacher")
    ap.add_argument("--resume-from", default="", help="owner/kernel-slug of an earlier run: its output (replies.jsonl, and the built llama-server) is attached to this kernel")
    ap.add_argument("--llama-dataset", default="", help="owner/slug of a Kaggle dataset that holds llama-bin/ (a built llama-server and its libraries): attached to the kernel, which then skips the llama.cpp build "
                    "(distill/backends/kaggle/README.md says how to publish one)")
    ap.add_argument("--teacher", default=None, help="a teacher of distill/teachers.py (its GGUF repo and file); default: the kernel's own default, CodeForge 27B heretic Q4")
    ap.add_argument("--model-repo", default=None, help=f"Hugging Face repo of the GGUF (default {run_teacher.DEFAULTS['model_repo']})")
    ap.add_argument("--model-file", default=None, help=f"the GGUF file (default {run_teacher.DEFAULTS['model_file']})")
    ap.add_argument("--force", action="store_true", help="overwrite the files of an earlier build (default: refuse)")


DEFAULT_K = 4
RUN_CONFIG_FLAGS = ("teacher", "k", "quota", "thinking", "max_tokens", "max_tokens_by_family", "temperature", "top_p", "top_k", "min_p")      # what `--run-config` sets


def settings(a: argparse.Namespace, domain) -> dict:
    """The run's settings from the flags, or from `--run-config` (then none of RUN_CONFIG_FLAGS may be given: two sources of one setting is a mistake). {teacher, k, quota, thinking, max_tokens,
    sampling_overrides, max_tokens_by_family}. The kit asks a reasoning or a thinking run: a plain config is refused."""
    if a.run_config is None:
        return {"teacher": a.teacher, "k": DEFAULT_K if a.k is None else a.k, "quota": a.quota, "thinking": True if a.thinking is None else a.thinking, "max_tokens": a.max_tokens,
                "sampling_overrides": {"temperature": a.temperature, "top_p": a.top_p, "top_k": a.top_k, "min_p": a.min_p},
                "max_tokens_by_family": config.parse_family_tokens(a.max_tokens_by_family or "", domain.kinds())}
    cfg = config.RunConfig.load(a.run_config)
    if cfg.domain != domain.name:
        raise SystemExit(f"--run-config {a.run_config} is for domain {cfg.domain!r}, this kit is {domain.name!r}'s")
    clash = [f for f in RUN_CONFIG_FLAGS if getattr(a, f) not in (None, "")]
    if clash:
        raise SystemExit(f"--run-config {a.run_config} sets the run: do not also give " + ", ".join("--" + f.replace("_", "-") for f in clash))
    if not cfg.reasoning:
        raise SystemExit(f"--run-config {a.run_config} is a plain run (mode {cfg.mode!r}): the Kaggle kit asks a reasoning or thinking run")
    sampling = cfg.sampling or {}
    return {"teacher": cfg.teacher, "k": cfg.k, "quota": cfg.quota, "thinking": cfg.thinking, "max_tokens": cfg.max_tokens or None, "sampling_overrides": {key: sampling.get(key) for key in config.SAMPLING_KEYS},
            "max_tokens_by_family": dict(cfg.max_tokens_by_family)}


def run(a: argparse.Namespace, domain) -> None:
    st = settings(a, domain)
    cfg = build(a.tasks, a.out, a.owner, st["quota"], st["k"], st["thinking"], st["max_tokens"], st["sampling_overrides"], a.deadline_h,
                a.dataset_slug, a.kernel_slug, a.resume_from, a.model_repo, a.model_file, a.force, st["teacher"], st["max_tokens_by_family"], a.llama_dataset, domain=domain)
    t = cfg["tasks"]
    slots, ctx = run_teacher.server_geometry({**cfg, "slots": cfg["slots"], "ctx_per_slot": cfg["ctx_per_slot"]}, [{"kind": kind} for kind in t["by_kind"]])
    print(f"{t['n']} tasks {t['by_kind']}, k={cfg['k']}, {'thinking' if cfg['thinking'] else 'visible reasoning'}, sampling {cfg['sampling']}, max_tokens {cfg['max_tokens']}"
          + (f" (per family {config.family_tokens_text(cfg['max_tokens_by_family'])})" if cfg.get("max_tokens_by_family") else "")
          + f", slot {cfg['ctx_per_slot']} tokens x {cfg['slots']}"
          + (f" (the kernel runs {slots} x {ctx}: a task of the run has a longer budget, the same total context)" if (slots, ctx) != (cfg["slots"], cfg["ctx_per_slot"]) else "")
          + f", deadline {cfg['deadline_h']} h -> {a.out}")
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
