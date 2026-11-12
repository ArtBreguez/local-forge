"""Build what a Kaggle teacher run needs: the tasks (with their exact messages), the run's config, the dataset and kernel metadata, and the kernel script.

    python3 tools/kaggle/build_tasks.py --tasks results/phase6/rea-data/decompiled-seed9-all.jsonl --owner <kaggle-user> --out ~/work/kaggle-rea \\
        --quota structs=220,crc=220,antidebug=220,bounds=220,fsm=220,pktsum=80,encoding=80 [--k 4] [--max-tokens 4096] [--no-thinking] [--force]

Writes under --out:
    data/tasks.jsonl             {"id", "kind", "messages"} per task: `reasoning.messages(task)` (SYSTEM_REASONING + the user message), nothing else of the task (the truth stays here)
    data/config.json             the run: model, k, thinking, sampling (Qwen3's thinking settings with --thinking, never greedy), max_tokens, slots, ctx_per_slot (prompt + max_tokens fit one slot),
                                 deadline, seed base, where the tasks came from
    data/dataset-metadata.json   for `kaggle datasets create -p <out>/data`
    kernel/run.py                tools/kaggle/run_teacher.py
    kernel/kernel-metadata.json  for `kaggle kernels push -p <out>/kernel` (private, GPU, internet on, the dataset attached; --resume-from also attaches a previous kernel's output)

The tasks are the first N per family by id, in the order `modal_rea_data.py --quota` takes them (`reasoning.apply_quota`, round-robin over the families so a run that is stopped early is balanced), from
a `decompiled-seed<seed>-<family>.jsonl` of the decompile stage; the held-out seed 1001 is refused. Stdlib only. tools/kaggle/README.md has the commands that follow.
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
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "tools" / "rea-bench"))
sys.path.insert(0, str(HERE))
from rea_bench import dedupe, outputs, reasoning as rsn  # noqa: E402
from rea_bench.scoring import check_mode, check_target  # noqa: E402
import run_teacher  # noqa: E402  (its DEFAULTS: model, slots, deadline, seed base)

SLUG = re.compile(r"[a-z0-9][a-z0-9-]*")


def _slug(what: str, value: str) -> str:
    if not SLUG.fullmatch(value or ""):
        raise SystemExit(f"{what} {value!r}: lower-case letters, digits and `-` (a Kaggle owner or slug)")
    return value


def pick_tasks(files, quota: str = "") -> list:
    """The decompiled tasks of `files` (records with pseudocode, sorted by id as the sample stage reads them), after `--quota` (`reasoning.apply_quota`). SystemExit for the held-out seed 1001, for a quota
    that names a kind no task has or selects nothing, and for a task the sample stage would refuse (`check_mode`, `check_target`)."""
    files = [pathlib.Path(f) for f in files]
    for f in files:
        seed = re.search(r"seed(\d+)", f.name)
        if seed:
            dedupe.check_sample_seed(int(seed.group(1)))
    tasks = sorted(outputs.load_tasks(files), key=lambda t: f"{t['id']}.json")
    if any(re.search(r"-1001-\d+$", t["id"]) for t in tasks):
        dedupe.check_sample_seed(dedupe.HELDOUT_SEED)
    if not tasks:
        raise SystemExit(f"no decompiled task (a record with pseudocode) in {', '.join(map(str, files))}")
    if quota:
        want = rsn.parse_quota(quota, rsn.known_kinds())
        tasks = rsn.apply_quota(tasks, want)
        have = collections.Counter(t["kind"] for t in tasks)
        short = [f"{kind} {have[kind]}/{n}" for kind, n in want.items() if have[kind] < n]
        if not tasks:
            raise SystemExit(f"--quota {quota} selects no task of {', '.join(map(str, files))}")
        if short:
            print(f"note: fewer tasks than the quota asks for: {', '.join(short)}")
    for t in tasks:
        check_mode(t)
        check_target(t)
    return tasks


def make_config(tasks: list, files, quota: str, k: int, thinking: bool, max_tokens, sampling_overrides: dict, deadline_h: float, model_repo, model_file) -> dict:
    """The config.json of a run. Sampling and `max_tokens` follow `thinking` (`reasoning.sampling_for`, `reasoning.default_max_tokens`); `ctx_per_slot` is `reasoning.slot_context(max_tokens)`: the prompt
    and the reply fit one slot."""
    if k < 1:
        raise SystemExit(f"--k {k}: at least one reply per task")
    max_tokens = max_tokens or rsn.default_max_tokens(True, thinking)
    if max_tokens < 1:
        raise SystemExit(f"--max-tokens {max_tokens}")
    return {"model_repo": model_repo or run_teacher.DEFAULTS["model_repo"], "model_file": model_file or run_teacher.DEFAULTS["model_file"], "k": k, "thinking": thinking,
            "sampling": rsn.sampling_for(thinking, **sampling_overrides), "max_tokens": max_tokens, "slots": run_teacher.DEFAULTS["slots"], "ctx_per_slot": rsn.slot_context(max_tokens),
            "deadline_h": deadline_h, "seed_base": run_teacher.DEFAULTS["seed_base"], "reasoning_format": run_teacher.DEFAULTS["reasoning_format"],
            "tasks": {"files": [pathlib.Path(f).name for f in files], "quota": quota, "n": len(tasks), "by_kind": dict(sorted(collections.Counter(t["kind"] for t in tasks).items())),
                      "system_sha": rsn.prompt_sha([{"role": "system", "content": rsn.SYSTEM_REASONING}])}}


def build(files, out, owner: str, quota: str = "", k: int = 4, thinking: bool = True, max_tokens=None, sampling_overrides=None, deadline_h: float = None,
          dataset_slug: str = "rea-reasoning-tasks", kernel_slug: str = "rea-teacher", resume_from: str = "", model_repo=None, model_file=None, force: bool = False) -> dict:
    """Write the files described above under `out`; returns the config. Refuses to overwrite a file that exists, unless `force`."""
    owner, dataset_slug, kernel_slug = _slug("--owner", owner), _slug("--dataset-slug", dataset_slug), _slug("--kernel-slug", kernel_slug)
    if resume_from and not re.fullmatch(r"[a-z0-9-]+/[a-z0-9-]+", resume_from):
        raise SystemExit(f"--resume-from {resume_from!r}: owner/kernel-slug of the earlier run")
    out = pathlib.Path(out)
    paths = {"tasks": out / "data" / "tasks.jsonl", "config": out / "data" / "config.json", "dataset": out / "data" / "dataset-metadata.json", "run": out / "kernel" / "run.py",
             "kernel": out / "kernel" / "kernel-metadata.json"}
    outputs.refuse_overwrite(list(paths.values()), force)
    tasks = pick_tasks(files, quota)
    cfg = make_config(tasks, files, quota, k, thinking, max_tokens, sampling_overrides or {}, deadline_h or run_teacher.DEFAULTS["deadline_h"], model_repo, model_file)
    for p in paths.values():
        p.parent.mkdir(parents=True, exist_ok=True)
    outputs.write_atomic(paths["tasks"], (json.dumps({"id": t["id"], "kind": t["kind"], "messages": rsn.messages(t)}) for t in tasks))
    paths["config"].write_text(json.dumps(cfg, indent=1))
    paths["dataset"].write_text(json.dumps({"title": dataset_slug, "id": f"{owner}/{dataset_slug}", "licenses": [{"name": "other"}]}, indent=1))
    shutil.copyfile(HERE / "run_teacher.py", paths["run"])
    paths["kernel"].write_text(json.dumps({"id": f"{owner}/{kernel_slug}", "title": kernel_slug, "code_file": "run.py", "language": "python", "kernel_type": "script", "is_private": True,
                                           "enable_gpu": True, "enable_internet": True, "dataset_sources": [f"{owner}/{dataset_slug}"], "competition_sources": [],
                                           "kernel_sources": [resume_from] if resume_from else []}, indent=1))
    return cfg


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Build the Kaggle inputs of a teacher run (tasks, config, metadata, kernel script)")
    ap.add_argument("--tasks", type=pathlib.Path, nargs="+", required=True, help="decompiled-seed<seed>-<family>.jsonl of the decompile stage (never seed 1001)")
    ap.add_argument("--owner", required=True, help="your Kaggle user name (the owner of the dataset and the kernel)")
    ap.add_argument("--out", type=pathlib.Path, required=True, help="directory: data/ (the dataset) and kernel/ are written under it")
    ap.add_argument("--quota", default="", help="kind=N,...: at most N tasks of each listed kind and none of another (the first N by id, round-robin); default every task")
    ap.add_argument("--k", type=int, default=4, help="replies per task (default %(default)s)")
    ap.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=True, help="the model's native reasoning (default; --no-thinking: visible reasoning, 768 tokens, temperature 0.7)")
    ap.add_argument("--max-tokens", type=int, default=None, help=f"per reply (default {rsn.DEFAULT_THINKING_MAX_TOKENS} thinking, {rsn.DEFAULT_MAX_TOKENS} otherwise)")
    ap.add_argument("--temperature", type=float, default=None, help="override the sampling (a thinking run is never greedy)")
    ap.add_argument("--top-p", type=float, default=None)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--min-p", type=float, default=None)
    ap.add_argument("--deadline-h", type=float, default=run_teacher.DEFAULTS["deadline_h"], help="hours after the start at which no new request is started (default %(default)s; a session ends at 12)")
    ap.add_argument("--dataset-slug", default="rea-reasoning-tasks")
    ap.add_argument("--kernel-slug", default="rea-teacher")
    ap.add_argument("--resume-from", default="", help="owner/kernel-slug of an earlier run: its output (replies.jsonl, and the built llama-server) is attached to this kernel")
    ap.add_argument("--model-repo", default=None, help=f"Hugging Face repo of the GGUF (default {run_teacher.DEFAULTS['model_repo']})")
    ap.add_argument("--model-file", default=None, help=f"the GGUF file (default {run_teacher.DEFAULTS['model_file']})")
    ap.add_argument("--force", action="store_true", help="overwrite the files of an earlier build (default: refuse)")
    a = ap.parse_args(argv)
    cfg = build(a.tasks, a.out, a.owner, a.quota, a.k, a.thinking, a.max_tokens, {"temperature": a.temperature, "top_p": a.top_p, "top_k": a.top_k, "min_p": a.min_p}, a.deadline_h,
                a.dataset_slug, a.kernel_slug, a.resume_from, a.model_repo, a.model_file, a.force)
    t = cfg["tasks"]
    print(f"{t['n']} tasks {t['by_kind']}, k={cfg['k']}, {'thinking' if cfg['thinking'] else 'visible reasoning'}, sampling {cfg['sampling']}, max_tokens {cfg['max_tokens']}, "
          f"slot {cfg['ctx_per_slot']} tokens x {cfg['slots']}, deadline {cfg['deadline_h']} h -> {a.out}")
    print(f"next: kaggle datasets create -p {a.out}/data   then   kaggle kernels push -p {a.out}/kernel   (tools/kaggle/README.md)")


if __name__ == "__main__":
    main()
