"""Training data for the REA skill, generated on Modal with automatic ground truth.

Stage 1 (CPU, parallel): generate tasks with tools/rea-bench (python3 tools/rea-bench list shows the families), compile each
one stripped with the task's own compiler and -O level (`cc` and `opt` in its JSON; gcc and -O1/-O2/-Os when absent), decompile `proc`
(and any `helpers` it calls) with REA + Ghidra, store pseudocode + question + truth. A task whose asked constant is not in the code
compiled here (`facts` in its JSON, checked with this image's gcc/clang and objdump) is returned as an error record, not decompiled.
    modal run tools/modal/modal_rea_data.py --stage decompile --n 300 [--family legacy|all|crc,cipher,...] [--cc mix|gcc|clang]
        [--exclude-seed 1001 (default for any other seed; 0 = none)] [--force]
Stage 2 (GPU): an uncensored model answers each task K times at T=0.7; only answers whose HEADLINE matches the truth are kept
(rejection sampling, rea_bench.scoring), and what is stored is the headline plus at most one justification sentence (the rest is cut
off, `truncated: true`; a sentence that hedges or names another value vetoes the reply, rea_bench.sft). A task whose chance level is
0.2 or more (CRC init and xorout, pktsum, fsm) is kept only when at least 2 of the K samples are right. The stage checkpoints every
task on the volume (`--resume` continues a run that timed out) and reports yield per family in `.stats.json`. Output: SFT pairs in
results/phase6/rea-data/.
    modal run tools/modal/modal_rea_data.py --stage sample --model <name> [--seed 7 --family legacy] [--k 6] [--resume] [--force]
A seed-1001 run (the held-out split) may ask for at most 120 tasks per family: the list that keeps them out of training covers 120.
The default family is `legacy`: its whole-reply substring scoring (frozen, comparable with the 34/60 -> 50/60 numbers) accepts a
content-free reply on about 6% of its tasks; the stored target is cut like any other, but prefer `--family all` for new data.
Outputs are keyed by seed and family and never overwritten without --force (rea_bench.outputs):
    decompiled-seed<seed>-<family>.jsonl   and /decompiled-seed<seed>-<family>/ on the volume
    sft-<model>-seed<seed>-<family>.jsonl  (+ .stats.json); the sample stage needs the same --seed and --family as its decompile run

Nothing from the 10-question evaluation set (tools/modal/rea-bench/rea-tasks.json) is here; the generator uses other seeds and
its own templates, and the eval binary is a different program.
"""
from __future__ import annotations

import json
import pathlib
import re
import subprocess
import sys
import time

import modal

_HERE = pathlib.Path(__file__).resolve()
ROOT = _HERE.parents[2] if len(_HERE.parents) > 2 else _HERE.parent  # inside the Modal container the file sits at /root
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
RB = ROOT / "tools" / "rea-bench"          # the generator package; inside the container it is mounted at /root/rea_bench
if (RB / "rea_bench").is_dir():
    sys.path.insert(0, str(RB))
from modal_qbench import MODELS, models_vol  # noqa: E402
from modal_rea import image, chat, SYSTEM_RECIPE  # noqa: E402  (same REA/Ghidra image)
from rea_bench import dedupe, families, outputs, sft, survive  # noqa: E402  (stdlib-only)
from rea_bench.scoring import check_mode, score  # noqa: E402  (stdlib-only, unit-tested without Modal; same scorer modal_rea.py uses)

image = image.add_local_python_source("modal_rea")     # modal_rea's image already mounts rea_bench at /root/rea_bench
app = modal.App("lf-rea-data", image=image)
data_vol = modal.Volume.from_name("lf-rea-data", create_if_missing=True)


def _sh(cmd, cwd, timeout=900):
    try:
        p = subprocess.run(["bash", "-lc", cmd], capture_output=True, text=True, timeout=timeout, cwd=cwd)
    except subprocess.TimeoutExpired:       # a Ghidra hang is a failed task, not a failed run (it used to abort the whole starmap)
        return f"[timeout after {timeout}s] {cmd[:80]}"
    return p.stdout + p.stderr


def _decompile_fn(d: str, name: str):
    """Pseudocode of function `name`: its address comes from the unstripped t.sym, the decompile from the stripped t.
    Returns (code, None) or (None, error-fields dict)."""
    line = [l for l in _sh(f"nm t.sym | grep ' T {name}$'", d).splitlines()]
    if not line:
        return None, {"error": f"{name} not found"}
    addr = "0x" + line[0].split()[0].lstrip("0")
    j = _sh(f"rea decompile --procedure {addr} {d}/t --format json --provider ghidra", d)
    m = re.search(r'"normalized_result"\s*:\s*"((?:[^"\\]|\\.)*)"', j)
    if not m:
        return None, {"error": "no pseudocode", "raw": j[-800:]}
    return json.loads('"' + m.group(1) + '"'), None


@app.function(cpu=2, memory=4096, timeout=1800, max_containers=40, volumes={"/data": data_vol})
def decompile_task(task: dict, src: str, tag: str = "seed7-legacy") -> dict:
    try:
        return _decompile_task(task, src, tag)
    except Exception as e:      # never raise out of a remote function: one task must not abort the whole starmap
        return {**task, "error": f"decompile failed: {type(e).__name__}: {e}"[:300]}


def _decompile_task(task: dict, src: str, tag: str) -> dict:
    import os, tempfile
    d = tempfile.mkdtemp()
    pathlib.Path(d, "t.c").write_text(src)
    cc, opt = task.get("cc", "gcc"), task["opt"]
    _sh(f"{cc} {opt} -fno-lto -o t.sym t.c && {cc} {opt} -fno-lto -s -o t t.c", d)
    if task.get("facts"):   # the generator checked this with ITS compilers; these are the ones that built the binary
        try:
            ok, why = survive.check(task["facts"], src, cc, opt)
        except Exception as e:      # never raise out of a remote function: it aborts the whole starmap
            ok, why = False, f"gate failed: {e!r}"[:200]
        if not ok:
            return {**task, "error": f"fact not in the compiled code ({why})"}
    t0 = time.time()
    code, err = _decompile_fn(d, "proc")
    if err:
        return {**task, **err}
    for h in task.get("helpers", []):   # callees the question needs: their pseudocode follows proc's (names are FUN_<addr>, as in proc's calls)
        hcode, err = _decompile_fn(d, h)
        if err:
            return {**task, **err}
        code += "\n\n" + hcode
    refs = ""
    if "DAT_" in code or "PTR_" in code:
        pathlib.Path(d, "code.c").write_text(code)
        refs = _sh("python3 /bench/refdata t < code.c", d).strip()
    rec = {**task, "pseudocode": code, "refs": refs, "decompile_s": round(time.time() - t0, 1)}
    pathlib.Path(f"/data/{outputs.decompiled_stem(tag)}").mkdir(exist_ok=True)
    pathlib.Path(f"/data/{outputs.decompiled_stem(tag)}/{task['id']}.json").write_text(json.dumps(rec))
    data_vol.commit()
    return rec


# `score` lives in tools/rea-bench/rea_bench/scoring.py (imported above) so it can be unit-tested without Modal.


def _messages(t: dict) -> list:
    """The system and user message the sampler answers (and the SFT record is made of)."""
    extra = f"\n\nData referenced by the function (bytes at those addresses):\n{t['refs']}" if t.get("refs") else ""
    return [{"role": "system", "content": SYSTEM_RECIPE},
            {"role": "user", "content": f"```c\n{t['pseudocode']}\n```{extra}\n\nQuestion: {t['question']}"}]


@app.function(gpu=["L40S", "A100-40GB"], volumes={"/models": models_vol, "/data": data_vol}, timeout=5 * 3600)
def sample_answers(name: str, repo: str, fname: str, k: int, tag: str = "seed7-legacy", resume: bool = False) -> dict:
    """K replies per task, checkpointed task by task on the volume (`<sft stem>.partial.jsonl`): a run that hits the timeout is continued
    with resume=True instead of paid for twice. The SFT records are built at the end from all replies (rea_bench.sft: the stored text is
    the headline plus one sentence, a task of chance >= 0.2 needs two right samples); the stats come per family."""
    from huggingface_hub import hf_hub_download
    import urllib.request
    data_vol.reload()
    src_dir = pathlib.Path(f"/data/{outputs.decompiled_stem(tag)}")
    if not src_dir.is_dir() and tag.startswith("seed") and tag.endswith("-legacy"):
        src_dir = pathlib.Path(f"/data/decompiled-{tag[:-len('-legacy')]}")     # made before the dirs were keyed by family
    tasks = [json.loads(p.read_text()) for p in sorted(src_dir.glob("*.json"))]
    tasks = [t for t in tasks if "pseudocode" in t]
    if not tasks:
        raise SystemExit(f"no decompiled tasks in {src_dir} on the volume (run --stage decompile with the same --seed and --family first)")
    for t in tasks:
        check_mode(t)       # an unknown score mode must stop the run before the model download and the server start
    partial = pathlib.Path(f"/data/{outputs.sft_stem(name, tag)}.partial.jsonl")
    results = outputs.load_partial(partial, resume)
    todo = [t for t in tasks if t["id"] not in results]
    if results:
        print(f"resuming: {len(results)} of {len(tasks)} tasks already sampled")
    path = hf_hub_download(repo, fname, cache_dir="/models")
    srv = subprocess.Popen(["/app/llama-server", "-m", path, "--host", "127.0.0.1", "--port", "8080", "-ngl", "999", "-c", str(8192 * 4),
                            "--parallel", "4", "-fa", "off"], stdout=subprocess.DEVNULL, stderr=open("/tmp/server.log", "w"))
    for _ in range(600):
        try:
            if urllib.request.urlopen("http://127.0.0.1:8080/health", timeout=2).status == 200:
                break
        except Exception:
            time.sleep(1)
    from concurrent.futures import ThreadPoolExecutor

    def work(t):
        msgs = _messages(t)
        outs = []
        for i in range(k):
            try:
                body_seed = 1234 + i
                import urllib.request as ur
                body = {"model": "local", "messages": msgs, "temperature": 0.7, "seed": body_seed, "max_tokens": 300,
                        "chat_template_kwargs": {"enable_thinking": False}, "cache_prompt": False}
                req = ur.Request("http://127.0.0.1:8080/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
                with ur.urlopen(req, timeout=600) as r:
                    txt = json.load(r)["choices"][0]["message"].get("content") or ""
                txt = re.sub(r"<think>.*?</think>", "", txt, flags=re.S).strip()
            except Exception as e:
                txt = f"[error {e!r}]"
            outs.append((txt, score(txt, t)))
        return t, outs

    try:
        with ThreadPoolExecutor(4) as ex, open(partial, "a") as ck:
            for i, (t, outs) in enumerate(ex.map(work, todo), 1):
                rec = {"id": t["id"], "kind": t["kind"], "outs": outs}
                results[t["id"]] = rec
                ck.write(json.dumps(rec) + "\n")
                ck.flush()
                if i % 20 == 0:
                    data_vol.commit()
    finally:
        srv.kill()
        data_vol.commit()
    kept, stats = sft.assemble(tasks, results, name, _messages)
    return {"stats": stats, "sft": kept}


def _decompile_stage(names, n, seed, cc, exclude_seed, force, local_gate, tag):
    out_path = ROOT / "results" / "phase6" / "rea-data" / f"{outputs.decompiled_stem(tag)}.jsonl"
    outputs.refuse_overwrite([out_path], force)
    gate = survive.Gate() if local_gate else None       # looks for gcc/clang/objdump only when a task with facts is drawn; warns if one is missing
    if gate is None:
        print("warning: no local fact gate (--local-gate false): tasks whose asked constant the compiler removes are only dropped by the "
              "Modal-side re-check")
    kept, dropped = dedupe.training_tasks(names, n, seed, cc, exclude_seed, gate)
    if gate:
        print(gate.summary())
    if dropped:
        print(f"dropped {len(dropped)} of {len(kept) + len(dropped)} generated tasks that are in the held-out list (seed "
              f"{dedupe.exclusion_seed(seed, exclude_seed)}): {', '.join(sorted({t.kind for t in dropped}))}")
    tasks = [(t.meta(), t.source(), tag) for t in kept]
    for task, _src, _tag in tasks:
        check_mode(task)
    outputs.rm_volume_dir("lf-rea-data", f"/{outputs.decompiled_stem(tag)}", subprocess.run)      # this run's dir only; a failure stops the run
    ok = err = 0
    records = []

    def lines():
        nonlocal ok, err
        for (meta, _src, _t), res in zip(tasks, decompile_task.starmap(tasks, return_exceptions=True)):
            rec = outputs.as_record(meta, res)
            records.append(rec)
            ok += "pseudocode" in rec
            err += "error" in rec
            yield json.dumps(rec)

    outputs.write_atomic(out_path, lines())
    print(f"decompiled {ok} ok, {err} errors -> {out_path}")
    if err:
        info = outputs.summarize_errors(records)
        print("errors by reason: " + ", ".join(f"{why} {c}" for why, c in sorted(info["by_reason"].items(), key=lambda kv: -kv[1])))
        for why, kinds in sorted(info["by_reason_kind"].items()):
            print(f"  {why}: " + ", ".join(f"{k} {c}" for k, c in sorted(kinds.items())))


@app.local_entrypoint()
def main(stage: str = "decompile", n: int = 300, seed: int = 7, model: str = "", k: int = 6, family: str = "legacy", cc: str = "mix",
         exclude_seed: int = -1, force: bool = False, local_gate: bool = True, resume: bool = False):
    """exclude_seed: tasks of that seed never enter this run (-1: the held-out seed 1001 for any other seed, 0: none).
    local_gate: draw tasks again when the asked constant is not in the code compiled HERE (needs gcc, objdump and, for clang tasks,
    clang on PATH; without them the draws it cannot check are counted and warned about, and only the Modal-side re-check applies).
    resume: continue a sample stage that timed out from its checkpoint on the volume.
    Outputs are keyed by seed and family (rea_bench.outputs); an existing output is never overwritten without --force."""
    dest = ROOT / "results" / "phase6" / "rea-data"
    dest.mkdir(parents=True, exist_ok=True)
    tag = outputs.run_tag(seed, family)
    names = list(families.modules()) if family == "all" else family.split(",")
    unknown = [x for x in names if x not in families.modules()]
    if unknown:
        raise SystemExit(f"unknown family {unknown}; known: {', '.join(families.modules())} (or 'all')")
    if "legacy" in names:
        print("note: the legacy family is scored on the whole reply by substring (frozen for comparability); about 6% of its tasks accept a "
              "content-free reply. The stored target is still cut to headline + one sentence; --family all is the safer default for new data.")
    if stage == "decompile":
        dedupe.check_heldout_range(names, n, seed)          # the pinned held-out list covers 120 tasks per family
        _decompile_stage(names, n, seed, cc, exclude_seed, force, local_gate, tag)
    else:
        m = [x for x in MODELS if x[0] == model]
        if not m:
            raise SystemExit(f"unknown model {model}; known: {[x[0] for x in MODELS]}")
        if k < sft.AGREE_MIN:
            print(f"warning: k={k}: no task with a chance level of {sft.AGREE_CHANCE} or more (CRC init/xorout, pktsum, fsm) can have "
                  f"{sft.AGREE_MIN} right samples, so none of them is kept")
        name, repo, fname, _ = m[0]
        out, stats = (dest / f"{outputs.sft_stem(name, tag)}.{ext}" for ext in ("jsonl", "stats.json"))
        outputs.refuse_overwrite([out, stats], force)       # before a GPU is rented
        res = sample_answers.remote(name, repo, fname, k, tag, resume=resume)
        outputs.write_atomic(out, (json.dumps(r) for r in res["sft"]))
        stats.write_text(json.dumps(res["stats"], indent=1))
        s = res["stats"]
        print(name, tag, {f: s.get(f) for f in ("tasks", "answers", "correct", "solved_tasks", "kept_tasks", "kept_records", "agreement", "vetoed")}, "->", out)
        for kind, row in s.get("by_kind", {}).items():
            print(f"  {kind:10s} {row['solved_tasks']:4d}/{row['tasks']:<4d} solved, {row['kept_tasks']:4d} kept ({row['agreement']} dropped for lack of "
                  f"agreement, {row['vetoed']} vetoed)")
