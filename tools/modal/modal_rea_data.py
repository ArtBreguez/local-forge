"""Training data for the REA skill, generated on Modal with automatic ground truth.

Stage 1 (CPU, parallel): generate tasks with tools/rea-bench (python3 tools/rea-bench list shows the families), compile each
one stripped with the task's own compiler and -O level (`cc` and `opt` in its JSON; gcc and -O1/-O2/-Os when absent), decompile `proc`
(and any `helpers` it calls) with REA + Ghidra, store pseudocode + question + truth. A task whose asked constant is not in the code
compiled here (`facts` in its JSON, checked with this image's gcc/clang and objdump) is returned as an error record, not decompiled (a strobf task carries an `object_gate` instead: the plaintext must not be in the object, the stored bytes must). So is a task whose code
holds a `static const` table of which a byte is outside every window `refdata` prints (survive.check_visible: the model is never shown it, round 9).
    modal run tools/modal/modal_rea_data.py --stage decompile --n 300 [--family legacy|all|crc,cipher,...] [--cc mix|gcc|clang]
        [--exclude-seed 1001 (default for any other seed; 0 = none)] [--force]
Stage 2 (GPU): an uncensored model answers each task K times at T=0.7. The model is a FILTER, not the author of the data (round 4): a task
enters the data only when the HEADLINE of a reply parses completely as the task's answer (the whitelist grammar, rea_bench.scoring.score_strict,
for EVERY family including legacy), which proves the task is answerable from the pseudocode. What is STORED is never the model's text: the assistant
message is rendered from the task's own truth (rea_bench.targets: the canonical headline and one factual sentence from a per-variant set of 3-6
phrasings, chosen by a seeded RNG from the task id); the record keeps `teacher_headline` (the model's raw headline, for analysis only) and
`target_source: "truth-template"`. Free prose repeatedly carried a wrong fact past every filter, so none of it is stored. A failed request
(`[error ...]`) is never scored or stored and is counted per family; more than 5% of failed requests prints a warning.
A task whose chance level is 0.05 or more (the best constant guess within its variant: CRC, pktsum, fsm, TEA/XTEA,
ChaCha/Salsa, several structs, legacy skip) is kept only when at least 2 of the K samples are right; every reply is scored again at the end,
so a `--resume` never keeps the verdict of an older scorer. The stage checkpoints every task on the volume (`--resume` continues a run that timed
out) and reports yield per family in `.stats.json`. The held-out seed 1001 is
never sampled (the stage refuses it). Output: SFT pairs in results/phase6/rea-data/.
    modal run tools/modal/modal_rea_data.py --stage sample --model <name> [--seed 7 --family legacy] [--k 6] [--resume] [--force]
A seed-1001 decompile run (the held-out split) may ask for at most 120 tasks per family: the list that keeps them out of training covers 120.
The default family is `legacy`. Its frozen whole-reply substring scorer (comparable with the 34/60 -> 50/60 numbers, accepts a content-free
reply on about 6% of its tasks) is used by modal_rea.py's EVALUATION only; this stage converts a legacy task to the grammar first
(rea_bench.scoring.strict_task), so the legacy data is scored as strictly as any other family.
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
from rea_bench.scoring import check_mode, check_target, score_strict  # noqa: E402  (stdlib-only, unit-tested without Modal; NOT modal_rea.py's evaluation scorer)

image = image.add_local_python_source("modal_rea")     # modal_rea's image already mounts rea_bench at /root/rea_bench
app = modal.App("lf-rea-data", image=image)
data_vol = modal.Volume.from_name("lf-rea-data", create_if_missing=True)


def _sh(cmd, cwd, timeout=900):
    try:
        p = subprocess.run(["bash", "-lc", cmd], capture_output=True, text=True, timeout=timeout, cwd=cwd)
    except subprocess.TimeoutExpired:       # a Ghidra hang is a failed task, not a failed run (it used to abort the whole starmap)
        return f"[timeout after {timeout}s] {cmd[:80]}"
    return p.stdout + p.stderr


def _sh_rc(cmd, cwd, timeout=900):
    """(return code, stdout + stderr): a failed command is told apart from one that printed nothing (round 4, m10)."""
    try:
        p = subprocess.run(["bash", "-lc", cmd], capture_output=True, text=True, timeout=timeout, cwd=cwd)
    except subprocess.TimeoutExpired:
        return 124, f"[timeout after {timeout}s] {cmd[:80]}"
    return p.returncode, p.stdout + p.stderr


def _decompile_fn(d: str, name: str):
    """Pseudocode of function `name`: its address comes from the unstripped t.sym, the decompile from the stripped t.
    Returns (code, None) or (None, error-fields dict). Only the STDOUT of nm is read: a missing or empty t.sym prints `nm: 't.sym': No such file` on
    stderr, which used to become the address `0xnm:` and the error "no pseudocode"."""
    try:
        nm = subprocess.run(["bash", "-lc", f"nm t.sym | grep ' T {name}$'"], capture_output=True, text=True, timeout=900, cwd=d).stdout
    except subprocess.TimeoutExpired:
        return None, {"error": f"{name} not found", "raw": "[timeout after 900s] nm"}
    line = [l for l in nm.splitlines() if re.fullmatch(r"[0-9a-fA-F]+\s+T\s+" + re.escape(name), l.strip())]
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
    rc, out = _sh_rc(f"{cc} {opt} -fno-lto -o t.sym t.c && {cc} {opt} -fno-lto -s -o t t.c", d)
    if rc != 0:             # a compile error or a missing compiler is told as such, not as a decompile problem (round 4, m10)
        return {**task, "error": f"compile failed ({cc} {opt}, exit {rc}): " + " | ".join(out.strip().splitlines()[-3:])[:300]}
    facts = task.get("facts") or (survive.legacy_facts(task.get("kind", ""), task.get("accept", [])) if not task.get("variant") else [])
    if facts:               # the generator checked this with ITS compilers; these are the ones that built the binary (a frozen legacy lcg task has its multiplier as the only fact)
        try:
            ok, why = survive.check(facts, src, cc, opt)
        except Exception as e:      # never raise out of a remote function: it aborts the whole starmap
            ok, why = False, f"gate failed: {e!r}"[:200]
        if not ok:
            return {**task, "error": f"fact not in the compiled code ({why})"}
    if task.get("object_gate"):     # strobf: the compiled object must hold the string encoded, not decoded (round 5, M1)
        try:
            ok, why = survive.check_object(task["object_gate"], src, cc, opt)
        except Exception as e:
            ok, why = False, f"gate failed: {e!r}"[:200]
        if not ok:
            return {**task, "error": f"fact not in the compiled code ({why})"}
    try:                            # every family: a table of the code (static const array) must be inside what refdata prints (round 9, B1); the build is the one below, as a PIE with symbols
        ok, why = survive.check_visible(src, cc, opt, task.get("helpers", []))
    except Exception as e:
        ok, why = False, f"gate failed: {e!r}"[:200]
    if not ok:
        return {**task, "error": f"data not shown to the model ({why})"}
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


# `score_strict` lives in tools/rea-bench/rea_bench/scoring.py (imported above) so it can be unit-tested without Modal. The sample stage scores EVERY
# family, legacy included, with the whitelist grammar; the frozen whole-reply substring scorer stays in modal_rea.py's evaluation only.


def _messages(t: dict) -> list:
    """The system and user message the sampler answers (and the SFT record is made of)."""
    extra = f"\n\nData referenced by the function (bytes at those addresses):\n{t['refs']}" if t.get("refs") else ""
    return [{"role": "system", "content": SYSTEM_RECIPE},
            {"role": "user", "content": f"```c\n{t['pseudocode']}\n```{extra}\n\nQuestion: {t['question']}"}]


@app.function(gpu=["L40S", "A100-40GB"], volumes={"/models": models_vol, "/data": data_vol}, timeout=5 * 3600)
def sample_answers(name: str, repo: str, fname: str, k: int, tag: str = "seed7-legacy", resume: bool = False) -> dict:
    """K replies per task, checkpointed task by task on the volume (`<sft stem>.partial.jsonl`): a run that hits the timeout is continued
    with resume=True instead of paid for twice; a resume asks again the samples that failed (`[error ...]`) or are missing, and only those. The SFT
    records are built at the end from all replies (rea_bench.sft: the teacher's headline is a filter, the stored text is rendered from the truth, a
    task of chance >= 0.05 needs two right samples); the stats come per family."""
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
        check_target(t)     # so must a task without a truth template, or one whose rendered target its own scorer rejects
    partial = pathlib.Path(f"/data/{outputs.sft_stem(name, tag)}.partial.jsonl")
    results = outputs.load_partial(partial, resume)
    todo = [(t, outputs.samples_to_ask(results.get(t["id"]), k)) for t in tasks]
    todo = [(t, ask) for t, ask in todo if ask]
    if results:
        redo = sum(len(ask) for t, ask in todo if t["id"] in results)
        print(f"resuming: {len(tasks) - len(todo)} of {len(tasks)} tasks complete, {len(todo)} to ask ({redo} failed or missing samples of checkpointed tasks)")
    path = hf_hub_download(repo, fname, cache_dir="/models")
    server_log = open("/tmp/server.log", "w")
    srv = subprocess.Popen(["/app/llama-server", "-m", path, "--host", "127.0.0.1", "--port", "8080", "-ngl", "999", "-c", str(8192 * 4),
                            "--parallel", "4", "-fa", "off"], stdout=subprocess.DEVNULL, stderr=server_log)
    for _ in range(600):
        try:
            if urllib.request.urlopen("http://127.0.0.1:8080/health", timeout=2).status == 200:
                break
        except Exception:
            time.sleep(1)
    from concurrent.futures import ThreadPoolExecutor

    def work(item):
        t, ask = item
        msgs = _messages(t)
        outs = [tuple(o) for o in results[t["id"]]["outs"]] if t["id"] in results else []
        outs += [("", False)] * (k - len(outs))                    # a checkpointed task keeps its answers; only the samples in `ask` are asked
        for i in ask:
            try:
                body_seed = 1234 + i
                import urllib.request as ur
                body = {"model": "local", "messages": msgs, "temperature": 0.7, "seed": body_seed, "max_tokens": 300,
                        "chat_template_kwargs": {"enable_thinking": False}, "cache_prompt": False}
                req = ur.Request("http://127.0.0.1:8080/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
                with ur.urlopen(req, timeout=600) as r:
                    txt = json.load(r)["choices"][0]["message"].get("content") or ""
                txt = re.sub(r"<think>.*?</think>", "", txt, flags=re.S).strip()
                ok = score_strict(txt, t)
            except Exception as e:
                txt, ok = f"[error {e!r}]", False       # the text of a failed request is never scored (B1) and never stored; stats count it; --resume asks it again
            outs[i] = (txt, ok)
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
        server_log.close()
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
    report: dict = {}
    kept, dropped = dedupe.training_tasks(names, n, seed, cc, exclude_seed, gate, report)
    if gate:
        print(gate.summary())
    for why, count in report.get("untrainable", {}).items():
        print(f"dropped {count} generated tasks: {why}")
    if report.get("answer_overlap"):
        print(f"dropped {sum(report['answer_overlap'].values())} generated tasks whose program is not held out but whose ANSWER is one that a held-out task of the same variant has "
              f"(dedupe.ANSWER_DISJOINT): " + ", ".join(f"{k} {c}" for k, c in sorted(report["answer_overlap"].items())))
    for line in dedupe.dropped_summary(kept, dropped, dedupe.exclusion_seed(seed, exclude_seed)):
        print(line)
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
    if stage not in ("decompile", "sample"):         # `--stage decompil` used to rent a GPU for a sample stage (round 4, m4)
        raise SystemExit(f"unknown stage {stage!r}; known: decompile, sample")
    dest = ROOT / "results" / "phase6" / "rea-data"
    dest.mkdir(parents=True, exist_ok=True)
    tag = outputs.run_tag(seed, family)
    names = list(families.modules()) if family == "all" else family.split(",")
    unknown = [x for x in names if x not in families.modules()]
    if unknown:
        raise SystemExit(f"unknown family {unknown}; known: {', '.join(families.modules())} (or 'all')")
    if "legacy" in names:
        print("note: the legacy family keeps its frozen whole-reply substring scoring in modal_rea.py's evaluation (comparable with 34/60 -> "
              "50/60); this script's sample stage scores it with the strict headline grammar like every other family (scoring.score_strict).")
    if stage == "decompile":
        dedupe.check_heldout_range(names, n, seed)          # the pinned held-out list covers 120 tasks per family
        _decompile_stage(names, n, seed, cc, exclude_seed, force, local_gate, tag)
    else:
        dedupe.check_sample_seed(seed)                       # the held-out split is never sampled for training (m7)
        m = [x for x in MODELS if x[0] == model]
        if not m:
            raise SystemExit(f"unknown model {model}; known: {[x[0] for x in MODELS]}")
        if k < sft.AGREE_MIN:
            print(f"warning: k={k}: no task with a chance level of {sft.AGREE_CHANCE} or more (CRC init/xorout, pktsum, fsm, TEA/XTEA, ...) can have "
                  f"{sft.AGREE_MIN} right samples, so none of them is kept")
        name, repo, fname, _ = m[0]
        out, stats = (dest / f"{outputs.sft_stem(name, tag)}.{ext}" for ext in ("jsonl", "stats.json"))
        outputs.refuse_overwrite([out, stats], force)       # before a GPU is rented
        res = sample_answers.remote(name, repo, fname, k, tag, resume=resume)
        outputs.write_atomic(out, (json.dumps(r) for r in res["sft"]))
        stats.write_text(json.dumps(res["stats"], indent=1))
        s = res["stats"]
        print(name, tag, {f: s.get(f) for f in ("tasks", "answers", "errors", "correct", "solved_tasks", "kept_tasks", "kept_records", "agreement")}, "->", out)
        warning = sft.error_warning(s)
        if warning:
            print("\n" + "!" * 100 + f"\n{warning}\n" + "!" * 100 + "\n")
        for kind, row in s.get("by_kind", {}).items():
            print(f"  {kind:10s} {row['solved_tasks']:4d}/{row['tasks']:<4d} solved, {row['kept_tasks']:4d} kept ({row['agreement']} dropped for lack of "
                  f"agreement, {row.get('errors', 0)} failed requests)")
