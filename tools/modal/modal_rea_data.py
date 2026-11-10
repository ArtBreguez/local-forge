"""Training data for the REA skill, generated on Modal with automatic ground truth.

Stage 1 (CPU, parallel): generate tasks with tools/rea-bench (python3 tools/rea-bench list shows the families), compile each
one stripped with the task's own compiler and -O level (`cc` and `opt` in its JSON; gcc and -O1/-O2/-Os when absent), decompile `proc`
(and any `helpers` it calls) with REA + Ghidra, store pseudocode + question + truth.
    modal run tools/modal/modal_rea_data.py --stage decompile --n 300 [--family legacy|all|crc,cipher,...] [--cc mix|gcc|clang]
Stage 2 (GPU): an uncensored model answers each task K times at T=0.7; only answers that match the truth are kept
(rejection sampling). Output: SFT pairs in results/phase6/rea-data/.
    modal run tools/modal/modal_rea_data.py --stage sample --model <name>

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
from rea_bench.scoring import check_mode, score  # noqa: E402  (stdlib-only, unit-tested without Modal; same scorer modal_rea.py uses)

image = image.add_local_python_source("modal_rea")     # modal_rea's image already mounts rea_bench at /root/rea_bench
app = modal.App("lf-rea-data", image=image)
data_vol = modal.Volume.from_name("lf-rea-data", create_if_missing=True)


def _sh(cmd, cwd, timeout=900):
    p = subprocess.run(["bash", "-lc", cmd], capture_output=True, text=True, timeout=timeout, cwd=cwd)
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
def decompile_task(task: dict, src: str, seed: int = 7) -> dict:
    import os, tempfile
    d = tempfile.mkdtemp()
    pathlib.Path(d, "t.c").write_text(src)
    cc, opt = task.get("cc", "gcc"), task["opt"]
    _sh(f"{cc} {opt} -fno-lto -o t.sym t.c && {cc} {opt} -fno-lto -s -o t t.c", d)
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
    pathlib.Path(f"/data/decompiled-seed{seed}").mkdir(exist_ok=True)
    pathlib.Path(f"/data/decompiled-seed{seed}/{task['id']}.json").write_text(json.dumps(rec))
    data_vol.commit()
    return rec


# `score` lives in tools/rea-bench/rea_bench/scoring.py (imported above) so it can be unit-tested without Modal.


@app.function(gpu=["L40S", "A100-40GB"], volumes={"/models": models_vol, "/data": data_vol}, timeout=5 * 3600)
def sample_answers(name: str, repo: str, fname: str, k: int, seed: int = 7) -> dict:
    from huggingface_hub import hf_hub_download
    import urllib.request
    data_vol.reload()
    tasks = [json.loads(p.read_text()) for p in sorted(pathlib.Path(f"/data/decompiled-seed{seed}").glob("*.json"))]
    tasks = [t for t in tasks if "pseudocode" in t]
    for t in tasks:
        check_mode(t)       # an unknown score mode must stop the run before the model download and the server start
    path = hf_hub_download(repo, fname, cache_dir="/models")
    srv = subprocess.Popen(["/app/llama-server", "-m", path, "--host", "127.0.0.1", "--port", "8080", "-ngl", "999", "-c", str(8192 * 4),
                            "--parallel", "4", "-fa", "off"], stdout=subprocess.DEVNULL, stderr=open("/tmp/server.log", "w"))
    for _ in range(600):
        try:
            if urllib.request.urlopen("http://127.0.0.1:8080/health", timeout=2).status == 200:
                break
        except Exception:
            time.sleep(1)
    kept, stats = [], {"tasks": len(tasks), "answers": 0, "correct": 0, "solved_tasks": 0}
    from concurrent.futures import ThreadPoolExecutor

    def work(t):
        extra = f"\n\nData referenced by the function (bytes at those addresses):\n{t['refs']}" if t.get("refs") else ""
        msgs = [{"role": "system", "content": SYSTEM_RECIPE},
                {"role": "user", "content": f"```c\n{t['pseudocode']}\n```{extra}\n\nQuestion: {t['question']}"}]
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
        with ThreadPoolExecutor(4) as ex:
            for t, outs in ex.map(work, tasks):
                good = [o for o, ok in outs if ok]
                stats["answers"] += len(outs); stats["correct"] += len(good); stats["solved_tasks"] += bool(good)
                for g in dict.fromkeys(good):   # distinct correct answers
                    extra = f"\n\nData referenced by the function (bytes at those addresses):\n{t['refs']}" if t.get("refs") else ""
                    kept.append({"messages": [{"role": "system", "content": SYSTEM_RECIPE},
                                              {"role": "user", "content": f"```c\n{t['pseudocode']}\n```{extra}\n\nQuestion: {t['question']}"},
                                              {"role": "assistant", "content": g}],
                                 "task": t["id"], "kind": t["kind"], "model": name})
    finally:
        srv.kill()
    return {"stats": stats, "sft": kept}


@app.local_entrypoint()
def main(stage: str = "decompile", n: int = 300, seed: int = 7, model: str = "", k: int = 6, family: str = "legacy", cc: str = "mix"):
    dest = ROOT / "results" / "phase6" / "rea-data"
    dest.mkdir(parents=True, exist_ok=True)
    if stage == "decompile":
        import tempfile
        # one seed dir per run: a new family needs its own --seed (or --family all), otherwise this wipes the previous one
        subprocess.run(["modal", "volume", "rm", "-r", "lf-rea-data", f"/decompiled-seed{seed}"], capture_output=True)
        d = tempfile.mkdtemp()
        subprocess.run([sys.executable, str(RB), "gen", "--family", family, "--n", str(n), "--seed", str(seed), "--cc", cc, "--out", d], check=True)
        tasks = [(json.loads(pathlib.Path(d, f"{p.stem}.json").read_text()), p.read_text(), seed) for p in sorted(pathlib.Path(d).glob("*.c"))]
        for task, _src, _seed in tasks:
            check_mode(task)
        ok = err = 0
        with open(dest / f"decompiled-seed{seed}.jsonl", "w") as out:
            for rec in decompile_task.starmap(tasks):
                out.write(json.dumps(rec) + "\n")
                ok += "pseudocode" in rec; err += "error" in rec
        print(f"decompiled {ok} ok, {err} errors -> {dest}")
    else:
        m = [x for x in MODELS if x[0] == model]
        if not m:
            raise SystemExit(f"unknown model {model}; known: {[x[0] for x in MODELS]}")
        name, repo, fname, _ = m[0]
        res = sample_answers.remote(name, repo, fname, k, seed)
        (dest / f"sft-{name}.jsonl").write_text("\n".join(json.dumps(r) for r in res["sft"]) + "\n")
        (dest / f"sft-{name}.stats.json").write_text(json.dumps(res["stats"], indent=1))
        print(name, res["stats"])
