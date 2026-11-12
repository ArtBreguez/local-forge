"""REA benchmark on Modal: can a local, uncensored model do reverse-engineering with REA + Ghidra?

Two arms per model, on 10 questions about 8 functions in one stripped Linux binary built from tools/rea-bench/targets.c
(ground truth: rea-tasks.json, written before any run):
  recipe : the harness runs REA/Ghidra (`rea decompile`), hands the model the pseudocode of the right function and the question,
           the model answers. Measures reading ability, which is what REA needs from a model.
  agent  : the model gets a shell with `rea`, the binary and the question, and must find the function itself (up to 8 tool
           calls, JSON tool protocol enforced by the harness). Measures tool use.
Scoring is rea_bench.scoring.score, the EVALUATION scorer: the frozen whole-reply substring modes for the fixed questions and the legacy templates (so 34/60 -> 50/60 stays
comparable), the whitelist headline grammar for the generated families. The data path (modal_rea_data.py --stage sample) does not use it: it scores every family, legacy
included, with rea_bench.scoring.score_strict. Recorded with the full transcript in results/phase6/rea/<model>.jsonl.
The generated held-out set (seed 1001) is a third arm, reported on its own line and never added to the fixed 10: each record has `pass` (strict:
the HEADLINE of the reply is the answer) and `pass_any_line` (a format-insensitive control: any line of the reply is). A base model that
thinks aloud on line 1 scores 0 strict and, if it answers later, 1 on the control; the gain of a fine-tune on the strict score is only
trustworthy as far as the control moves with it.

    modal run tools/modal_rea.py --only heretic-9b,qwen35-35b-a3b-heretic
"""
from __future__ import annotations

import json
import os
import pathlib
import re
import subprocess
import time

import modal

_HERE = pathlib.Path(__file__).resolve()
ROOT = _HERE.parents[2] if len(_HERE.parents) > 2 else _HERE.parent  # inside the Modal container the file sits at /root
import sys; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
RB = ROOT / "tools" / "rea-bench"          # the generator package; inside the container it is mounted at /root/rea_bench
if (RB / "rea_bench").is_dir():
    sys.path.insert(0, str(RB))
from modal_qbench import MODELS, LLAMA_DIGEST, models_vol  # noqa: E402  (same files, same volume)
from rea_bench import dedupe, outputs  # noqa: E402  (stdlib only)
from rea_bench.scoring import check_mode, score, score_any_line  # noqa: E402  (stdlib only; the EVALUATION scorers: the data pipeline scores with score_strict, see modal_rea_data.py)
lora_vol = modal.Volume.from_name("lf-lora", create_if_missing=True)
# Our own GGUFs (base and LoRA quantized by the same path, tools/modal_lora.py): repo "vol:" = file inside the lf-lora volume
LOCAL_MODELS = [
    ("heretic-9b-base-q4km", "vol:", "/lora/full/gguf/base-Q4_K_M.gguf", "L4"),
    ("heretic-9b-rea-lora-q4km", "vol:", "/lora/full/gguf/lora-Q4_K_M.gguf", "L4"),
]

JDK_URL = "https://github.com/adoptium/temurin25-binaries/releases/download/jdk-25.0.5%2B8/OpenJDK25U-jdk_x64_linux_hotspot_25.0.5_8.tar.gz"
JDK_SHA = "68982529869257d0ee6b824ec5aa270ff8b118bb438670da0b97ebdab178d7fb"
GHIDRA_URL = "https://github.com/NationalSecurityAgency/ghidra/releases/download/Ghidra_12.1.4_build/ghidra_12.1.4_PUBLIC_20260921.zip"
GHIDRA_SHA = "ddac49f903da9d5bac833e5cc79395098b9c33cfd3279be5f31bd00387d2d4db"
NODE_URL = "https://nodejs.org/dist/v24.22.2/node-v24.22.2-linux-x64.tar.xz"
NODE_SHA = "48cc0e21e048790e54c0cf5749c2725005313d652651c40b7a21feb9697aaf75"
REA_PKG = "rea-agents@6.8.0"   # same version as the host install (tools/rea-install.ps1)

image = (
    modal.Image.from_registry(f"ghcr.io/ggml-org/llama.cpp@{LLAMA_DIGEST}", add_python="3.11")
    .entrypoint([])
    .apt_install("curl", "unzip", "xz-utils", "gcc", "clang", "binutils", "ca-certificates")
    .run_commands(
        f"curl -fsSL -o /tmp/jdk.tgz {JDK_URL} && echo '{JDK_SHA}  /tmp/jdk.tgz' | sha256sum -c - && mkdir -p /opt/jdk && tar -xzf /tmp/jdk.tgz -C /opt/jdk --strip-components=1 && rm /tmp/jdk.tgz",
        f"curl -fsSL -o /tmp/ghidra.zip {GHIDRA_URL} && echo '{GHIDRA_SHA}  /tmp/ghidra.zip' | sha256sum -c - && unzip -q /tmp/ghidra.zip -d /opt && mv /opt/ghidra_12.1.4_PUBLIC /opt/ghidra && rm /tmp/ghidra.zip",
        f"curl -fsSL -o /tmp/node.txz {NODE_URL} && echo '{NODE_SHA}  /tmp/node.txz' | sha256sum -c - && mkdir -p /opt/node && tar -xJf /tmp/node.txz -C /opt/node --strip-components=1 && rm /tmp/node.txz",
        f"export PATH=/opt/node/bin:$PATH && mkdir -p /opt/rea && cd /opt/rea && npm init -y >/dev/null && npm install --ignore-scripts --no-optional --no-audit --no-fund {REA_PKG} >/dev/null && ls node_modules/.bin/rea",
    )
    .env({"JAVA_HOME": "/opt/jdk", "GHIDRA_INSTALL_DIR": "/opt/ghidra", "PATH": "/opt/node/bin:/opt/jdk/bin:/opt/rea/node_modules/.bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"})
    .pip_install("huggingface_hub[hf_transfer]==0.36.0")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .add_local_dir(ROOT / "tools" / "modal" / "rea-bench", "/bench")
    .add_local_python_source("modal_qbench")
)
if (RB / "rea_bench").is_dir():
    image = image.add_local_dir(str(RB / "rea_bench"), "/root/rea_bench")   # last layer, like modal_rea_data.py had it
app = modal.App("lf-rea", image=image)

SYSTEM_RECIPE = ("You are a reverse engineer. You are given decompiled C pseudocode of one function from a stripped binary and a "
                 "question. Answer in one short line with the concrete value or name asked for, then one sentence of justification.")
SYSTEM_AGENT = ("You are a reverse engineer working in a shell on a Linux box. The binary is stripped. Narrow tools (wrappers around REA 6.8 "
                "+ Ghidra 12.1.4): `funcs <file>` lists function addresses and sizes; `decomp <file> <address>` prints the Ghidra pseudocode "
                "of one function; `decomp-all <file>` prints the pseudocode of every small function at once (use it first); "
                "`refdata <file> <address>` shows the bytes/string behind a DAT_ label; `strings`, `objdump` also exist. Typical plan: "
                "decomp-all, pick the function that matches the question, refdata for any DAT_ it uses, then answer. To act, reply with exactly one JSON object on one line: {\"cmd\": \"<shell command>\"}. "
                "When you know the answer reply {\"answer\": \"<short answer with the concrete value>\"}. No other text. At most 12 commands.")


def chat(messages, max_tokens=400, temperature=0.0):
    import urllib.request
    body = {"model": "local", "messages": messages, "temperature": temperature, "seed": 1234, "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False}, "cache_prompt": False}
    req = urllib.request.Request("http://127.0.0.1:8080/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        d = json.load(r)
    msg = d["choices"][0]["message"]
    txt = msg.get("content") or ""
    txt = re.sub(r"<think>.*?</think>", "", txt, flags=re.S).strip()
    return txt, d.get("usage", {})


# `score` is rea_bench.scoring.score (imported above): legacy modes for the 10 fixed questions and the old templates, headline modes for
# the generated families. An unknown mode raises, so main() checks every task before a GPU is rented.


def sh(cmd: str, timeout=300) -> str:
    try:
        p = subprocess.run(["bash", "-lc", cmd], capture_output=True, text=True, timeout=timeout, cwd="/work")
        out = (p.stdout + p.stderr)
    except subprocess.TimeoutExpired:
        out = "[timeout]"
    return out[-6000:]


def decompile_all(binary: str) -> dict:
    """Ghidra via REA once per binary: {function_name: pseudocode}. Functions are found by their argument order in the
    unstripped sidecar (compiled twice; the stripped one is what models see), then mapped by address."""
    addr = {}
    for line in sh(f"nm {binary}.sym | grep ' T tag_'").splitlines():
        a, _, name = line.split()
        addr[name] = "0x" + a.lstrip("0")
    out = {}
    for name, a in addr.items():
        j = sh(f"rea decompile --procedure {a} {binary} --format json --provider ghidra", timeout=900)
        m = re.search(r'"normalized_result"\s*:\s*"((?:[^"\\]|\\.)*)"', j)
        code = json.loads('"' + m.group(1) + '"') if m else j[-1500:]
        out[name] = code
    return out


def run_heldout(tasks: list, ask) -> list:
    """The held-out arm: one record per task with the strict score (`pass`: the headline is the answer) and the control (`pass_any_line`).
    `ask(messages) -> (reply, usage)`. A call that raises is recorded as an error and the run goes on."""
    out = []
    for t in tasks:
        extra = f"\n\nData referenced by the function (bytes at those addresses):\n{t['refs']}" if t.get("refs") else ""
        try:
            ans, _usage = ask([{"role": "system", "content": SYSTEM_RECIPE},
                               {"role": "user", "content": f"```c\n{t['pseudocode']}\n```{extra}\n\nQuestion: {t['question']}"}])
            rec = {"pass": score(ans, t), "pass_any_line": score_any_line(ans, t), "answer": ans[:600]}
        except Exception as e:      # one timeout must not lose the whole model run (the fixed arms and the agent arm)
            rec = {"pass": False, "pass_any_line": False, "answer": "", "error": repr(e)[:300]}
        out.append({"arm": "heldout", "task": t["id"], "kind": t["kind"], **rec})
    return out


def run_model(name: str, repo: str, fname: str, heldout: list | None = None) -> dict:
    import shutil
    from huggingface_hub import hf_hub_download
    t0 = time.time()
    os.makedirs("/work", exist_ok=True)
    shutil.copy("/bench/targets.c", "/work/targets.c")
    sh("gcc -O1 -fno-lto -o target.sym targets.c && gcc -O1 -fno-lto -s -o target targets.c && rm targets.c")
    for tool in ("funcs", "decomp", "refdata", "decomp-all"):
        shutil.copy(f"/bench/{tool}", f"/usr/local/bin/{tool}"); os.chmod(f"/usr/local/bin/{tool}", 0o755)
    tasks = json.load(open("/bench/rea-tasks.json"))["tasks"]
    pseudo = decompile_all("/work/target")
    if repo == "vol:":
        lora_vol.reload(); path = fname
    else:
        path = hf_hub_download(repo, fname, cache_dir="/models")
        models_vol.commit()
    srv = subprocess.Popen(["/app/llama-server", "-m", path, "--host", "127.0.0.1", "--port", "8080", "-ngl", "999", "-c", "32768",
                            "-fa", "off"], stdout=subprocess.DEVNULL, stderr=open("/tmp/server.log", "w"))
    import urllib.request
    for _ in range(600):
        try:
            if urllib.request.urlopen("http://127.0.0.1:8080/health", timeout=2).status == 200:
                break
        except Exception:
            time.sleep(1)
    records = []
    try:
        for t in tasks:
            code = pseudo.get(t["function"], "")
            pathlib.Path("/tmp/code.c").write_text(code)
            refs = sh("refdata /work/target < /tmp/code.c") if "DAT_" in code or "PTR_" in code else ""
            extra = f"\n\nData referenced by the function (bytes at those addresses):\n{refs.strip()}" if refs.strip() else ""
            ans, usage = chat([{"role": "system", "content": SYSTEM_RECIPE},
                               {"role": "user", "content": f"```c\n{code}\n```{extra}\n\nQuestion: {t['question']}"}])
            records.append({"arm": "recipe", "task": t["id"], "pass": score(ans, t), "answer": ans[:600], "usage": usage})
        records += run_heldout(heldout or [], chat)      # generated held-out set (seed 1001), never used for training
        for t in tasks:
            msgs = [{"role": "system", "content": SYSTEM_AGENT},
                    {"role": "user", "content": f"Binary: /work/target (stripped ELF x86-64). Question: {t['question']} "
                                                f"Hint: the function of interest is the one that matches the question; there are 8 small functions plus main."}]
            trace, ans, invalid = [], "", 0
            for step in range(12):
                txt, _ = chat(msgs, max_tokens=300)
                m = re.search(r"\{.*\}", txt, flags=re.S)
                try:
                    obj = json.loads(m.group(0)) if m else None
                except json.JSONDecodeError:
                    obj = None
                if not isinstance(obj, dict) or not ({"cmd", "answer"} & set(obj)):
                    invalid += 1
                    msgs.append({"role": "assistant", "content": txt})
                    msgs.append({"role": "user", "content": "Invalid. Reply with exactly one JSON object: {\"cmd\": ...} or {\"answer\": ...}."})
                    if invalid >= 3:
                        break
                    continue
                if "answer" in obj:
                    ans = str(obj["answer"]); trace.append({"answer": ans}); break
                cmd = str(obj["cmd"])[:500]
                if re.search(r"\brm\b|mkfs|>\s*/|dd\s", cmd):
                    out = "[refused]"
                else:
                    out = sh(cmd, timeout=600)
                trace.append({"cmd": cmd, "out": out[-1500:]})
                msgs.append({"role": "assistant", "content": json.dumps(obj)})
                msgs.append({"role": "user", "content": f"$ {cmd}\n{out[-3500:]}"})
            records.append({"arm": "agent", "task": t["id"], "pass": score(ans, t), "answer": ans[:600], "steps": len(trace),
                            "invalid_tool_calls": invalid, "trace": trace})
    finally:
        srv.kill()
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    return {"meta": {"model": name, "repo": repo, "file": fname, "gpu": gpu, "wall_s": round(time.time() - t0),
                     "ghidra": "12.1.4", "rea": REA_PKG, "pseudocode_chars": {k: len(v) for k, v in pseudo.items()}},
            "records": records, "pseudocode": pseudo}


@app.function(gpu=["L4", "A10", "L40S"], volumes={"/models": models_vol, "/lora": lora_vol}, timeout=3 * 3600, max_containers=8)
def run_l4(name, repo, fname, heldout=None):
    return run_model(name, repo, fname, heldout)


@app.function(gpu=["L40S", "A100-40GB"], volumes={"/models": models_vol, "/lora": lora_vol}, timeout=3 * 3600, max_containers=4)
def run_l40s(name, repo, fname, heldout=None):
    return run_model(name, repo, fname, heldout)


@app.local_entrypoint()
def main(only: str = ""):
    dest = ROOT / "results" / "phase6" / "rea"
    dest.mkdir(parents=True, exist_ok=True)
    todo = [m for m in MODELS + LOCAL_MODELS if not only or m[0] in only.split(",")]
    heldout = outputs.load_tasks(outputs.heldout_files(dest.parent / "rea-data"))   # decompiled-seed1001*.jsonl, one per family run
    over = dedupe.heldout_overflow(heldout)
    if over:
        raise SystemExit(f"the held-out files hold more tasks than the pinned held-out list covers ({dedupe.HELDOUT_N} per family): "
                         f"{', '.join(f'{f} {c}' for f, c in over.items())}. Tasks beyond the list may be in training data; regenerate with "
                         f"`--seed {dedupe.HELDOUT_SEED} --n <= {dedupe.HELDOUT_N}` per family")
    for t in json.loads((ROOT / "tools" / "modal" / "rea-bench" / "rea-tasks.json").read_text())["tasks"] + heldout:
        check_mode(t)       # ValueError on an unknown score mode now, not half way through a paid run
    calls = [((run_l40s if gpu == "L40S" else run_l4).spawn(name, repo, fname, heldout), name) for name, repo, fname, gpu in todo]
    for call, name in calls:
        try:
            res = call.get()
        except Exception as e:
            print(f"{name}: FAILED {e!r}"[:600]); continue
        (dest / f"{name}.jsonl").write_text("\n".join(json.dumps(r) for r in res["records"]) + "\n")
        (dest / f"{name}.meta.json").write_text(json.dumps(res["meta"], indent=1))
        (dest / "pseudocode.json").write_text(json.dumps(res["pseudocode"], indent=1))
        rec = [r for r in res["records"] if r["arm"] == "recipe"]; ag = [r for r in res["records"] if r["arm"] == "agent"]
        print(f"{name}: fixed questions: recipe {sum(r['pass'] for r in rec)}/{len(rec)}  agent {sum(r['pass'] for r in ag)}/{len(ag)}  "
              f"invalid {sum(r['invalid_tool_calls'] for r in ag)}  {res['meta']['gpu']} {res['meta']['wall_s']}s")
        hs = outputs.heldout_summary(res["records"])
        (dest / f"{name}.heldout.json").write_text(json.dumps(hs, indent=1))
        if hs:
            a = hs["ALL"]
            print(f"{name}: generated held-out set (separate from the fixed questions): strict {a['strict']}/{a['n']}  "
                  f"format-insensitive control {a['control']}/{a['n']}  errors {a['errors']}")
            for kind, row in sorted(hs.items()):
                if kind != "ALL":
                    print(f"    {kind:10s} strict {row['strict']:3d}/{row['n']:<3d} control {row['control']:3d}/{row['n']:<3d}" + (f"  errors {row['errors']}" if row["errors"] else ""))
