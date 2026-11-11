"""Kaggle GPU kernel: a teacher model answers REA tasks, and `python3 tools/rea-bench kaggle-score` turns the replies into SFT records (tools/kaggle/README.md has the commands).

Standard library only (Kaggle's image has no copy of rea_bench): it is pushed as the kernel's `run.py` by `tools/kaggle/build_tasks.py`. Everything that decides what is asked and what is stored is a
function of this file, so tests/test_rea_kaggle_kit.py runs it against a stub server and checks it against `rea_bench.reasoning` (the thinking sampling, the stored text, the prompt fingerprint).

What it does, in order:
  1. reads `tasks.jsonl` ({id, kind, messages}: the exact messages of `reasoning.messages(task)`, SYSTEM_REASONING + the user message) and `config.json` from the attached dataset;
  2. gets a llama-server with CUDA for the T4 (sm_75): a prebuilt one from an attached dataset (the `llama-bin` folder a previous run saved), else it builds llama.cpp (about 30 minutes;
     the libcuda stub fix: Kaggle has the driver library but not where cmake looks, so its path is passed as CUDA_cuda_driver_LIBRARY);
  3. downloads the GGUF, starts `llama-server --jinja --reasoning-format deepseek` over the two T4 (layer split), `slots` requests in flight, `ctx_per_slot` tokens each;
  4. asks every task `k` times with the sampling of the config (a thinking run: Qwen3's recommended settings, never greedy), `enable_thinking` as the config says, `max_tokens` per reply;
  5. appends each finished task to `/kaggle/working/replies.jsonl` at once (a checkpoint) and stops starting requests when the deadline comes (`deadline_h` hours after the start);
  6. resumes: every `replies.jsonl` found under /kaggle/input (a previous session's output attached to the kernel or uploaded as a dataset) is read first; only the samples that are missing or failed
     are asked again; a task that is complete is not;
  7. writes `run.json` (settings, throughput, llama.cpp commit, what was skipped) next to the replies, which `kaggle-score` reads.

A line of `replies.jsonl`: {"id", "kind", "prompt_sha", "outs": [{"text", "finish", "tokens", "prompt_tokens", "secs"}]}. In a thinking run `text` is `<think>\\nreasoning\\n</think>\\n\\nvisible reply`
(the server's `reasoning_content` and `content`: `stored_text`, the form `kaggle-score --thinking` reads); `finish` is the server's finish_reason ("length" = cut off at max_tokens, which is rejected
downstream), or "error" with `[error ...]` as the text for a request that failed (never scored). `prompt_sha` fingerprints the messages that were sent; `kaggle-score` refuses a reply to another prompt.
Nothing is scored here: the truth is not in the dataset.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

# The thinking settings and the length rules are the ones of rea_bench/reasoning.py (THINKING_SAMPLING, DEFAULT_SAMPLING, DEFAULT_THINKING_MAX_TOKENS, ...); tests/test_rea_kaggle_kit.py pins every copy.
THINKING_SAMPLING = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0}      # Qwen3's recommended thinking settings
DEFAULT_SAMPLING = {"temperature": 0.7}                                                 # a run that does not think: phase A's
LENGTH_OVERHEAD = 32                                                                    # tokens the chat template adds around the messages (rea_bench.sft.LENGTH_OVERHEAD)
SAMPLING_KEYS = tuple(THINKING_SAMPLING)

DEFAULTS = {
    "model_repo": "mradermacher/Qwen3.8-27B-CodeForge-3xMerge-UNCENSORED-heretic-i1-GGUF",
    "model_file": "Qwen3.8-27B-CodeForge-3xMerge-UNCENSORED-heretic.i1-Q4_K_M.gguf",
    "k": 4,                       # replies per task
    "thinking": True,             # `enable_thinking`: the model's native reasoning
    "slots": 4,                   # requests in flight (llama-server --parallel)
    "ctx_per_slot": 8192,         # tokens one request may take: the prompt and max_tokens (4096 + 4096)
    "deadline_h": 10.5,           # hours after the start at which no new request is started (a Kaggle GPU session ends at 12 h; a reply in flight may take up to request_timeout_s more)
    "request_timeout_s": 1800,    # a 4096-token reply at ~4.6 tokens/s per slot (27B Q4 on two T4, four slots, measured 18 tokens/s in all) takes ~15 minutes
    "seed_base": 1234,            # sample i is asked with seed seed_base + i, as the Modal stage does
    "reasoning_format": "deepseek",     # llama-server --reasoning-format: the reasoning comes back as `reasoning_content`
    "llama_cpp_ref": "",          # a llama.cpp tag or commit to build ("" = the head of the default branch; run.json records what was built)
}

T0 = time.time()
OUT = pathlib.Path("/kaggle/working")
INPUT = pathlib.Path("/kaggle/input")
URL = "http://127.0.0.1:8080"


def log(*a, log_file=None):
    line = f"[{time.time() - T0:7.0f}s] " + " ".join(str(x) for x in a)
    print(line, flush=True)
    with open(log_file or OUT / "run_log.txt", "a") as f:
        f.write(line + "\n")


def sh(cmd, **kw):
    log("$", cmd if isinstance(cmd, str) else " ".join(cmd))
    return subprocess.run(cmd, shell=isinstance(cmd, str), check=True, **kw)


# ---- configuration -------------------------------------------------------------------------------------------------------------------------

def resolve_config(user: dict) -> dict:
    """DEFAULTS, then the dataset's config.json. Whatever it leaves out follows `thinking`: a thinking run samples with THINKING_SAMPLING and asks for 4096 tokens, any other run with DEFAULT_SAMPLING
    and 768. The flat keys `temperature`, `top_p`, `top_k`, `min_p` of an older config.json are folded into `sampling`. A thinking run is never greedy (ValueError for temperature 0 or top_k 1)."""
    cfg = {**DEFAULTS, **user}
    thinking = bool(cfg["thinking"])
    sampling = dict(user.get("sampling") or (THINKING_SAMPLING if thinking else DEFAULT_SAMPLING))
    sampling.update({k: user[k] for k in SAMPLING_KEYS if k in user and "sampling" not in user})
    if thinking and (sampling.get("temperature", 0) <= 0 or sampling.get("top_k") == 1):
        raise ValueError(f"a thinking run is never greedy: {sampling} (temperature 0 or top_k 1 makes the reasoning loop); Qwen3 recommends {THINKING_SAMPLING}")
    cfg["thinking"], cfg["sampling"] = thinking, sampling
    cfg["max_tokens"] = int(user.get("max_tokens", 4096 if thinking else 768))
    return cfg


def request_body(cfg: dict, messages: list, i: int) -> dict:
    """The request of sample `i`: the messages of the task exactly as the dataset has them, the config's sampling, `enable_thinking`, the token budget. The same body as the Modal stage's sampler."""
    return {"model": "local", "messages": messages, **cfg["sampling"], "seed": cfg["seed_base"] + i, "max_tokens": cfg["max_tokens"],
            "chat_template_kwargs": {"enable_thinking": bool(cfg["thinking"])}, "cache_prompt": False}


def prompt_sha(messages: list) -> str:
    """rea_bench.reasoning.prompt_sha: a fingerprint of the messages that were sent, written with every line so that `kaggle-score` can refuse a reply to another prompt."""
    return hashlib.sha256(json.dumps(messages, sort_keys=True, ensure_ascii=True).encode()).hexdigest()[:16]


def stored_text(content, reasoning_content, thinking: bool) -> str:
    """What `text` holds. A thinking run whose server returned `reasoning_content`: `<think>\\nreasoning\\n</think>\\n\\ncontent`, the trace form `kaggle-score --thinking` expects (`reasoning_content` is not
    stored apart: one text, one way to read it). Anything else (the server left the reasoning inline in `content`, or the run does not think): `content` as the server sent it; kaggle-score reads
    an inline `<think>...</think>` the same way."""
    content = content or ""
    reasoning = (reasoning_content or "").strip()
    return f"<think>\n{reasoning}\n</think>\n\n{content.strip()}" if thinking and reasoning else content


# ---- one request, one task -----------------------------------------------------------------------------------------------------------------

def is_failed(out) -> bool:
    return not isinstance(out, dict) or out.get("finish") == "error" or str(out.get("text", "")).startswith("[error")


def samples_to_ask(have: list, k: int) -> list:
    """The sample indices to ask for a task whose earlier replies are `have` (outs in sample order): every index the list does not reach, and every one whose reply is a failed request. A cut-off reply
    ("length") is an answer and is not asked again (it is rejected downstream, and asking again with the same seed and budget gives the same cut)."""
    return [i for i in range(k) if i >= len(have) or is_failed(have[i])]


def can_start(now: float, deadline: float, mean_secs: float) -> bool:
    """True when a request that takes the mean time of the finished ones still ends before the deadline."""
    return now + mean_secs <= deadline


def ask_one(cfg: dict, messages: list, i: int, base_url: str = URL, urlopen=None) -> dict:
    """One reply: {"text", "finish", "tokens" (completion), "prompt_tokens", "secs"}. A request that fails is {"text": "[error ...]", "finish": "error"}: it is never scored, and a resume asks it again."""
    urlopen = urlopen or urllib.request.urlopen
    req = urllib.request.Request(base_url + "/v1/chat/completions", json.dumps(request_body(cfg, messages, i)).encode(), {"Content-Type": "application/json"})
    start = time.time()
    try:
        r = json.load(urlopen(req, timeout=cfg["request_timeout_s"]))
        choice = r["choices"][0]
        msg = choice["message"]
        usage = r.get("usage") or {}
        return {"text": stored_text(msg.get("content"), msg.get("reasoning_content"), cfg["thinking"]), "finish": choice.get("finish_reason"), "tokens": usage.get("completion_tokens", 0),
                "prompt_tokens": usage.get("prompt_tokens", 0), "secs": round(time.time() - start, 1)}
    except Exception as e:      # never scored downstream
        return {"text": f"[error {type(e).__name__}: {e}]", "finish": "error", "tokens": 0, "prompt_tokens": 0, "secs": round(time.time() - start, 1)}


def prompt_tokens(messages: list, base_url: str = URL, urlopen=None) -> int:
    """Tokens of the prompt by the server's tokenizer (POST /tokenize on each message, plus the chat template's overhead): what must leave room for max_tokens in a slot."""
    urlopen = urlopen or urllib.request.urlopen
    total = LENGTH_OVERHEAD
    for m in messages:
        req = urllib.request.Request(base_url + "/tokenize", json.dumps({"content": m["content"], "add_special": False}).encode(), {"Content-Type": "application/json"})
        total += len(json.load(urlopen(req, timeout=120))["tokens"])
    return total


class Session:
    """The shared counters of a run: tokens and seconds of the finished requests (for the deadline rule and run.json), under one lock."""

    def __init__(self):
        self.lock = threading.Lock()
        self.requests = self.completion_tokens = self.cut_off = self.failed = 0
        self.secs = 0.0
        self.tasks_done = 0
        self.stopped_by_deadline = False

    def add(self, out: dict) -> None:
        with self.lock:
            self.requests += 1
            self.completion_tokens += out["tokens"]
            self.secs += out["secs"]
            self.cut_off += out["finish"] == "length"
            self.failed += out["finish"] == "error"

    def mean_secs(self) -> float:
        with self.lock:
            return self.secs / self.requests if self.requests else 0.0


def ask_task(cfg: dict, task: dict, have: list, deadline: float, session: Session, base_url: str = URL, urlopen=None):
    """The line to checkpoint for a task: its earlier replies `have`, with the missing and failed samples asked again, one after the other (a task occupies one slot). The deadline is checked before every
    request; when it comes, the samples answered so far are kept and the rest left for the next session (the list simply ends: `samples_to_ask`). None when nothing was asked and nothing is known."""
    outs = list(have) + [None] * (cfg["k"] - len(have))
    for i in samples_to_ask(have, cfg["k"]):
        if not can_start(time.time(), deadline, session.mean_secs()):
            session.stopped_by_deadline = True
            break
        outs[i] = ask_one(cfg, task["messages"], i, base_url, urlopen)
        session.add(outs[i])
    while outs and outs[-1] is None:
        outs.pop()
    if not outs:
        return None
    return {"id": task["id"], "kind": task["kind"], "prompt_sha": prompt_sha(task["messages"]), "outs": outs}


# ---- the checkpoint and the resume -----------------------------------------------------------------------------------------------------------

def read_lines(path) -> list:
    """The lines of a replies.jsonl that are JSON objects with an id and a list of outs (a line cut off by a timeout is skipped)."""
    out = []
    for line in pathlib.Path(path).read_text().splitlines():
        try:
            rec = json.loads(line)
            if isinstance(rec["id"], str) and isinstance(rec["outs"], list):
                out.append(rec)
        except (json.JSONDecodeError, KeyError, TypeError):
            continue
    return out


def load_previous(paths) -> dict:
    """{task id: record} of the replies files of earlier sessions (the last line of an id wins, files in the order given)."""
    have = {}
    for p in paths:
        for rec in read_lines(p):
            have[rec["id"]] = rec
    return have


def find_previous(root: pathlib.Path = INPUT) -> list:
    """Every replies.jsonl under `root` (an earlier session's output attached to the kernel, or uploaded as a dataset), sorted by path."""
    return sorted(pathlib.Path(root).rglob("replies.jsonl")) if pathlib.Path(root).exists() else []


def run_session(cfg: dict, tasks: list, previous: dict, checkpoint: pathlib.Path, deadline: float, base_url: str = URL, urlopen=None, log_fn=print) -> dict:
    """Ask what is left. `previous` = {id: record} from earlier sessions: they are copied to `checkpoint` first (so a session that asks nothing loses nothing), a task with every sample answered is skipped,
    a task with missing or failed samples is asked for those only. A task whose prompt and `max_tokens` do not fit a slot is not asked (`skipped_too_long`). Each finished task is appended and flushed.
    Returns the counters for run.json."""
    session = Session()
    checkpoint = pathlib.Path(checkpoint)
    checkpoint.write_text("".join(json.dumps(r) + "\n" for r in previous.values()))
    todo, too_long = [], []
    for t in tasks:
        have = previous.get(t["id"], {}).get("outs", [])
        if not samples_to_ask(have, cfg["k"]):
            continue
        if prompt_tokens(t["messages"], base_url, urlopen) + cfg["max_tokens"] > cfg["ctx_per_slot"]:
            too_long.append(t["id"])
            continue
        todo.append((t, have))
    log_fn(f"{len(tasks)} tasks, {sum(1 for t in tasks if not samples_to_ask(previous.get(t['id'], {}).get('outs', []), cfg['k']))} complete already, {len(todo)} to ask, "
           f"{len(too_long)} skipped (prompt + max_tokens {cfg['max_tokens']} over {cfg['ctx_per_slot']} tokens)")
    start = time.time()
    with ThreadPoolExecutor(cfg["slots"]) as ex, open(checkpoint, "a") as ck:
        for rec in ex.map(lambda item: ask_task(cfg, item[0], item[1], deadline, session, base_url, urlopen), todo):
            if rec is None:
                continue
            ck.write(json.dumps(rec) + "\n")
            ck.flush()
            session.tasks_done += 1
            if session.tasks_done % 10 == 0:
                el = time.time() - start
                log_fn(f"{session.tasks_done}/{len(todo)} tasks, {session.completion_tokens} tokens out, {session.completion_tokens / max(el, 1):.1f} tok/s aggregate, "
                       f"{session.cut_off} cut off, {session.failed} failed")
    el = time.time() - start
    return {"tasks": len(tasks), "asked": len(todo), "tasks_written": session.tasks_done, "skipped_too_long": too_long, "requests": session.requests, "completion_tokens": session.completion_tokens,
            "cut_off": session.cut_off, "failed": session.failed, "seconds": round(el), "tok_per_s": round(session.completion_tokens / max(el, 1e-9), 1),
            "deadline_hit": session.stopped_by_deadline}


# ---- llama.cpp for the T4 -----------------------------------------------------------------------------------------------------------------

LIBCUDA_PATTERNS = ("/usr/local/cuda/lib64/stubs/libcuda.so", "/usr/local/cuda/targets/x86_64-linux/lib/stubs/libcuda.so", "/usr/lib/x86_64-linux-gnu/libcuda.so*",
                    "/usr/local/nvidia/lib64/libcuda.so*", "/usr/lib64/libcuda.so*")


def find_libcuda(patterns=LIBCUDA_PATTERNS):
    """The first libcuda.so the patterns match, the toolkit's stub before the driver's copy. cmake's FindCUDA does not look where Kaggle keeps the driver library
    (/usr/local/nvidia/lib64), so the path is given to it as -DCUDA_cuda_driver_LIBRARY; without that the configure step fails on the missing CUDA::cuda_driver."""
    return next(iter([p for pat in patterns for p in sorted(glob.glob(pat))]), None)


def find_prebuilt(root: pathlib.Path = INPUT):
    """A llama-server saved by an earlier run (its `llama-bin` folder attached as a dataset), or None: it saves the 30-minute build. It must sit beside its shared libraries."""
    root = pathlib.Path(root)
    return next(iter(sorted(root.rglob("llama-server"))), None) if root.exists() else None


def prepare_prebuilt(binary: pathlib.Path, dest: pathlib.Path = pathlib.Path("/tmp/llama-prebuilt")) -> pathlib.Path:
    """Copy the folder of an attached llama-server (its shared libraries beside it) to `dest` and make the server executable: an attached dataset or kernel output is a read-only mount that does not
    have to keep the executable bit. Returns the path of the copy."""
    shutil.rmtree(dest, ignore_errors=True)
    shutil.copytree(binary.parent, dest)
    new = dest / binary.name
    new.chmod(0o755)
    return new


def binary_works(binary, env=None) -> bool:
    """True when `<binary> --version` runs and exits 0 (the libraries it needs are found): a prebuilt server that does not start here is not used, the kernel builds one."""
    try:
        return subprocess.run([str(binary), "--version"], capture_output=True, timeout=120, env=env).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def build_llama(cfg: dict) -> pathlib.Path:
    """Build llama-server with CUDA for sm_75 (T4) into /tmp/llama.cpp/build/bin; the build is about 30 minutes. Returns the path of the binary; exits with the tail of the cmake and build logs on a failure."""
    src = pathlib.Path("/tmp/llama.cpp")
    binary = src / "build" / "bin" / "llama-server"
    if binary.exists():
        return binary
    sh(f"git clone --depth 1 https://github.com/ggml-org/llama.cpp {src}")
    if cfg.get("llama_cpp_ref"):
        sh(f"cd {src} && git fetch --depth 1 origin {cfg['llama_cpp_ref']} && git checkout FETCH_HEAD")
    if not shutil.which("cmake"):
        sh([sys.executable, "-m", "pip", "install", "-q", "cmake"])
    nvcc = shutil.which("nvcc") or next(iter(sorted(glob.glob("/usr/local/cuda*/bin/nvcc"))), None)
    log("cmake", shutil.which("cmake"), "nvcc", nvcc)
    if nvcc:
        os.environ["PATH"] = os.path.dirname(nvcc) + ":" + os.environ["PATH"]
        os.environ["CUDACXX"] = nvcc
    libcuda = find_libcuda()
    log("libcuda", libcuda)
    driver = f"-DCUDA_cuda_driver_LIBRARY={libcuda} " if libcuda else ""
    r = subprocess.run(f"cd {src} && cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75 -DLLAMA_CURL=OFF {driver}-DLLAMA_BUILD_TESTS=OFF -DCMAKE_BUILD_TYPE=Release > /tmp/cmake.log 2>&1 && "
                       f"cmake --build build --target llama-server -j 4 > /tmp/build.log 2>&1", shell=True)
    if r.returncode:
        for f in ("/tmp/cmake.log", "/tmp/build.log"):
            if os.path.exists(f):
                print(f"==== {f}\n" + open(f).read()[-4000:])
        raise SystemExit("llama.cpp build failed")
    return binary


def server_command(binary, model_path: str, cfg: dict) -> list:
    """llama-server over both T4 (layer split), `slots` x `ctx_per_slot` tokens of context, the chat template of the model (`--jinja`: it decides `enable_thinking`) and the reasoning returned
    apart as `reasoning_content`."""
    return [str(binary), "-m", model_path, "--host", "127.0.0.1", "--port", "8080", "-ngl", "999", "--split-mode", "layer", "-c", str(cfg["slots"] * cfg["ctx_per_slot"]),
            "--parallel", str(cfg["slots"]), "--jinja", "--reasoning-format", cfg["reasoning_format"], "-fa", "on"]


def wait_for_server(proc, base_url: str = URL, tries: int = 600) -> None:
    for _ in range(tries):
        try:
            if urllib.request.urlopen(base_url + "/health", timeout=5).status == 200:
                return
        except Exception:
            pass
        if proc.poll() is not None:
            print(open("/tmp/server.log").read()[-4000:])
            raise SystemExit("llama-server died")
        time.sleep(2)
    raise SystemExit("llama-server did not come up in 20 minutes")


# ---- the kernel ----------------------------------------------------------------------------------------------------------------------------

def main() -> None:
    data_dir = next(INPUT.rglob("tasks.jsonl")).parent
    user = json.loads((data_dir / "config.json").read_text()) if (data_dir / "config.json").exists() else {}
    cfg = resolve_config(user)
    log("config", cfg)
    subprocess.run("nvidia-smi --query-gpu=name,memory.total --format=csv || ls /dev/nvidia*", shell=True)
    binary, prebuilt = None, find_prebuilt()
    if prebuilt:
        candidate = prepare_prebuilt(prebuilt)
        os.environ["LD_LIBRARY_PATH"] = str(candidate.parent) + ":" + os.environ.get("LD_LIBRARY_PATH", "")
        if binary_works(candidate):
            binary = candidate
            log("llama-server from an attached output:", prebuilt)
        else:
            log("the attached llama-server does not start here; building llama.cpp")
    if binary is None:
        binary = build_llama(cfg)
    sh(f"mkdir -p {OUT}/llama-bin && cp -n {binary.parent}/llama-server {binary.parent}/*.so* {OUT}/llama-bin/ 2>/dev/null || true")     # every session's output carries it: a later one attaches it and skips the build
    commit = f"prebuilt {prebuilt}" if binary.parent == pathlib.Path("/tmp/llama-prebuilt") else subprocess.run("cd /tmp/llama.cpp && git log -1 --format='%h %cd' 2>/dev/null", shell=True, capture_output=True,
                                                                                                          text=True).stdout.strip()
    log("llama.cpp", commit)

    sh([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub>=0.30", "hf_transfer"])
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(cfg["model_repo"], cfg["model_file"], local_dir="/tmp/model")
    log("model", path, os.path.getsize(path) // 2**20, "MiB")

    srv_log = open("/tmp/server.log", "w")
    srv = subprocess.Popen(server_command(binary, path, cfg), stdout=srv_log, stderr=subprocess.STDOUT)
    try:
        wait_for_server(srv)
        log("server up")
        tasks = [json.loads(line) for line in open(data_dir / "tasks.jsonl") if line.strip()]
        previous = load_previous([*find_previous(), *([OUT / 'replies.jsonl'] if (OUT / 'replies.jsonl').exists() else [])])      # earlier sessions, then this one's own file if the kernel restarted
        log(len(previous), "tasks answered (fully or in part) by earlier sessions")
        summary = run_session(cfg, tasks, previous, OUT / "replies.jsonl", T0 + cfg["deadline_h"] * 3600, log_fn=log)
        log("finished", summary)
        gpus = subprocess.run("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader", shell=True, capture_output=True, text=True).stdout.strip().splitlines()
        (OUT / "run.json").write_text(json.dumps({**cfg, **summary, "llama_cpp": commit, "gpus": gpus, "started": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(T0)),
                                                  "session_seconds": round(time.time() - T0)}, indent=1))
    finally:
        srv.kill()


if __name__ == "__main__":
    main()
