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
for EVERY family including legacy), which proves the task is answerable from the pseudocode. What is STORED (default mode) is never the model's text: the assistant
message is rendered from the task's own truth (rea_bench.targets: the canonical headline and one factual sentence from a per-variant set of 3-6
phrasings, chosen by a seeded RNG from the task id); the record keeps `target_source: "truth-template"` and nothing of the model (round 9, m1: not even its headline;
the checkpoint on the volume holds the raw replies for `--resume` and is never downloaded or uploaded). Free prose repeatedly carried a wrong fact past every filter, so none of it is stored. A failed request
(`[error ...]`) is never scored or stored and is counted per family; more than 5% of failed requests prints a warning.
A task whose chance level is 0.05 or more (the best constant guess within its variant: CRC, pktsum, fsm, TEA/XTEA,
ChaCha/Salsa, several structs, legacy skip) is kept only when at least 2 of the K samples are right; every reply is scored again at the end,
so a `--resume` never keeps the verdict of an older scorer. The stage checkpoints every task on the volume (`--resume` continues a run that timed
out) and reports yield per family in `.stats.json`. The held-out seed 1001 is
never sampled (the stage refuses it). Output: SFT pairs in results/phase6/rea-data/. A pair whose system + user + assistant text is longer than the training window (`modal_lora.MAX_LEN`, 2048 tokens
with the chat template; the training run drops it, never cuts it) is dropped here and counted per family (round 9, m2: `over_length` in `.stats.json`, a line printed after the run).
    modal run tools/modal/modal_rea_data.py --stage sample --model <name> [--seed 7 --family legacy] [--k 6] [--resume] [--force]
Teacher-reasoning mode (round 11, `--reasoning`, rea_bench/reasoning.py; README "Next run"): the opposite of the above for the stored text, with the industry-standard rule (rejection sampling,
STaR, RFT, R1 distills): the teacher writes short steps and a LAST line `Answer: <headline>`; the reply is accepted when the text after the last `Answer:` line passes `score_strict` (plus the agreement
rule, counted on the answers), the steps are kept as they are and NOT checked (some may be wrong). A kept task gives the records with the same messages: `target_source: "truth-template"` (one, as above) and
`"teacher-reasoning"` (round 12: up to `--max-traces` = 3 DISTINCT accepted replies, in sample order, each verbatim, <think> blocks stripped, whitespace trimmed, after the R1-style quality filters: mixed
language, repetition loop, unreadable, self-contradiction; two replies are duplicates when their normalized text is identical or their token-set Jaccard is >= 0.9; every drop is counted per family in
`.stats.json` under `reasoning`). Own checkpoint and output (`-reasoning` in the tag), thinking off, `--max-tokens` (768), `--max-len` (required: the training window; the run reports how many records fit
2048, 3072 and 4096), `--quota kind=N,...` (sizing), and a projected GPU-hours/cost line before the GPU is rented.
    modal run tools/modal/modal_rea_data.py --stage sample --model codeforge-27b-heretic-q4 --reasoning --max-len 3072 --k 4 --quota crc=220,structs=220
`--thinking` (needs `--reasoning`; README "Thinking dataset (next run)"): the model's native reasoning. The request sets `enable_thinking: true` and Qwen3's recommended thinking sampling (temperature 0.6, top_p 0.95,
top_k 20, min_p 0, never greedy; `--temperature --top-p --top-k --min-p` override), `--max-tokens` defaults to 4096, the server slot holds the prompt and that many tokens (a prompt that leaves less is not asked and is
counted), the server runs with `--jinja --reasoning-format deepseek`, the server's reasoning (`reasoning_content`, or an inline <think> block) is kept and the record text is `<think>reasoning</think>` + the visible reply;
only the visible text's last `Answer:` line is scored. The records carry `target_source: "teacher-reasoning"` too; the tag, the files and the stats say `thinking` (`-reasoning-think`), so a thinking run never mixes with
a visible-reasoning one; the stats record the sampling (`sampling`, `slot`), the length report counts what fits 4096 and 8192. `modal_lora.py` trains these records (and, with `--mix`, the truth-template ones).
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

import collections
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
from modal_lora import MAX_LEN  # noqa: E402  (the training window: one number for the length rule of the sample stage and of the training run)
from rea_bench import dedupe, families, outputs, reasoning as rsn, sft, survive  # noqa: E402  (stdlib-only)
from rea_bench.scoring import check_mode, check_target, score_strict  # noqa: E402  (stdlib-only, unit-tested without Modal; NOT modal_rea.py's evaluation scorer)

image = image.add_local_python_source("modal_rea", "modal_lora")     # modal_rea's image already mounts rea_bench at /root/rea_bench; modal_lora: the training window (MAX_LEN)
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


def _messages(t: dict, reasoning: bool = False) -> list:
    """The system and user message the sampler answers (and the SFT record is made of). `reasoning`: the system message asks for the steps and a last `Answer:` line
    (rea_bench.reasoning.SYSTEM_REASONING); SYSTEM_RECIPE is untouched, and the user message is the same."""
    return [{"role": "system", "content": rsn.SYSTEM_REASONING if reasoning else SYSTEM_RECIPE}, {"role": "user", "content": rsn.user_message(t)}]


def _server_token_count(text: str) -> int:
    """Tokens of `text` in the tokenizer of the llama-server that answered the samples (POST /tokenize, no special tokens)."""
    import urllib.request as ur
    req = ur.Request("http://127.0.0.1:8080/tokenize", data=json.dumps({"content": text, "add_special": False}).encode(), headers={"Content-Type": "application/json"})
    with ur.urlopen(req, timeout=120) as r:
        return len(json.load(r)["tokens"])


def _apply_length_rule(kept: list, stats: dict, max_len: int = MAX_LEN, report_windows: tuple = ()) -> tuple:
    """Round 9, m2: the training run (modal_lora.encode_ids) drops a pair whose prompt + answer + EOS is longer than the window (MAX_LEN tokens, or the `--max-len` of a run that trains on a
    longer one). Measured here with the tokenizer of the
    sampling server (a Qwen3.x GGUF: the vocabulary of the training base, not the training tokenizer byte for byte; sft.LENGTH_OVERHEAD covers the chat template and the report
    counts the records near the limit), the over-length records are dropped before the file is written and counted per family. A measurement that fails keeps every record, says so
    in the stats, and leaves the authoritative drop to the training run. `report_windows`: also count the records that fit each of those windows (a reasoning run: 2048, 3072, 4096)."""
    try:
        return sft.drop_over_length(kept, stats, _server_token_count, max_len, report_windows=report_windows)
    except Exception as e:
        return kept, {**stats, "over_length": {"max_len": max_len, "error": f"{type(e).__name__}: {e}"[:200]}}


def _window(reasoning: bool, max_len: int) -> int:
    """The training window the length rule uses: `--max-len`, or MAX_LEN (2048) for a default run. A reasoning run must name it: its records are several times longer, and the number of
    them that survive depends on it (the report says how many fit 2048, 3072 and 4096)."""
    if max_len < 0:
        raise SystemExit(f"--max-len {max_len} is not a window")
    if reasoning and not max_len:
        raise SystemExit(f"a --reasoning run needs --max-len: the training window in tokens that the length rule drops against ({', '.join(map(str, rsn.REPORT_WINDOWS))}); "
                         f"the run reports how many records fit each, and modal_lora.py needs the same --max-len to train on what is kept")
    return max_len or MAX_LEN


def _check_thinking(reasoning: bool, thinking: bool, max_tokens: int, max_traces: int) -> None:
    """A thinking run is a reasoning run whose token budget defaults to `reasoning.DEFAULT_THINKING_MAX_TOKENS` (4096): the reasoning is part of the reply, so the 768 of visible reasoning would cut
    nearly every reply off. `--max-tokens` is not negative, `--max-traces` is at least 1."""
    if thinking and not reasoning:
        raise SystemExit("--thinking is a variant of --reasoning (the reply is scored by its last `Answer:` line): pass --reasoning too")
    if max_tokens < 0:
        raise SystemExit(f"--max-tokens {max_tokens}: a number of tokens, or leave it out for the default ({rsn.DEFAULT_THINKING_MAX_TOKENS} with --thinking, {rsn.DEFAULT_MAX_TOKENS} with --reasoning, 300)")
    if max_traces < 1:
        raise SystemExit(f"--max-traces {max_traces}: at least 1 teacher record per kept task")


SAMPLER_TIMEOUT_H = 5       # the `timeout` of sample_answers below, in hours (the cost line says how many calls a run needs)
SAMPLER_SLOTS = 4           # requests in flight: `--parallel` of the server and the threads that ask
THINKING_SERVER_ARGS = ["--jinja", "--reasoning-format", "deepseek"]     # a thinking run: the chat template decides `enable_thinking`, and the server returns the reasoning as `reasoning_content`
SEED_BASE = 1234            # the seed of sample i is SEED_BASE + i


@app.function(gpu=["L40S", "A100-40GB"], volumes={"/models": models_vol, "/data": data_vol}, timeout=SAMPLER_TIMEOUT_H * 3600)
def sample_answers(name: str, repo: str, fname: str, k: int, tag: str = "seed7-legacy", resume: bool = False, reasoning: bool = False, max_tokens: int = 0,
                   max_len: int = 0, quota: str = "", thinking: bool = False, max_traces: int = rsn.DEFAULT_MAX_TRACES, sampling: dict = None) -> dict:
    """K replies per task, checkpointed task by task on the volume (`<sft stem>.partial.jsonl`): a run that hits the timeout is continued
    with resume=True instead of paid for twice; a resume asks again the samples that failed (`[error ...]`) or are missing, and only those. The SFT
    records are built at the end from all replies (rea_bench.sft: the teacher's headline is a filter, the stored text is rendered from the truth, a
    task of chance >= 0.05 needs two right samples); the stats come per family.
    `reasoning` (round 11, rea_bench.reasoning): the teacher is asked for the steps and a last `Answer:` line, `max_tokens` defaults to 768 (300 otherwise), the final answer is the filter and a kept
    task gives the truth-rendered record AND up to `max_traces` distinct clean accepted replies verbatim (round 12: quality filters, duplicate rule, cap: rea_bench.reasoning.pick_traces); the checkpoint and the
    output have their own tag (`-reasoning`), the thinking stays off. `thinking` (needs `reasoning`): `enable_thinking: true`, `max_tokens` defaults to 4096, the server's reasoning is kept in
    the reply and in the record as `<think>...</think>` + the visible reply, only the visible text's last `Answer:` line is scored; own tag (`-reasoning-think`). `sampling`: the request's sampling values
    (`reasoning.sampling_for`: `{"temperature": 0.7}` as always, Qwen3's thinking settings with `thinking`); a reasoning run records it, the slot and the prompts that did not fit in its stats. `max_len`: the
    training window of the length rule (required with `reasoning`, MAX_LEN otherwise). `quota`: `kind=N,...`, at most N tasks of each listed kind and no task of another kind."""
    from huggingface_hub import hf_hub_download
    import urllib.request
    _check_thinking(reasoning, thinking, max_tokens, max_traces)
    window = _window(reasoning, max_len)
    max_tokens = max_tokens or rsn.default_max_tokens(reasoning, thinking)
    sampling = rsn.sampling_for(thinking, **(sampling or {}))                    # validated again where it is used: a thinking run is never greedy, whoever called
    slot_ctx = rsn.slot_context(max_tokens) if reasoning else rsn.SLOT_CONTEXT      # tokens per slot of the server: the prompt and max_tokens fit (a default run: the 8192 it always had)
    run_tag = outputs.sample_tag(tag, reasoning, thinking)        # the checkpoint and the output of a reasoning run are their own (`-reasoning`, `-reasoning-think`); the decompiled tasks are read by `tag`
    score = rsn.score_reasoning if reasoning else score_strict
    data_vol.reload()
    src_dir = pathlib.Path(f"/data/{outputs.decompiled_stem(tag)}")
    if not src_dir.is_dir() and tag.startswith("seed") and tag.endswith("-legacy"):
        src_dir = pathlib.Path(f"/data/decompiled-{tag[:-len('-legacy')]}")     # made before the dirs were keyed by family
    tasks = [json.loads(p.read_text()) for p in sorted(src_dir.glob("*.json"))]
    tasks = [t for t in tasks if "pseudocode" in t]
    if not tasks:
        raise SystemExit(f"no decompiled tasks in {src_dir} on the volume (run --stage decompile with the same --seed and --family first)")
    if quota:
        want = rsn.parse_quota(quota)
        tasks = rsn.apply_quota(tasks, want)
        have = collections.Counter(t["kind"] for t in tasks)
        print(f"--quota {quota}: {len(tasks)} tasks to sample: " + ", ".join(f"{kind} {have[kind]}/{n}" for kind, n in want.items()))
        if not tasks:
            raise SystemExit(f"--quota {quota} selects no decompiled task of {src_dir}")
    for t in tasks:
        check_mode(t)       # an unknown score mode must stop the run before the model download and the server start
        check_target(t)     # so must a task without a truth template, or one whose rendered target its own scorer rejects
    partial = pathlib.Path(f"/data/{outputs.sft_stem(name, run_tag)}.partial.jsonl")
    results = outputs.load_partial(partial, resume)
    todo = [(t, outputs.samples_to_ask(results.get(t["id"]), k)) for t in tasks]
    todo = [(t, ask) for t, ask in todo if ask]
    if results:
        redo = sum(len(ask) for t, ask in todo if t["id"] in results)
        print(f"resuming: {len(tasks) - len(todo)} of {len(tasks)} tasks complete, {len(todo)} to ask ({redo} failed or missing samples of checkpointed tasks)")
    path = hf_hub_download(repo, fname, cache_dir="/models")
    server_log = open("/tmp/server.log", "w")
    srv = subprocess.Popen(["/app/llama-server", "-m", path, "--host", "127.0.0.1", "--port", "8080", "-ngl", "999", "-c", str(SAMPLER_SLOTS * slot_ctx),
                            "--parallel", str(SAMPLER_SLOTS), "-fa", "off", *(THINKING_SERVER_ARGS if thinking else [])], stdout=subprocess.DEVNULL, stderr=server_log)
    for _ in range(600):
        try:
            if urllib.request.urlopen("http://127.0.0.1:8080/health", timeout=2).status == 200:
                break
        except Exception:
            time.sleep(1)
    too_long = collections.Counter()
    if thinking:            # a prompt that leaves less than max_tokens in its slot gets a reply the server cuts off early: it is not asked, and it is counted (kind -> tasks)
        fits = []
        for t, ask in todo:
            n = sum(_server_token_count(m["content"]) for m in _messages(t, reasoning)) + sft.LENGTH_OVERHEAD
            if rsn.prompt_fits(n, max_tokens, slot_ctx):
                fits.append((t, ask))
            else:
                too_long[t["kind"]] += 1
        if too_long:
            print(f"{sum(too_long.values())} task(s) not asked: prompt + max_tokens {max_tokens} do not fit a {slot_ctx}-token slot ({', '.join(f'{k} {c}' for k, c in sorted(too_long.items()))})")
        todo = fits
    from concurrent.futures import ThreadPoolExecutor
    import threading
    perf, perf_lock = {"requests": 0, "completion_tokens": 0}, threading.Lock()

    def work(item):
        t, ask = item
        msgs = _messages(t, reasoning)
        outs = [tuple(o) for o in results[t["id"]]["outs"]] if t["id"] in results else []
        outs += [("", False)] * (k - len(outs))                    # a checkpointed task keeps its answers; only the samples in `ask` are asked
        for i in ask:
            try:
                body_seed = SEED_BASE + i
                import urllib.request as ur
                body = {"model": "local", "messages": msgs, **sampling, "seed": body_seed, "max_tokens": max_tokens,
                        "chat_template_kwargs": {"enable_thinking": thinking}, "cache_prompt": False}
                req = ur.Request("http://127.0.0.1:8080/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
                with ur.urlopen(req, timeout=600) as r:
                    reply = json.load(r)
                msg = reply["choices"][0]["message"]
                if reasoning:       # the reply as the checkpoint stores it (rea_bench.reasoning.reply_text: think blocks stripped, or kept as the trace in a thinking run; a cut-off reply ends in the TRUNCATED marker)
                    txt = rsn.reply_text(msg.get("content"), reply["choices"][0].get("finish_reason"), msg.get("reasoning_content"), thinking)
                else:
                    txt = msg.get("content") or ""
                    txt = re.sub(r"<think>.*?</think>", "", txt, flags=re.S).strip()
                with perf_lock:
                    perf["requests"] += 1
                    perf["completion_tokens"] += (reply.get("usage") or {}).get("completion_tokens", 0)
                ok = score(txt, t)
            except Exception as e:
                txt, ok = f"[error {e!r}]", False       # the text of a failed request is never scored (B1) and never stored; stats count it; --resume asks it again
            outs[i] = (txt, ok)
        return t, outs

    t_start = time.time()
    try:
        with ThreadPoolExecutor(SAMPLER_SLOTS) as ex, open(partial, "a") as ck:
            for i, (t, outs) in enumerate(ex.map(work, todo), 1):
                rec = {"id": t["id"], "kind": t["kind"], "outs": outs}
                results[t["id"]] = rec
                ck.write(json.dumps(rec) + "\n")
                ck.flush()
                if i % 20 == 0:
                    data_vol.commit()
        wall = max(time.time() - t_start, 1e-9)
        kept, stats = sft.assemble(tasks, results, name, lambda t: _messages(t, reasoning), reasoning, thinking, max_traces)
        if reasoning:           # this call's own throughput, next to the cost line's assumption (a resumed run reports only the replies it asked)
            stats["perf"] = {**perf, "wall_s": round(wall), "tok_per_s": round(perf["completion_tokens"] / wall, 1), "s_per_request": round(wall / max(perf["requests"], 1), 2)}
            stats["sampling"] = rsn.sampling_record(thinking, sampling, max_tokens, seed_base=SEED_BASE, slot_context=slot_ctx, parallel=SAMPLER_SLOTS)
            if thinking:
                stats["slot"] = {"slot_context": slot_ctx, "max_prompt_tokens": slot_ctx - max_tokens, "prompt_too_long": sum(too_long.values()), "by_kind": dict(sorted(too_long.items()))}
        windows = (rsn.THINKING_REPORT_WINDOWS if thinking else rsn.REPORT_WINDOWS) if reasoning else ()
        kept, stats = _apply_length_rule(kept, stats, window, windows)       # round 9, m2: what the training run would drop is dropped (and counted) here, while the server is still up
    finally:
        srv.kill()
        server_log.close()
        data_vol.commit()
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
         exclude_seed: int = -1, force: bool = False, local_gate: bool = True, resume: bool = False, reasoning: bool = False, max_tokens: int = 0,
         max_len: int = 0, quota: str = "", budget_usd: float = 30.0, gpu_usd_per_h: float = rsn.L40S_USD_PER_H, thinking: bool = False,
         max_traces: int = rsn.DEFAULT_MAX_TRACES, temperature: float = -1.0, top_p: float = -1.0, top_k: int = -1, min_p: float = -1.0):
    """exclude_seed: tasks of that seed never enter this run (-1: the held-out seed 1001 for any other seed, 0: none).
    local_gate: draw tasks again when the asked constant is not in the code compiled HERE (needs gcc, objdump and, for clang tasks,
    clang on PATH; without them the draws it cannot check are counted and warned about, and only the Modal-side re-check applies).
    resume: continue a sample stage that timed out from its checkpoint on the volume.
    reasoning (sample stage): the teacher-reasoning mode (rea_bench/reasoning.py): the teacher writes short steps and a last `Answer:` line, its final answer is checked against the truth and
    the reply is stored as it is next to the truth-rendered record; own checkpoint and output (`-reasoning` in the tag); needs --max-len. max_tokens: per reply (0: 300, 768 with --reasoning, 4096 with --thinking).
    max_traces (--reasoning): up to this many distinct accepted replies per kept task, after the trace quality filters (default 3; the truth-rendered record stays one). thinking (needs --reasoning): the
    model reasons (`enable_thinking: true`) with Qwen3's thinking sampling, max_tokens defaults to 4096, the record is `<think>reasoning</think>` + the visible reply, only the visible text's last `Answer:` line is
    scored; files, tag and stats say `thinking` (`-reasoning-think`). temperature, top_p, top_k, min_p: override one sampling value (negative: the default of the mode: 0.7 alone, or 0.6 / 0.95 / 20 / 0 with
    --thinking; a thinking run refuses greedy decoding).
    max_len: the training window in tokens the length rule drops against (0: 2048; required with --reasoning; the report says how many teacher records fit 2048, 3072 and 4096).
    quota: `kind=N,...`, sample at most N tasks of each listed kind and none of another (sizing a run). budget_usd, gpu_usd_per_h: the cost line printed before a reasoning run.
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
        if reasoning or quota or max_tokens or max_len or thinking or max_traces != rsn.DEFAULT_MAX_TRACES:
            print("note: --reasoning, --thinking, --max-traces, --max-tokens, --max-len and --quota belong to --stage sample; the decompile stage ignores them (--family and --n are what limit it)")
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
        _check_thinking(reasoning, thinking, max_tokens, max_traces)      # --thinking without --reasoning stops here, before anything is rented
        window = _window(reasoning, max_len)                 # a reasoning run without --max-len stops here, before anything else is checked or rented
        given = lambda v: None if v < 0 else v               # a negative value is "not given": the default of the mode
        sampling = rsn.sampling_for(thinking, temperature=given(temperature), top_p=given(top_p), top_k=given(top_k), min_p=given(min_p))      # a thinking run that is greedy stops here too
        want = rsn.parse_quota(quota, rsn.known_kinds()) if quota else {}
        rtag = outputs.sample_tag(tag, reasoning, thinking)  # a reasoning run's checkpoint and output are its own: `-reasoning` (`-reasoning-think` with --thinking) in the tag
        out, stats = (dest / f"{outputs.sft_stem(name, rtag)}.{ext}" for ext in ("jsonl", "stats.json"))
        outputs.refuse_overwrite([out, stats], force)       # before a GPU is rented
        if reasoning:
            tokens = max_tokens or rsn.default_max_tokens(reasoning, thinking)
            count, source = rsn.local_task_count(dest / f"{outputs.decompiled_stem(tag)}.jsonl", want, n)
            plan = rsn.project(count, k, tokens, gpu_usd_per_h, SAMPLER_TIMEOUT_H)
            print(rsn.thinking_cost_line(plan, source) if thinking else rsn.cost_line(plan, source))
            warning = rsn.thinking_budget_warning(plan, budget_usd) if thinking else rsn.budget_warning(plan, budget_usd)
            if warning:
                print("\n" + "!" * 100 + f"\n{warning}\n" + "!" * 100 + "\n")
            print(f"window of the length rule: --max-len {window}; max_tokens {tokens}; up to {max_traces} traces per kept task; checkpoint and output tag {rtag}")
            print(f"sampling: {sampling}; server slot {rsn.slot_context(tokens):,} tokens (prompt + max_tokens)")
            if thinking and window < rsn.THINKING_WINDOW:
                print(f"note: --max-len {window} is below the recommended {rsn.THINKING_WINDOW} for thinking records: a record longer than it is dropped (the length report counts what fits "
                      f"{' and '.join(map(str, rsn.THINKING_REPORT_WINDOWS))}); modal_lora.py needs the same --max-len")
        res = sample_answers.remote(name, repo, fname, k, tag, resume=resume, reasoning=reasoning, max_tokens=max_tokens, max_len=max_len, quota=quota,
                                    thinking=thinking, max_traces=max_traces, sampling=sampling)
        outputs.write_atomic(out, (json.dumps(r) for r in res["sft"]))
        stats.write_text(json.dumps(res["stats"], indent=1))
        s = res["stats"]
        print(name, rtag, {f: s.get(f) for f in ("tasks", "answers", "errors", "correct", "solved_tasks", "kept_tasks", "kept_records", "agreement")}, "->", out)
        warning = sft.error_warning(s)
        if warning:
            print("\n" + "!" * 100 + f"\n{warning}\n" + "!" * 100 + "\n")
        length_line = sft.over_length_line(s)
        if length_line:
            print(length_line)
        for line in (sft.fits_line(s), *sft.reasoning_report(s)):
            if line:
                print(line)
        for kind, row in s.get("by_kind", {}).items():
            print(f"  {kind:10s} {row['solved_tasks']:4d}/{row['tasks']:<4d} solved, {row['kept_tasks']:4d} kept ({row['agreement']} dropped for lack of "
                  f"agreement, {row.get('errors', 0)} failed requests)")
