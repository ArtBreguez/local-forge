"""The sample stage on Modal, for any subject: a teacher answers each task K times, the replies are verified, filtered and written as SFT records. No `import modal` here (stdlib only; Modal and the model
libraries are imported inside the function that needs them), so it is unit-tested with a scripted fake server.

Moved from `tools/modal/modal_rea_data.py` (`--stage sample`) with its behaviour. What a subject decides comes from its `Domain` (`distill/domain.py`): the tasks, the messages the teacher is asked, the verdict on an
answer, the truth record. What is here is the same for every subject: the llama-server, the checkpoint and `--resume`, the token budget and slot, the length rule, the cost line printed before a GPU is rented.

  sample_answers   the body of the stage. A subject's wrapper registers it with Modal (`@app.function(gpu=..., volumes={"/models": ..., "/data": data_vol})`) and calls it with its domain and the volume.
  project, cost_line, ...   the projected GPU hours and dollars of a run (`thinking_cost_line` for a thinking run, which has no expected figure), from the one measurement there is: REA's phase A on one L40S.

The teacher is open-weight (see `distill/teachers.py`); no output of an OpenAI or Anthropic model enters a dataset.
"""
from __future__ import annotations

import collections
import json
import math
import pathlib
import re
import subprocess
import time

from .. import config, outputs, records, traces
from ..config import MAX_LEN  # noqa: F401  (the training window: one number for the length rule of the sample stage and of the training run)
from ..domain import mode_of
from ..traces import DEFAULT_MAX_TRACES, is_error_reply


def _server_token_count(text: str) -> int:
    """Tokens of `text` in the tokenizer of the llama-server that answered the samples (POST /tokenize, no special tokens)."""
    import urllib.request as ur
    req = ur.Request("http://127.0.0.1:8080/tokenize", data=json.dumps({"content": text, "add_special": False}).encode(), headers={"Content-Type": "application/json"})
    with ur.urlopen(req, timeout=120) as r:
        return len(json.load(r)["tokens"])


def _apply_length_rule(kept: list, stats: dict, max_len: int = config.MAX_LEN, report_windows: tuple = ()) -> tuple:
    """Round 9, m2: the training run (modal_lora.encode_ids) drops a pair whose prompt + answer + EOS is longer than the window (MAX_LEN tokens, or the `--max-len` of a run that trains on a
    longer one). Measured here with the tokenizer of the
    sampling server (a Qwen3.x GGUF: the vocabulary of the training base, not the training tokenizer byte for byte; records.LENGTH_OVERHEAD covers the chat template and the report
    counts the records near the limit), the over-length records are dropped before the file is written and counted per family. A measurement that fails keeps every record, says so
    in the stats, and leaves the authoritative drop to the training run. `report_windows`: also count the records that fit each of those windows (a reasoning run: 2048, 3072, 4096)."""
    try:
        return records.drop_over_length(kept, stats, _server_token_count, max_len, report_windows=report_windows)
    except Exception as e:
        return kept, {**stats, "over_length": {"max_len": max_len, "error": f"{type(e).__name__}: {e}"[:200]}}


def _window(reasoning: bool, max_len: int) -> int:
    """The training window the length rule uses: `--max-len`, or MAX_LEN (2048) for a default run. A reasoning run must name it: its records are several times longer, and the number of
    them that survive depends on it (the report says how many fit 2048, 3072 and 4096)."""
    if max_len < 0:
        raise SystemExit(f"--max-len {max_len} is not a window")
    if reasoning and not max_len:
        raise SystemExit(f"a --reasoning run needs --max-len: the training window in tokens that the length rule drops against ({', '.join(map(str, config.REPORT_WINDOWS))}); "
                         f"the run reports how many records fit each, and the training run needs the same --max-len to train on what is kept")
    return max_len or config.MAX_LEN


def _check_thinking(reasoning: bool, thinking: bool, max_tokens: int, max_traces: int) -> None:
    """A thinking run is a reasoning run whose token budget defaults to `config.DEFAULT_THINKING_MAX_TOKENS` (4096): the reasoning is part of the reply, so the 768 of visible reasoning would cut
    nearly every reply off. `--max-tokens` is not negative, `--max-traces` is at least 1."""
    if thinking and not reasoning:
        raise SystemExit("--thinking is a variant of --reasoning (the reply is scored by its last `Answer:` line): pass --reasoning too")
    if max_tokens < 0:
        raise SystemExit(f"--max-tokens {max_tokens}: a number of tokens, or leave it out for the default ({config.DEFAULT_THINKING_MAX_TOKENS} with --thinking, {config.DEFAULT_MAX_TOKENS} with --reasoning, 300)")
    if max_traces < 1:
        raise SystemExit(f"--max-traces {max_traces}: at least 1 teacher record per kept task")


SAMPLER_TIMEOUT_H = 5       # the `timeout` of sample_answers below, in hours (the cost line says how many calls a run needs)
SAMPLER_SLOTS = 4           # requests in flight: `--parallel` of the server and the threads that ask
THINKING_SERVER_ARGS = ["--jinja", "--reasoning-format", "deepseek"]     # a thinking run: the chat template decides `enable_thinking`, and the server returns the reasoning as `reasoning_content`
SEED_BASE = 1234            # the seed of sample i is SEED_BASE + i




def sample_answers(domain, data_vol, tasks_source, name: str, repo: str, fname: str, k: int, tag: str = "seed7-legacy", resume: bool = False, reasoning: bool = False, max_tokens: int = 0,
                   max_len: int = 0, quota: str = "", thinking: bool = False, max_traces: int = DEFAULT_MAX_TRACES, sampling: dict = None, max_tokens_by_family: dict = None) -> dict:
    """The sample stage, for any subject: `domain` supplies the tasks (`domain.load_tasks(tasks_source)`: for REA the directory of the volume the decompile stage wrote), the messages, the verdict and the truth; `data_vol` is the Modal
    Volume mounted at /data (`reload` and `commit` are called on it); the model is read from /models. The wrapper of a subject (tools/modal/modal_rea_data.py for REA) registers this function with Modal and gives it its own.
    K replies per task, checkpointed task by task on the volume (`<sft stem>.partial.jsonl`): a run that hits the timeout is continued
    with resume=True instead of paid for twice; a resume asks again the samples that failed (`[error ...]`) or are missing, and only those; in a reasoning run also the STALE ones: a reply carries
    the fingerprint of the prompt it answered and the budget it was asked with (`outputs.out_meta`), and one that answered another prompt than the task's now, or was cut off at a smaller budget than
    the task's now (`max_tokens_by_family` grew), is asked again; a reply of an older checkpoint has no fingerprint (it is kept) and counts as asked with `max_tokens`. The SFT
    records are built at the end from all replies (`distill.records`: the teacher's answer is a filter, the stored text is rendered from the truth, a
    task that `domain.needs_agreement` needs two right samples); the stats come per family.
    `reasoning` (round 11, `distill.traces`): the teacher is asked for the steps and a last `Answer:` line, `max_tokens` defaults to 768 (300 otherwise), the final answer is the filter and a kept
    task gives the truth-rendered record AND up to `max_traces` distinct clean accepted replies verbatim (round 12: quality filters, duplicate rule, cap: `traces.pick_traces`); the checkpoint and the
    output have their own tag (`-reasoning`), the thinking stays off. `thinking` (needs `reasoning`): `enable_thinking: true`, `max_tokens` defaults to 4096, the server's reasoning is kept in
    the reply and in the record as `<think>...</think>` + the visible reply, only the visible text's last `Answer:` line is scored; own tag (`-reasoning-think`). `sampling`: the request's sampling values
    (`config.sampling_for`: `{"temperature": 0.7}` as always, Qwen3's thinking settings with `thinking`); a reasoning run records it, the slot and the prompts that did not fit in its stats. `max_len`: the
    training window of the length rule (required with `reasoning`, MAX_LEN otherwise). `quota`: `kind=N,...`, at most N tasks of each listed kind and no task of another kind.
    `max_tokens_by_family` (a reasoning run): {kind: tokens}, the reply budget of the tasks of that kind instead of `max_tokens` (`config.task_max_tokens`); each request asks its own task's budget, the prompt-fit check uses
    it, and the server's slot is sized for the longest budget among the tasks that are asked (`config.server_plan`: 4 slots of 8192 tokens, or 2 of 12288 when a task of the run has 8192; the same total context)."""
    from huggingface_hub import hf_hub_download
    import urllib.request
    _check_thinking(reasoning, thinking, max_tokens, max_traces)
    window = _window(reasoning, max_len)
    max_tokens = max_tokens or config.default_max_tokens(reasoning, thinking)
    sampling = config.sampling_for(thinking, **(sampling or {}))                    # validated again where it is used: a thinking run is never greedy, whoever called
    by_family = config.check_family_tokens(max_tokens_by_family or {}, domain.kinds())
    if by_family and not reasoning:
        raise SystemExit("max_tokens_by_family is for a reasoning or thinking run")
    mode = mode_of(reasoning, thinking)
    run_tag = outputs.sample_tag(tag, reasoning, thinking)        # the checkpoint and the output of a reasoning run are their own (`-reasoning`, `-reasoning-think`); the decompiled tasks are read by `tag`
    score = (lambda txt, t: traces.score_reasoning(txt, t, domain=domain)) if reasoning else (lambda txt, t: records.plain_ok(txt, t, domain))
    data_vol.reload()
    tasks = domain.load_tasks(tasks_source)
    if not tasks:
        raise SystemExit(f"no tasks in {tasks_source} on the volume (run the task stage of the subject first, with the same seed and family)")
    if quota:
        want = config.parse_quota(quota)
        tasks = config.apply_quota(tasks, want)
        have = collections.Counter(t["kind"] for t in tasks)
        print(f"--quota {quota}: {len(tasks)} tasks to sample: " + ", ".join(f"{kind} {have[kind]}/{n}" for kind, n in want.items()))
        if not tasks:
            raise SystemExit(f"--quota {quota} selects no task of {tasks_source}")
    longest = config.run_max_tokens([t["kind"] for t in tasks], max_tokens, by_family)       # the longest budget of a task that is in this run: what the slot holds
    slots, slot_ctx = config.server_plan(max_tokens, longest, SAMPLER_SLOTS) if reasoning else (SAMPLER_SLOTS, config.SLOT_CONTEXT)      # requests in flight, tokens per slot: the prompt and the budget fit (a run without a longer family: 4 x the 8192 it always had)
    budget = lambda t: config.task_max_tokens(t["kind"], max_tokens, by_family)
    for t in tasks:
        domain.check_task(t)    # a task the domain cannot score, or whose truth its own verify rejects, must stop the run before the model download and the server start
    partial = pathlib.Path(f"/data/{outputs.sft_stem(name, run_tag)}.partial.jsonl")
    results = outputs.load_partial(partial, resume)
    # a reasoning run's replies carry the prompt they answered and the budget they were asked with: a resume asks again what is stale (another prompt, or cut off at a smaller budget than the task's now); a default run's
    # checkpoint keeps its two-element replies (its record is rendered from the truth, the teacher's text is not stored)
    prompt_of = lambda t: outputs.prompt_sha(domain.messages(t, mode))
    stale_args = lambda t: (prompt_of(t), budget(t), max_tokens) if reasoning else ()
    todo = [(t, outputs.samples_to_ask(results.get(t["id"]), k, *stale_args(t))) for t in tasks]
    todo = [(t, ask) for t, ask in todo if ask]
    if results:
        redo = sum(len(ask) for t, ask in todo if t["id"] in results)
        stale = sum(1 for t, ask in todo if t["id"] in results for i in ask if i < len(results[t["id"]]["outs"]) and not is_error_reply(results[t["id"]]["outs"][i][0]))
        print(f"resuming: {len(tasks) - len(todo)} of {len(tasks)} tasks complete, {len(todo)} to ask (" + (f"{redo} failed or missing samples of checkpointed tasks)" if not stale else
              f"{redo - stale} failed or missing and {stale} stale samples of checkpointed tasks: a stale one answered another prompt than the task's now, or was cut off at a smaller budget than its own)"))
    path = hf_hub_download(repo, fname, cache_dir="/models")
    server_log = open("/tmp/server.log", "w")
    srv = subprocess.Popen(["/app/llama-server", "-m", path, "--host", "127.0.0.1", "--port", "8080", "-ngl", "999", "-c", str(slots * slot_ctx),
                            "--parallel", str(slots), "-fa", "off", *(THINKING_SERVER_ARGS if thinking else [])], stdout=subprocess.DEVNULL, stderr=server_log)
    for _ in range(600):
        try:
            if urllib.request.urlopen("http://127.0.0.1:8080/health", timeout=2).status == 200:
                break
        except Exception:
            time.sleep(1)
    too_long = collections.Counter()
    if thinking or by_family:            # a prompt that leaves less than its budget in the slot gets a reply the server cuts off early: it is not asked, and it is counted (kind -> tasks)
        fits = []
        for t, ask in todo:
            n = sum(_server_token_count(m["content"]) for m in domain.messages(t, mode)) + records.LENGTH_OVERHEAD
            if config.prompt_fits(n, budget(t), slot_ctx):
                fits.append((t, ask))
            else:
                too_long[t["kind"]] += 1
        if too_long:
            own = f" (and each family's own: {config.family_tokens_text(by_family)})" if by_family else ""
            print(f"{sum(too_long.values())} task(s) not asked: prompt + max_tokens {max_tokens}{own} do not fit a {slot_ctx}-token slot ({', '.join(f'{k} {c}' for k, c in sorted(too_long.items()))})")
        todo = fits
    from concurrent.futures import ThreadPoolExecutor
    import threading
    perf, perf_lock = {"requests": 0, "completion_tokens": 0}, threading.Lock()

    def work(item):
        t, ask = item
        msgs = domain.messages(t, mode)
        outs = [tuple(o) for o in results[t["id"]]["outs"]] if t["id"] in results else []
        outs += [("", False)] * (k - len(outs))                    # a checkpointed task keeps its answers; only the samples in `ask` are asked
        meta = {"prompt_sha": outputs.prompt_sha(msgs), "max_tokens": budget(t)}       # what a reasoning run's reply answers and was asked with (`outputs.out_meta`)
        for i in ask:
            try:
                body_seed = SEED_BASE + i
                import urllib.request as ur
                body = {"model": "local", "messages": msgs, **sampling, "seed": body_seed, "max_tokens": budget(t),
                        "chat_template_kwargs": {"enable_thinking": thinking}, "cache_prompt": False}
                req = ur.Request("http://127.0.0.1:8080/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
                with ur.urlopen(req, timeout=max(600, budget(t) * 600 // config.DEFAULT_THINKING_MAX_TOKENS)) as r:       # 600 s up to 4096 tokens, longer for a longer budget
                    reply = json.load(r)
                msg = reply["choices"][0]["message"]
                if reasoning:       # the reply as the checkpoint stores it (`traces.reply_text`: think blocks stripped, or kept as the trace in a thinking run; a cut-off reply ends in the TRUNCATED marker)
                    txt = traces.reply_text(msg.get("content"), reply["choices"][0].get("finish_reason"), msg.get("reasoning_content"), thinking)
                else:
                    txt = msg.get("content") or ""
                    txt = re.sub(r"<think>.*?</think>", "", txt, flags=re.S).strip()
                with perf_lock:
                    perf["requests"] += 1
                    perf["completion_tokens"] += (reply.get("usage") or {}).get("completion_tokens", 0)
                ok = score(txt, t)
            except Exception as e:
                txt, ok = f"[error {e!r}]", False       # the text of a failed request is never scored (B1) and never stored; stats count it; --resume asks it again
            outs[i] = (txt, ok, meta) if reasoning else (txt, ok)
        return t, outs

    by_id = {t["id"]: t for t in tasks}
    t_start = time.time()
    try:
        with ThreadPoolExecutor(slots) as ex, open(partial, "a") as ck:
            for i, (t, outs) in enumerate(ex.map(work, todo), 1):
                rec = {"id": t["id"], "kind": t["kind"], "outs": outs}
                results[t["id"]] = rec
                ck.write(json.dumps(rec) + "\n")
                ck.flush()
                if i % 20 == 0:
                    data_vol.commit()
        wall = max(time.time() - t_start, 1e-9)
        if reasoning:           # a reply to another prompt that could not be asked again (its task does not fit the slot now) is not stored under the task's prompt
            results = {tid: {**rec, "outs": outputs.current_outs(rec["outs"], prompt_of(by_id[tid]))} if tid in by_id else rec for tid, rec in results.items()}
        kept, stats = records.assemble(tasks, results, name, lambda t: domain.messages(t, mode), reasoning, thinking, max_traces, domain=domain)
        if reasoning:           # this call's own throughput, next to the cost line's assumption (a resumed run reports only the replies it asked)
            stats["perf"] = {**perf, "wall_s": round(wall), "tok_per_s": round(perf["completion_tokens"] / wall, 1), "s_per_request": round(wall / max(perf["requests"], 1), 2)}
            stats["sampling"] = config.sampling_record(thinking, sampling, max_tokens, seed_base=SEED_BASE, slot_context=slot_ctx, parallel=slots,
                                                       **({"max_tokens_by_family": by_family} if by_family else {}))
            if thinking:
                stats["slot"] = {"slot_context": slot_ctx, "max_prompt_tokens": slot_ctx - max_tokens, "prompt_too_long": sum(too_long.values()), "by_kind": dict(sorted(too_long.items())),
                                 **({"max_prompt_tokens_by_kind": {k: slot_ctx - n for k, n in sorted(by_family.items())}} if by_family else {})}
        windows = (config.THINKING_REPORT_WINDOWS if thinking else config.REPORT_WINDOWS) if reasoning else ()
        kept, stats = _apply_length_rule(kept, stats, window, windows)       # round 9, m2: what the training run would drop is dropped (and counted) here, while the server is still up
    finally:
        srv.kill()
        server_log.close()
        data_vol.commit()
    return {"stats": stats, "sft": kept}




# ---- the cost, before the GPU is rented ----------------------------------------------------------------------

def local_task_count(decompiled: pathlib.Path, quota: dict, n: int) -> tuple:
    """(tasks to sample, where the number comes from) for the cost line, which is printed before any GPU is rented. The task stage wrote `decompiled-<tag>.jsonl` here: its
    tasks with pseudocode, after the quota. Without that file: the sum of the quota (an upper bound: tasks the Modal compilers dropped are not known), else `--n`."""
    path = pathlib.Path(decompiled)
    if path.exists():
        tasks = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        tasks = [t for t in tasks if "pseudocode" in t]
        return len(config.apply_quota(tasks, quota)), f"{len(tasks)} decompiled tasks in {path.name}" + (f", capped by --quota" if quota else "")
    if quota:
        return sum(quota.values()), f"the sum of --quota ({path.name} is not here: tasks the gate drops are not counted)"
    return n, f"--n {n} ({path.name} is not here)"


# ---- the cost, before the GPU is rented --------------------------------------------------------------------------------------------

# MEASURED (owner's figures for the phase-A CodeForge run; not recorded in the repo's logs): 3,187 tasks x 6 short replies in about 5.5 GPU hours on one L40S.
PHASE_A_REPLIES = 3187 * 6
PHASE_A_GPU_HOURS = 5.5
PHASE_A_REPLY_S = PHASE_A_GPU_HOURS * 3600 / PHASE_A_REPLIES        # 1.04 s of GPU per reply, prefill and decode, four requests in flight
# ASSUMED (not measured: the phase-A run did not record `usage`): a phase-A reply is a headline and one sentence, about 50 tokens; a reasoning reply is 5 to 8 times as long
# (the owner's estimate), so its time is the phase-A time scaled by its length. Prefill does not scale with the reply, so this OVERestimates the time: the line is a budget guard.
PHASE_A_REPLY_TOKENS = 50
LENGTH_FACTORS = {"low": 5.0, "expected": 6.5, "high": 8.0}
L40S_USD_PER_H = 1.95                # Modal list price, US$0.000542 per second (https://modal.com/pricing); CPU and memory of the container add a few cents per hour


def reply_seconds(max_tokens: int, factor: float) -> float:
    """GPU seconds per reply of `factor` times the phase-A length, cut off at `max_tokens`."""
    return PHASE_A_REPLY_S * min(max_tokens, factor * PHASE_A_REPLY_TOKENS) / PHASE_A_REPLY_TOKENS


def project(n_tasks: int, k: int, max_tokens: int, usd_per_h: float = L40S_USD_PER_H, call_hours: float = 5.0) -> dict:
    """GPU hours and dollars of a sample run: `n_tasks` x `k` replies of 5x, 6.5x and 8x the phase-A length (low, expected, high), and the worst case in which every reply runs to
    `max_tokens`. `calls`: how many sampler calls the expected time needs (a call times out after `call_hours`; the next one continues with --resume)."""
    replies = n_tasks * k
    out = {"tasks": n_tasks, "k": k, "replies": replies, "max_tokens": max_tokens, "usd_per_h": usd_per_h, "call_hours": call_hours}
    for name, factor in {**LENGTH_FACTORS, "worst": max_tokens / PHASE_A_REPLY_TOKENS}.items():
        s = reply_seconds(max_tokens, factor)
        hours = replies * s / 3600
        out[name] = {"s_per_reply": round(s, 2), "gpu_hours": round(hours, 2), "usd": round(hours * usd_per_h, 2)}
    out["calls"] = max(1, math.ceil(out["expected"]["gpu_hours"] / call_hours))
    return out


def cost_line(p: dict, source: str = "") -> str:
    lo, ex, hi, worst = p["low"], p["expected"], p["high"], p["worst"]
    return (f"projected cost: {p['tasks']:,} tasks x k={p['k']} = {p['replies']:,} replies, {ex['s_per_reply']} s of GPU each = {ex['gpu_hours']} GPU-h on one L40S = about "
            f"US${ex['usd']:.2f} at US${p['usd_per_h']:.2f}/h (replies 5x to 8x as long as phase A's: {lo['gpu_hours']}-{hi['gpu_hours']} GPU-h, US${lo['usd']:.2f}-US${hi['usd']:.2f}; "
            f"worst case, every reply runs to max_tokens={p['max_tokens']}: {worst['gpu_hours']} GPU-h, US${worst['usd']:.2f}). Assumes phase A's measured {PHASE_A_REPLY_S:.2f} s per reply "
            f"({PHASE_A_GPU_HOURS} GPU-h for {PHASE_A_REPLIES:,} replies) for a reply of ~{PHASE_A_REPLY_TOKENS} tokens, scaled by length (an overestimate: prefill does not scale); "
            f"GPU only, no decompile. A sampler call times out after {p['call_hours']:g} h: this takes ~{p['calls']} call(s), each after the first with --resume."
            + (f" Tasks: {source}." if source else ""))


def budget_warning(p: dict, budget_usd: float):
    """A warning text when even the high estimate is over budget (or the expected one, a stronger text), else None. The decompile stage costs about US$1 per 1,200 tasks on top."""
    if p["expected"]["usd"] > budget_usd:
        return f"WARNING: the expected cost US${p['expected']['usd']:.2f} is over the budget US${budget_usd:.2f}: lower --quota or --k, or raise --budget-usd on purpose"
    if p["high"]["usd"] > budget_usd:
        return f"WARNING: the expected cost fits the budget US${budget_usd:.2f}, the high estimate (replies 8x as long) US${p['high']['usd']:.2f} does not; --quota or --k lower"
    return None


def thinking_cost_line(p: dict, source: str = "", by_family: dict = None, tokens: int = 0, slots: int = 0) -> str:
    """The line of a THINKING run (printed before the GPU is rented). There is no expected cost: the 5x-8x length of the visible-reasoning mode does not hold when the model reasons first and the
    length of its reasoning is not measured. What is known is the worst case, every reply running to `--max-tokens`. The slot of the server holds the prompt and that many tokens (`slot_context`).
    A run with a budget per family (`by_family`, `tokens` = `--max-tokens`, `slots` = the requests in flight of `config.server_plan`) has `p["max_tokens"]` = the longest budget: the room for a prompt is
    that budget's for the families that have it and the base budget's for the others, and the line says that the run has fewer requests in flight than the 4 the phase-A time was measured at."""
    w = p["worst"]
    slot = config.slot_context(p["max_tokens"])
    if by_family and p["max_tokens"] > tokens:
        room = (f"The prompt and the reply share one {slot:,}-token slot of the server, {slots} requests in flight (a prompt over {slot - p['max_tokens']:,} tokens is not asked and is counted for a family with a budget of "
                f"{p['max_tokens']:,}, over {slot - tokens:,} for the others; the worst case above assumes the longest budget for every reply, and phase A's four requests in flight, not {slots}: what that does to the speed is not measured).")
    else:
        room = f"The prompt and the reply share one {slot:,}-token slot of the server (a prompt over {slot - p['max_tokens']:,} tokens is not asked and is counted)."
    return (f"projected cost (thinking pilot): {p['tasks']:,} tasks x k={p['k']} = {p['replies']:,} replies. No estimate: the 5x-8x length assumption is for visible reasoning, and how long this model "
            f"reasons is not measured. Worst case, every reply runs to max_tokens={p['max_tokens']}: {w['gpu_hours']} GPU-h on one L40S = about US${w['usd']:.2f} at US${p['usd_per_h']:.2f}/h "
            f"(an overestimate: it scales the phase-A time per reply by length). Keep --quota small, then read `perf` and the share of replies cut off at max_tokens. {room} "
            f"A sampler call times out after {p['call_hours']:g} h: the worst case takes ~{max(1, math.ceil(w['gpu_hours'] / p['call_hours']))} call(s), each after the first with "
            f"--resume." + (f" Tasks: {source}." if source else ""))


def thinking_budget_warning(p: dict, budget_usd: float):
    """A warning when even the pilot's worst case is over the budget (there is no expected cost to compare), else None."""
    if p["worst"]["usd"] > budget_usd:
        return (f"WARNING: the worst case US${p['worst']['usd']:.2f} of this thinking run is over the budget US${budget_usd:.2f}; the expected cost is not known: lower --quota, --k or --max-tokens, "
                f"or raise --budget-usd on purpose")
    return None
