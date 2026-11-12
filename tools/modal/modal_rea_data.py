"""Training data for the REA skill, generated on Modal with automatic ground truth. WRAPPER, for the sample stage: the decompile stage (Ghidra) is REA's and stays here; `--stage sample` is
`distill/backends/modal_sample.py` (any subject) registered below with REA's image, volumes and domain (`distill/domains/rea`), and every command of the READMEs works as before.

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
never sampled (the stage refuses it). Output: SFT pairs in results/phase6/rea-data/. A pair whose system + user + assistant text is longer than the training window (`distill.config.MAX_LEN`, 2048 tokens
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
if (ROOT / "distill").is_dir() and str(ROOT) not in sys.path:      # the pipeline's own package; inside the container it is mounted at /root/distill
    sys.path.insert(0, str(ROOT))
from modal_qbench import MODELS, models_vol  # noqa: E402
from modal_rea import image, chat, SYSTEM_RECIPE  # noqa: E402  (same REA/Ghidra image)
from distill import config, records  # noqa: E402  (stdlib-only)
from distill.backends import modal_sample  # noqa: E402  (the sample stage: no Modal import in it)
from distill.backends.modal_sample import MAX_LEN, SAMPLER_SLOTS, SAMPLER_TIMEOUT_H, SEED_BASE, THINKING_SERVER_ARGS, _apply_length_rule, _check_thinking, _window  # noqa: E402,F401  (the stage's names, as they were here)
from distill.domains.rea import AGREE_CHANCE, DOMAIN  # noqa: E402  (REA as a subject: the adapter of rea_bench)
from rea_bench import dedupe, families, outputs, survive  # noqa: E402  (stdlib-only)
from rea_bench.scoring import check_mode  # noqa: E402  (stdlib-only, unit-tested without Modal)

image = image.add_local_python_source("modal_rea")     # modal_rea's image already mounts rea_bench at /root/rea_bench and distill at /root/distill (the sample stage and the REA domain)
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


def _messages(t: dict, reasoning: bool = False) -> list:
    """The system and user message the sampler answers (and the SFT record is made of): `DOMAIN.messages`. `reasoning`: the system message asks for the steps and a last `Answer:` line
    (rea_bench.reasoning.SYSTEM_REASONING); SYSTEM_RECIPE is untouched, and the user message is the same."""
    return DOMAIN.messages(t, "reasoning" if reasoning else "plain")


def _tasks_dir(tag: str) -> pathlib.Path:
    """Where the decompile stage's tasks of `tag` are on the volume (`/decompiled-<tag>/`; a run made before the dirs were keyed by family is `/decompiled-seed<seed>/`)."""
    src_dir = pathlib.Path(f"/data/{outputs.decompiled_stem(tag)}")
    if not src_dir.is_dir() and tag.startswith("seed") and tag.endswith("-legacy"):
        src_dir = pathlib.Path(f"/data/decompiled-{tag[:-len('-legacy')]}")     # made before the dirs were keyed by family
    return src_dir


@app.function(gpu=["L40S", "A100-40GB"], volumes={"/models": models_vol, "/data": data_vol}, timeout=SAMPLER_TIMEOUT_H * 3600)
def sample_answers(name: str, repo: str, fname: str, k: int, tag: str = "seed7-legacy", resume: bool = False, reasoning: bool = False, max_tokens: int = 0,
                   max_len: int = 0, quota: str = "", thinking: bool = False, max_traces: int = config.DEFAULT_MAX_TRACES, sampling: dict = None, max_tokens_by_family: dict = None) -> dict:
    """The sample stage (`distill/backends/modal_sample.py`, which has the documentation of every argument) for the REA domain, on this app's GPU function."""
    data_vol.reload()
    return modal_sample.sample_answers(DOMAIN, data_vol, _tasks_dir(tag), name, repo, fname, k, tag, resume, reasoning, max_tokens, max_len, quota, thinking, max_traces, sampling, max_tokens_by_family)


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
         max_len: int = 0, quota: str = "", budget_usd: float = 30.0, gpu_usd_per_h: float = modal_sample.L40S_USD_PER_H, thinking: bool = False,
         max_traces: int = config.DEFAULT_MAX_TRACES, temperature: float = -1.0, top_p: float = -1.0, top_k: int = -1, min_p: float = -1.0, run_config: str = "", max_tokens_by_family: str = ""):
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
    max_tokens_by_family (--reasoning): `kind=N,...`, the reply budget of the tasks of those families instead of --max-tokens (the thinking pilot cut every cipher reply off at 4096: `cipher=8192`); the server's slot is
    sized for the longest budget of a task in the run (4 slots of 8192 tokens, or 2 of 12288 when a task needs 8192).
    run_config: a JSON file of `distill.config.RunConfig` (`distill/configs/rea-*.json`) that sets model, k, mode, max_tokens, max_tokens_by_family, max_len, quota, max_traces and sampling instead of their flags (--stage sample).
    Outputs are keyed by seed and family (rea_bench.outputs); an existing output is never overwritten without --force."""
    if stage not in ("decompile", "sample"):         # `--stage decompil` used to rent a GPU for a sample stage (round 4, m4)
        raise SystemExit(f"unknown stage {stage!r}; known: decompile, sample")
    if run_config:
        if stage != "sample":
            raise SystemExit("--run-config is for --stage sample")
        f = config.merge_flags(run_config, dict(model=model, k=k, reasoning=reasoning, thinking=thinking, max_tokens=max_tokens, max_len=max_len, quota=quota, max_traces=max_traces,
                                                temperature=temperature, top_p=top_p, top_k=top_k, min_p=min_p, max_tokens_by_family=max_tokens_by_family), DOMAIN.name)
        model, k, reasoning, thinking, max_tokens, max_len, quota, max_traces = f["model"], f["k"], f["reasoning"], f["thinking"], f["max_tokens"], f["max_len"], f["quota"], f["max_traces"]
        temperature, top_p, top_k, min_p = f["temperature"], f["top_p"], f["top_k"], f["min_p"]
        max_tokens_by_family = f["max_tokens_by_family"]
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
        if reasoning or quota or max_tokens or max_len or thinking or max_traces != config.DEFAULT_MAX_TRACES or max_tokens_by_family:
            print("note: --reasoning, --thinking, --max-traces, --max-tokens, --max-tokens-by-family, --max-len and --quota belong to --stage sample; the decompile stage ignores them (--family and --n are what limit it)")
        dedupe.check_heldout_range(names, n, seed)          # the pinned held-out list covers 120 tasks per family
        _decompile_stage(names, n, seed, cc, exclude_seed, force, local_gate, tag)
    else:
        dedupe.check_sample_seed(seed)                       # the held-out split is never sampled for training (m7)
        m = [x for x in MODELS if x[0] == model]
        if not m:
            raise SystemExit(f"unknown model {model}; known: {[x[0] for x in MODELS]}")
        if k < records.AGREE_MIN:
            print(f"warning: k={k}: no task with a chance level of {AGREE_CHANCE} or more (CRC init/xorout, pktsum, fsm, TEA/XTEA, ...) can have "
                  f"{records.AGREE_MIN} right samples, so none of them is kept")
        name, repo, fname, _ = m[0]
        _check_thinking(reasoning, thinking, max_tokens, max_traces)      # --thinking without --reasoning stops here, before anything is rented
        window = _window(reasoning, max_len)                 # a reasoning run without --max-len stops here, before anything else is checked or rented
        given = lambda v: None if v < 0 else v               # a negative value is "not given": the default of the mode
        sampling = config.sampling_for(thinking, temperature=given(temperature), top_p=given(top_p), top_k=given(top_k), min_p=given(min_p))      # a thinking run that is greedy stops here too
        want = config.parse_quota(quota, DOMAIN.kinds()) if quota else {}
        by_family = config.parse_family_tokens(max_tokens_by_family, DOMAIN.kinds())      # a typo'd family stops the run here, before anything is rented
        if by_family and not reasoning:
            raise SystemExit("--max-tokens-by-family is for --reasoning or --thinking")
        rtag = outputs.sample_tag(tag, reasoning, thinking)  # a reasoning run's checkpoint and output are its own: `-reasoning` (`-reasoning-think` with --thinking) in the tag
        out, stats = (dest / f"{outputs.sft_stem(name, rtag)}.{ext}" for ext in ("jsonl", "stats.json"))
        outputs.refuse_overwrite([out, stats], force)       # before a GPU is rented
        if reasoning:
            tokens = max_tokens or config.default_max_tokens(reasoning, thinking)
            count, source = modal_sample.local_task_count(dest / f"{outputs.decompiled_stem(tag)}.jsonl", want, n)
            longest = config.run_max_tokens(list(want), tokens, by_family) if want else max([tokens, *by_family.values()])      # the longest budget of a family in the run (without a quota: of every family)
            plan = modal_sample.project(count, k, longest, gpu_usd_per_h, SAMPLER_TIMEOUT_H)      # an overestimate when only some families have it: the line is a budget guard
            slots, slot_tokens = config.server_plan(tokens, longest, modal_sample.SAMPLER_SLOTS)
            print(modal_sample.thinking_cost_line(plan, source, by_family, tokens, slots) if thinking else modal_sample.cost_line(plan, source))
            warning = modal_sample.thinking_budget_warning(plan, budget_usd) if thinking else modal_sample.budget_warning(plan, budget_usd)
            if warning:
                print("\n" + "!" * 100 + f"\n{warning}\n" + "!" * 100 + "\n")
            print(f"window of the length rule: --max-len {window}; max_tokens {tokens}; up to {max_traces} traces per kept task; checkpoint and output tag {rtag}")
            print(f"sampling: {sampling}; server slot {slot_tokens:,} tokens (prompt + max_tokens)"
                  + (f", {slots} slots; max_tokens per family {config.family_tokens_text(by_family)} (the cost line above assumes the longest, {longest}, for every reply)" if by_family else ""))
            if thinking and window < config.THINKING_WINDOW:
                print(f"note: --max-len {window} is below the recommended {config.THINKING_WINDOW} for thinking records: a record longer than it is dropped (the length report counts what fits "
                      f"{' and '.join(map(str, config.THINKING_REPORT_WINDOWS))}); modal_lora.py needs the same --max-len")
        res = sample_answers.remote(name, repo, fname, k, tag, resume=resume, reasoning=reasoning, max_tokens=max_tokens, max_len=max_len, quota=quota,
                                    thinking=thinking, max_traces=max_traces, sampling=sampling, max_tokens_by_family=by_family or None)
        outputs.write_atomic(out, (json.dumps(r) for r in res["sft"]))
        stats.write_text(json.dumps(res["stats"], indent=1))
        s = res["stats"]
        print(name, rtag, {f: s.get(f) for f in ("tasks", "answers", "errors", "correct", "solved_tasks", "kept_tasks", "kept_records", "agreement")}, "->", out)
        warning = records.error_warning(s)
        if warning:
            print("\n" + "!" * 100 + f"\n{warning}\n" + "!" * 100 + "\n")
        length_line = records.over_length_line(s)
        if length_line:
            print(length_line)
        for line in (records.fits_line(s), *records.reasoning_report(s)):
            if line:
                print(line)
        for kind, row in s.get("by_kind", {}).items():
            print(f"  {kind:10s} {row['solved_tasks']:4d}/{row['tasks']:<4d} solved, {row['kept_tasks']:4d} kept ({row['agreement']} dropped for lack of "
                  f"agreement, {row.get('errors', 0)} failed requests)")
