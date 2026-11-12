"""Inputs and procedures of the byte-for-byte check of the REA runs across the `distill/` refactor (tests/test_distill_golden.py).

`tests/golden/distill/expected/*` was written by the code of origin/dataset/phase-a (commit 9afd8ae, before the refactor) by `tests/golden_baseline.py` (it runs the procedures of this file in a worktree of that
commit; `tests/golden/distill/provenance.json` records the sha256 of what it writes and of every expected file, and test_distill_golden.Frozen checks them). The procedures are the ones in this
file: every function here takes the module or the test case it needs as an argument, so the same procedure runs against the old modules (`rea_bench.reasoning`, `rea_bench.sft`,
`modal_rea_data`, `tools/kaggle/build_tasks.py`) and the new ones, and nothing here imports either of them. The tasks are frozen in `tests/golden/distill/tasks.jsonl` (made once by
`write_tasks`), the replies are built from them by plain string code below, so the only thing that can move an expected file is a change in what the pipeline does.
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import pathlib
import tempfile

from .helpers import ROOT

GOLDEN = ROOT / "tests" / "golden" / "distill"
WRONG = "0x0badc0de"          # not the answer of any task, numeric or string
CJK = "这个函数读取指针并加上偏移量。"
STEPS = [
    "The function reads a field at a fixed offset from the base.\nThe constant is what the question asks for.",
    "Following the loop gives the key material directly.\nOnly one value of it is needed here.",
    "Looking at the comparison the literal stands out clearly.\nThat literal answers the question.",
]
LOOP = "\n".join(["Wait, let me recheck the code again."] * 5)
MODES = {
    "plain": {},
    "reasoning": dict(reasoning=True, max_len=4096),
    "thinking": dict(reasoning=True, thinking=True, max_tokens=900, max_len=8192),
    "reasoning-quota": dict(reasoning=True, max_len=4096, quota="structs=2,crc=1,xor-key=1"),
}
K = 3


# ---- the frozen tasks -------------------------------------------------------------------------------------------------------------------

def write_tasks() -> list:
    """(Re)make tests/golden/distill/tasks.jsonl from the generator: the 12 legacy tasks of seed 7 and one task of ten variants of the new families, each with a stand-in pseudocode that holds its id
    (the fake server finds the task by it); the first has a non-ASCII identifier of its own. Run once, from the code that wrote the expected files."""
    from .rea_tasks import meta
    from rea_bench import families
    out = []
    for i, t in enumerate(families.generate("legacy", 12, 7)):
        out.append({**t.meta(), "pseudocode": f"/* {t.id} */ void f(void) {{ int größe = {i}; }}"})
    for fam, variant, extra in [("structs", "linked_next", {"chance": 0.07}), ("structs", "array_stride", {}), ("pktsum", "sum8", {"chance": 0.01}), ("crc", "bitwise_init", {"chance": 0.55}),
                                ("cipher", "xtea_dec", {}), ("bounds", "window", {}), ("encoding", "base64_encode", {}), ("antidebug", "env", {}), ("multi", "decode_secret", {}),
                                ("strobf", "stack_string", {})]:
        t = meta(fam, variant, **extra)
        out.append({**t, "pseudocode": f"/* {t['id']} */ void f(void) {{}}", **({"refs": "00 01 02 03"} if variant == "array_stride" else {})})
    GOLDEN.mkdir(parents=True, exist_ok=True)
    (GOLDEN / "tasks.jsonl").write_text("".join(json.dumps(t) + "\n" for t in out))
    return out


def load_tasks() -> list:
    return [json.loads(line) for line in (GOLDEN / "tasks.jsonl").read_text().splitlines() if line.strip()]


def right(t: dict) -> str:
    from rea_bench import scoring
    return scoring.canonical(scoring.strict_task(t))


# ---- what the fake server answers ---------------------------------------------------------------------------------------------------------

def reply_text(t: dict, steps: str, answer: str = None) -> str:
    return f"{steps}\nAnswer: {right(t) if answer is None else answer}"


def server_text(tasks: list, tid: str, i: int, mode: str) -> tuple:
    """(content, finish) of sample i of task `tid` in `mode`: right and wrong answers, a duplicate, mixed language, a repetition loop, an answer cut off, text after the answer."""
    n = next(j for j, t in enumerate(tasks) if t["id"] == tid)
    t = tasks[n]
    if mode == "plain":
        if i == 0:
            return f"{right(t)}\nThe value is read straight from the code.", "stop"
        if i == 1:
            return [right(t), WRONG, f"{right(t)}\nIt is the immediate operand.", f"{right(t)}\n{WRONG}"][n % 4], "stop"
        return [WRONG, right(t), f"{right(t)}\nThe code returns it.", "The constant is"][n % 4], ("length" if n % 4 == 3 else "stop")
    if i == 0:
        return reply_text(t, STEPS[0]), "stop"
    if i == 1:
        return (reply_text(t, STEPS[0]).replace("The function", "the function") if n % 4 == 0 else reply_text(t, STEPS[1])), "stop"
    kind = n % 6
    if kind == 0:
        return reply_text(t, STEPS[2]), "stop"
    if kind == 1:
        return reply_text(t, STEPS[2], WRONG), "stop"
    if kind == 2:
        return reply_text(t, f"Step for {tid}.\n{CJK}"), "stop"
    if kind == 3:
        return "Step for it.\nThe constant is", "length"
    if kind == 4:
        return reply_text(t, LOOP), "stop"
    return f"{reply_text(t, STEPS[2])}\nSo that is it.", "stop"


def server_response(tasks: list, tid: str, i: int, mode: str) -> dict:
    n = next(j for j, t in enumerate(tasks) if t["id"] == tid)
    content, finish = server_text(tasks, tid, i, mode)
    msg = {"content": content}
    if mode == "thinking":
        msg["reasoning_content"] = f"Reasoning about {tid}, sample {i}: the code reads it, then returns it."
    return {"choices": [{"message": msg, "finish_reason": finish}], "usage": {"completion_tokens": 17 + i, "prompt_tokens": 100 + n}}


def kaggle_lines(tasks: list, mode: str) -> list:
    """The replies.jsonl lines the Kaggle notebook would have written for the same answers (`text` holds the reasoning in a `<think>` block in a thinking run)."""
    lines = []
    for t in tasks:
        outs = []
        for i in range(K):
            r = server_response(tasks, t["id"], i, mode)
            ch = r["choices"][0]
            text = ch["message"].get("content") or ""
            if ch["message"].get("reasoning_content"):
                text = f"<think>{ch['message']['reasoning_content']}</think>\n\n{text}"
            outs.append({"text": text, "finish": ch["finish_reason"], "tokens": r["usage"]["completion_tokens"], "prompt_tokens": r["usage"]["prompt_tokens"], "secs": 1.0})
        lines.append({"id": t["id"], "kind": t["kind"], "outs": outs})
    return lines


# ---- the procedures ---------------------------------------------------------------------------------------------------------------------

def stage_artifacts(case, mode: str) -> dict:
    """Run the sample stage of a `StubbedSampleStage` case on the frozen tasks, first against a server that fails the second sample of every fifth task, then again with --resume:
    stats (without the wall-clock `perf`), the records, the requests (sorted), the checkpoint and what was printed, of both calls."""
    from rea_bench import outputs
    tasks = load_tasks()
    src = case.root / "data" / outputs.decompiled_stem(case.tag)
    for p in src.glob("*.json"):
        p.unlink()
    case.tasks = {t["id"]: t for t in tasks}
    for t in tasks:
        (src / f"{t['id']}.json").write_text(json.dumps(t))
    case.k = K
    use = "thinking" if mode == "thinking" else "plain" if mode == "plain" else "reasoning"
    case.reply = lambda tid, i, body: server_response(tasks, tid, i, use)
    dead = {t["id"] for j, t in enumerate(tasks) if j % 5 == 0}
    kw = MODES[mode]
    out = {}
    for step, (resume, failing) in (("first", (False, True)), ("resumed", (True, False))):
        case.fail = (lambda tid, i: tid in dead and i == 1) if failing else (lambda tid, i: False)
        res = case.run_stage(resume, **kw)
        stats = {k: v for k, v in res["stats"].items() if k != "perf"}
        partial = (case.root / "data" / f"{outputs.sft_stem('m', outputs.sample_tag(case.tag, kw.get('reasoning', False), kw.get('thinking', False)))}.partial.jsonl").read_text()
        bodies = sorted(case.bodies, key=lambda b: (b["seed"], json.dumps(b["messages"])))
        out[step] = {"stats": stats, "sft": res["sft"], "requests": sorted(case.requests), "bodies_sha256": sha(json.dumps(bodies, sort_keys=True)), "bodies": len(bodies),
                     "first_body": bodies[0] if bodies else None, "partial_sha256": sha(partial), "printed": case.printed, "tokenized": len(case.tokenized)}
    return out


def score_artifacts(kaggle_score, mode: str) -> dict:
    """`kaggle_score.score` on the frozen tasks and the replies of the same fake server, and the CLI's own output and files."""
    tasks = load_tasks()
    thinking = mode == "thinking"
    run_info = {"thinking": thinking, "sampling": {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0} if thinking else {"temperature": 0.7}, "max_tokens": 900 if thinking else 768,
                "seed_base": 1234, "ctx_per_slot": 8192, "slots": 4}
    with tempfile.TemporaryDirectory() as d:
        d = pathlib.Path(d)
        tp = d / "decompiled-seed7-all.jsonl"
        tp.write_text("".join(json.dumps(t) + "\n" for t in tasks))
        rp = d / "replies.jsonl"
        lines = kaggle_lines(tasks, mode)
        for ln in lines:                       # a notebook's fingerprint of the prompt it asked is checked against the rebuilt one: one line is for another prompt
            ln["prompt_sha"] = "0" * 16 if ln["id"] == tasks[3]["id"] else None
        rp.write_text("".join(json.dumps({k: v for k, v in ln.items() if v is not None}) + "\n" for ln in lines))
        records, stats = kaggle_score.score(rp, tp, "m", thinking=thinking, run_info=run_info)
        (d / "run.json").write_text(json.dumps(run_info))
        printed = io.StringIO()
        from rea_bench import cli
        with contextlib.redirect_stdout(printed), contextlib.redirect_stderr(io.StringIO()):
            cli.main(["kaggle-score", "--replies", str(rp), "--tasks", str(tp), "--model", "m", *(["--thinking"] if thinking else []), "--max-traces", "2"])
        written = {p.name: sha(p.read_text()) for p in sorted(d.glob("sft-*"))}
        return {"records": records, "stats": stats, "cli_stdout": printed.getvalue().replace(str(d), "<tmp>"), "cli_files": written}


def plan_artifacts(ns) -> dict:
    """The planning functions (cost line, sampling, slot, quota, fingerprint) for a few inputs, from a namespace that has them under their old names."""
    out = {"sampling": {}, "max_tokens": {}, "slot": {}, "fits": {}, "errors": {}}
    for thinking in (False, True):
        out["sampling"][str(thinking)] = ns.sampling_for(thinking)
        out["sampling"][f"{thinking}-over"] = ns.sampling_for(thinking, temperature=0.5, top_p=0.9)
        for reasoning in (False, True):
            if reasoning or not thinking:
                out["max_tokens"][f"{reasoning}-{thinking}"] = ns.default_max_tokens(reasoning, thinking)
    for mt in (768, 2048, 4096, 6000, 8192):
        out["slot"][str(mt)] = ns.slot_context(mt)
        out["fits"][str(mt)] = [ns.prompt_fits(p, mt, ns.slot_context(mt)) for p in (100, 4096, 4097, 5000)]
    for bad in ({"temperature": 3}, {"top_k": -2}):
        try:
            ns.sampling_for(False, **bad)
        except SystemExit as e:
            out["errors"][json.dumps(bad)] = str(e)
    try:
        ns.sampling_for(True, temperature=0)
    except SystemExit as e:
        out["errors"]["greedy"] = str(e)
    plan = ns.project(120, 4, 768)
    out["project"] = plan
    out["cost_line"] = ns.cost_line(plan, "120 tasks")
    out["budget_warning"] = [ns.budget_warning(ns.project(n, 4, 768), 30.0) for n in (120, 1260, 4000)]
    tp = ns.project(40, 2, 4096)
    out["thinking"] = {"project": tp, "line": ns.thinking_cost_line(tp, "40 tasks"), "warning": [ns.thinking_budget_warning(ns.project(n, 4, 4096), 30.0) for n in (40, 400, 4000)]}
    out["sampling_record"] = ns.sampling_record(True, ns.sampling_for(True), 4096, seed_base=1234, slot_context=8192, parallel=4)
    out["quota"] = ns.parse_quota("crc=250,structs=100")
    for bad in ("", "crc", "crc=0", "crc=1,crc=2", "x=1"):
        try:
            ns.parse_quota(bad, ["crc", "structs"])
        except SystemExit as e:
            out["errors"][f"quota {bad!r}"] = str(e)
    tasks = load_tasks()
    out["apply_quota"] = [t["id"] for t in ns.apply_quota(tasks, {"crc": 1, "structs": 2, "xor-key": 1})]
    out["prompt_sha"] = ns.prompt_sha([{"role": "user", "content": "x"}])
    return out


def lora_artifacts(lora) -> dict:
    """The pure functions of the training script, on a grid of inputs (`lora` = the module that has them)."""
    out = {"run_tag": {}, "batch_plan": {str(m): lora.batch_plan(m) for m in (1024, 2048, 3072, 4096, 6144, 8192)}, "parse": {}, "fate": {}}
    for smoke in (False, True):
        for config, tag, mix in [("rea_sft", "", False), ("rea_reasoning", "", False), ("rea_thinking", "", True), ("rea_sft", "mine", False), ("a b/c", "", False)]:
            key = f"{smoke}|{config}|{tag}|{mix}"
            try:
                out["run_tag"][key] = lora.run_tag(smoke, config, tag, mix)
            except SystemExit as e:
                out["run_tag"][key] = f"SystemExit: {e}"
    for tag in ("full", "smoke", "-", ""):
        try:
            out["run_tag"][f"reserved {tag}"] = lora.run_tag(False, "x", tag)
        except SystemExit as e:
            out["run_tag"][f"reserved {tag}"] = f"SystemExit: {e}"
    for text in ("plain answer", "<think>\nwhy\n</think>\n\nanswer", "<think>\n\n</think>\n\nanswer", "<think>\nwhy", "<think>a</think>b<think>c</think>d", "x <think>a</think> y", "<think>\nwhy\n</think>"):
        out["parse"][text] = list(lora.parse_assistant(text))
        out["fate"][text] = [lora.record_fate(text, c, m) for c in (False, True) for m in (False, True)]
    out["ids"] = [lora.encode_ids([1, 2], [1, 2, 3, 4], m) for m in (3, 4, 100)] + [lora.encode_ids([1, 2], [1, 2], 100)]
    out["fate_counts"] = lora.fate_counts([("think", True), ("think", False), ("plain", False), ("plain", True), ("skipped_plain", False), ("malformed", False)])
    return out


def kit_artifacts(build_tasks) -> dict:
    """The files `build_tasks.build` writes for the frozen tasks, except the kernel script (a copy of run_teacher.py)."""
    tasks = load_tasks()
    out = {}
    with tempfile.TemporaryDirectory() as d:
        d = pathlib.Path(d)
        tp = d / "decompiled-seed9-all.jsonl"
        tp.write_text("".join(json.dumps(t) + "\n" for t in tasks))
        for name, kw in {"thinking": dict(quota="structs=2,crc=1"), "visible": dict(quota="structs=2,pktsum=1", thinking=False, k=3), "tokens": dict(max_tokens=2048, sampling_overrides={"temperature": 0.5})}.items():
            o = d / name
            cfg = build_tasks.build([tp], o, "someone", **kw)
            out[name] = {"config": cfg, "files": {str(p.relative_to(o)): (p.read_text() if p.stat().st_size < 3000 else sha(p.read_text())) for p in sorted(o.rglob("*")) if p.is_file() and p.name != "run.py"}}
    return out


def trace_artifacts(rsn) -> dict:
    """The answer-line, think and trace-filter functions on crafted replies for every frozen task (`rsn` = a namespace that has them under their old names): which filter drops what, the traces a task
    keeps under each cap, with and without thinking, and the helpers of the duplicate rule."""
    tasks = load_tasks()
    out = {"verdict": {}, "drop": {}, "final": {}, "target": {}, "pick": {}, "text": {}, "dup": {}, "direct": {}}
    for t in tasks:
        k = right(t)
        wrong_num = "0x1234" if k.lower().startswith("0x") else "7777"
        cases = {
            "right": reply_text(t, STEPS[0]),
            "restating": f"The answer is {k}.\nAnswer: {k}",
            "no analysis": f"Answer: {k}",
            "contradiction": f"The code loads a field.\nSo the answer is {wrong_num}.\nAnswer: {k}",
            "corrected": f"The code loads a field.\nSo the answer is {wrong_num}.\nWait, actually it is another value.\nAnswer: {k}",
            "hedge": f"The code loads a field.\nSo the answer is {wrong_num} or {k}.\nAnswer: {k}",
            "wrong": reply_text(t, STEPS[1], WRONG),
            "text after": f"{reply_text(t, STEPS[1])}\nThat is all.",
            "no line": f"{STEPS[1]}\nThe value is {k}.",
            "truncated": f"{STEPS[1]}\nAnswer: {k}\n{rsn.TRUNCATED}",
            "think tag": f"<think>\nhidden\n</think>\n{reply_text(t, STEPS[1])}",
            "think open": f"<think>\nhidden\n{reply_text(t, STEPS[1])}",
            "error": "[error TimeoutError()]",
            "cjk": reply_text(t, f"Step.\n{CJK}"),
            "loop": reply_text(t, LOOP),
            "bold": f"{STEPS[2]}\n**Answer:** {k}",
            "größe": reply_text(t, "The variable größe holds it.\nThe constant is the offset."),
        }
        for name, text in cases.items():
            key = f"{t['id']}|{name}"
            out["verdict"][key] = rsn.verdict(text, t)
            out["drop"][key] = [rsn.quality_drop(text, t, False), rsn.quality_drop(text, t, True), rsn.score_reasoning(text, t)]
            out["final"][key] = rsn.final_answer(text)
        out["target"][t["id"]] = [rsn.teacher_target(cases["right"], t, False), rsn.teacher_target(cases["wrong"], t, False)]
        thought = rsn.thinking_reply(reply_text(t, STEPS[0]), f"I reason about {t['id']}.")
        out["text"][t["id"]] = {"thinking_reply": thought, "inline": rsn.thinking_reply(f"<think>why</think>{reply_text(t, STEPS[1])}"), "dangling": rsn.thinking_reply(f"why</think>\n{reply_text(t, STEPS[1])}"),
                                "reply_text": [rsn.reply_text(reply_text(t, STEPS[0]), "length"), rsn.reply_text(reply_text(t, STEPS[0]), "stop", "why", True), rsn.reply_text(reply_text(t, STEPS[0]), "stop", "", False)],
                                "trace": [rsn.trace_text(thought, True), rsn.trace_text(thought, False)], "think_drop": rsn.quality_drop(thought, t, True)}
        replies = [reply_text(t, s) for s in STEPS] + [reply_text(t, STEPS[0]).replace("The function", "the function"), reply_text(t, f"Step.\n{CJK}"), reply_text(t, STEPS[2], WRONG)]
        for cap in (1, 2, 3):
            for thinking in (False, True):
                batch = [rsn.thinking_reply(r, f"Reasoning {j} for {t['id']}: unique words {j * 17}.") for j, r in enumerate(replies)] if thinking else replies
                kept, dropped = rsn.pick_traces([r for r in batch if rsn.score_reasoning(r, t)], t, cap, thinking)
                out["pick"][f"{t['id']}|{cap}|{thinking}"] = [kept, dict(dropped)]
    a, b = reply_text(tasks[0], STEPS[0]), reply_text(tasks[0], STEPS[0]).replace("The function", "Function")
    out["dup"] = {"normalized": rsn.normalized(f"<think>X  Y</think>\n  Z 0x20"), "token_set": sorted(rsn.token_set(a)), "jaccard": [rsn.jaccard(rsn.token_set(a), rsn.token_set(b)), rsn.jaccard(set(), set())],
                  "is_duplicate": [rsn.is_duplicate(a, b), rsn.is_duplicate(a, reply_text(tasks[0], STEPS[1])), rsn.is_duplicate(a, a.upper())]}
    out["direct"] = {"unreadable": [rsn.unreadable("forty", "40"), rsn.unreadable("the answer is 0x20", "32"), rsn.unreadable("It reads the table.", "32"), rsn.unreadable("", "x")],
                     "mixed": [rsn.mixed_language(CJK), rsn.mixed_language("é" * 2 + "a" * 200), rsn.mixed_language("größe is used", tasks[0]), rsn.mixed_language("größe is used"), rsn.mixed_language("")],
                     "repetition": [rsn.repetition(LOOP), rsn.repetition(STEPS[0]), rsn.repetition("a b c d e f g h " * 5)],
                     "messages": [rsn.messages(tasks[0]), rsn.user_message({**tasks[0], "refs": "00 ff"})], "prompt_sha": rsn.prompt_sha(rsn.messages(tasks[0])), "clean": rsn.clean_reply("<think>a</think>  x \n"),
                     "compose": [rsn.compose_trace("why", "vis"), rsn.compose_trace("", "vis")], "kinds": rsn.known_kinds()}
    return out


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def dumps(obj) -> str:
    """The form of the expected files: insertion order kept (it is the order of the stats.json a run writes), one space of indent."""
    return json.dumps(obj, indent=1, ensure_ascii=False) + "\n"
