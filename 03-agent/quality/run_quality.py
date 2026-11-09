#!/usr/bin/env python3
"""Run the phase 3 problems against ONE server configuration (an "arm") and write JSONL.

    python run_quality.py --endpoint http://127.0.0.1:8081/v1 --arm A --thinking off --out results/phase3/A.jsonl

One request per problem to `<endpoint>/chat/completions`: temperature 0 and a fixed seed by default, `cache_prompt` off (so a request
never depends on what the server saw before), and the thinking switch sent explicitly as `chat_template_kwargs.enable_thinking`
(the template of this model thinks unless told not to). The code is extracted from the answer and run against the hidden tests.
The output file starts with a `meta` line, then one `result` line per (problem, sample); a rerun skips what is already there.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import qbench

RUNNER_VERSION = 1
FAIL_KINDS = ("no_code", "syntax", "test_fail", "timeout", "api_error")


def http_json(url: str, payload: dict[str, Any] | None = None, timeout: float = 30.0) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"} if data else {})
    with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310 - the endpoint is chosen by the operator
        return json.loads(r.read().decode("utf-8"))


def chat(endpoint: str, payload: dict[str, Any], timeout: float, retries: int = 2) -> dict[str, Any]:
    last: Exception | None = None
    for attempt in range(retries + 1):
        try:
            return http_json(endpoint.rstrip("/") + "/chat/completions", payload, timeout)
        except urllib.error.HTTPError as e:
            last = RuntimeError(f"HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:300]}")
            if e.code < 500:
                break
        except (urllib.error.URLError, OSError, ValueError, TimeoutError) as e:
            last = e
        time.sleep(2 + 3 * attempt)
    raise RuntimeError(str(last))


def tool_version(cmd: list[str]) -> str | None:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=20, stdin=subprocess.DEVNULL)
        return (r.stdout or r.stderr).strip().splitlines()[0]
    except (OSError, subprocess.SubprocessError, IndexError):
        return None


def server_props(endpoint: str) -> dict[str, Any]:
    """What llama-server says about itself (`/props`), trimmed. Empty for other servers."""
    root = endpoint.rstrip("/")
    root = root[:-3] if root.endswith("/v1") else root
    try:
        props = http_json(root + "/props", timeout=10)
    except Exception:  # noqa: BLE001 - optional information
        return {}
    gen = props.get("default_generation_settings") or {}
    keep = {k: props.get(k) for k in ("build_info", "model_path", "total_slots", "chat_template_caps") if k in props}
    keep["n_ctx"] = gen.get("n_ctx")
    keep["chat_template_chars"] = len(props.get("chat_template") or "")
    return keep


def build_payload(p: qbench.Problem, a: argparse.Namespace, sample: int) -> dict[str, Any]:
    thinking = a.thinking == "on"
    budget = p.max_tokens + (a.think_budget if thinking else 0)
    return {"model": a.model_name, "messages": qbench.build_messages(p, a.prompt_variant), "temperature": a.temperature, "seed": a.seed + sample,
            "max_tokens": budget, "stream": False, "cache_prompt": False,
            "chat_template_kwargs": {"enable_thinking": thinking}}


def run_problem(p: qbench.Problem, a: argparse.Namespace, sample: int) -> dict[str, Any]:
    thinking = a.thinking == "on"
    payload = build_payload(p, a, sample)
    rec: dict[str, Any] = {"type": "result", "problem": p.id, "lang": p.lang, "difficulty": p.difficulty, "subtle": p.subtle, "arm": a.arm,
                           "sample": sample, "prompt_variant": a.prompt_variant, "temperature": a.temperature, "seed": payload["seed"], "thinking": thinking,
                           "max_tokens": payload["max_tokens"], "at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    t0 = time.monotonic()
    try:
        resp = chat(a.endpoint, payload, a.request_timeout)
    except RuntimeError as e:
        rec.update({"pass": False, "kind": "api_error", "detail": str(e)[:300], "wall_s": round(time.monotonic() - t0, 2), "test_fraction": 0.0})
        return rec
    wall = time.monotonic() - t0
    choice = (resp.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or ""
    usage = resp.get("usage") or {}
    timings = resp.get("timings") or {}
    comp = usage.get("completion_tokens", timings.get("predicted_n"))
    tok_s = timings.get("predicted_per_second") or (comp / wall if comp and wall else None)
    finish = choice.get("finish_reason")
    rec.update({
        "finish_reason": finish, "cap_hit": finish == "length", "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": comp,
        "tok_s": round(tok_s, 3) if tok_s else None, "wall_s": round(wall, 2),
        "timings": {k: timings[k] for k in ("prompt_n", "prompt_ms", "prompt_per_second", "predicted_n", "predicted_ms", "predicted_per_second",
                                            "draft_n", "draft_n_accepted") if k in timings},
        "response": content, "reasoning": reasoning, "answer_chars": len(content), "reasoning_chars": len(reasoning),
        "budget_exhausted": bool(a.budget_message) and (a.budget_message in reasoning or a.budget_message in content)})
    code, how = qbench.extract_code(content, p.lang, thinking=thinking, reasoning_split=bool(reasoning))
    rec["extract"] = how
    if code is None:
        rec.update({"pass": False, "kind": "no_code", "detail": how, "test_fraction": 0.0})
        return rec
    rec["code"] = code
    out = qbench.run_tests(p, code, timeout=a.test_timeout, python=a.python, node=a.node)
    rec.update({"pass": out.passed, "kind": out.kind, "detail": out.detail, "tests_total": out.tests_total, "tests_failed": out.tests_failed,
                "test_s": round(out.seconds, 2), "normalized_exports": out.normalized})
    rec["test_fraction"] = round(out.fraction, 4)
    return rec


def done_keys(path: Path) -> set[tuple[str, int]]:
    if not path.exists():
        return set()
    _, rows = qbench.read_results(path)
    return {(r["problem"], r["sample"]) for r in rows if r.get("kind") != "api_error"}


def select_problems(a: argparse.Namespace) -> list[qbench.Problem]:
    probs = qbench.load_problems()
    if a.lang:
        probs = [p for p in probs if p.lang == a.lang]
    if a.problems:
        wanted = [w.strip() for w in a.problems.split(",") if w.strip()]
        probs = [p for p in probs if any(p.id == w or p.id.startswith(w) for w in wanted)]
    if a.max_problems:
        probs = probs[: a.max_problems]
    return probs


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--endpoint", default="http://127.0.0.1:8080/v1")
    ap.add_argument("--arm", required=True, help="label written into every record (A, B, ...)")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--thinking", choices=["on", "off"], default="off")
    ap.add_argument("--think-budget", type=int, default=2048, help="tokens added to the problem's budget when thinking is on (the cap)")
    ap.add_argument("--budget-message", default="", help="the text the server injects when its reasoning budget runs out (LLAMA_ARG_THINK_BUDGET_MESSAGE); "
                    "finding it in the answer or the reasoning is recorded as budget_exhausted")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--prompt-variant", choices=sorted(qbench.PROMPT_VARIANTS), default=qbench.DEFAULT_VARIANT, help="wording of the language note (see qbench.PROMPT_VARIANTS)")
    ap.add_argument("--samples", type=int, default=1, help="samples per problem (sample i uses seed+i)")
    ap.add_argument("--lang", choices=["py", "js"])
    ap.add_argument("--problems", help="comma-separated ids or id prefixes")
    ap.add_argument("--max-problems", type=int)
    ap.add_argument("--test-timeout", type=float, default=30.0)
    ap.add_argument("--request-timeout", type=float, default=1800.0)
    ap.add_argument("--model-name", default="local")
    ap.add_argument("--python", default=None, help="interpreter for the Python tests (default: this one)")
    ap.add_argument("--node", default=None, help="node executable (default: PATH, $QBENCH_NODE, C:\\Program Files\\nodejs)")
    ap.add_argument("--serve-json", type=Path, help="the JSON `forge serve` printed (command line, config); stored in the meta line")
    ap.add_argument("--stop-file", type=Path, help="stop cleanly (between two requests) as soon as this file exists; a rerun resumes")
    ap.add_argument("--note", default="")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    a = parse_args(argv)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    probs = select_problems(a)
    todo = [(p, s) for s in range(a.samples) for p in probs]
    seen = done_keys(a.out)
    todo = [(p, s) for p, s in todo if (p.id, s) not in seen]
    if not a.out.exists() or a.out.stat().st_size == 0:
        serve = json.loads(a.serve_json.read_text(encoding="utf-8-sig")) if a.serve_json and a.serve_json.exists() else None
        meta = {"type": "meta", "runner_version": RUNNER_VERSION, "arm": a.arm, "endpoint": a.endpoint, "thinking": a.thinking == "on",
                "think_budget": a.think_budget if a.thinking == "on" else 0, "budget_message": a.budget_message, "prompt_variant": a.prompt_variant, "temperature": a.temperature, "seed": a.seed, "samples": a.samples,
                "started": time.strftime("%Y-%m-%dT%H:%M:%S"), "host": platform.node(), "platform": platform.platform(),
                "python": sys.version.split()[0], "node": tool_version([qbench.find_node(a.node) or "node", "--version"]),
                "server_props": server_props(a.endpoint), "serve": serve, "note": a.note, "argv": sys.argv[1:]}
        with a.out.open("a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(meta) + "\n")
    print(f"[{a.arm}] {len(todo)} requests to do ({len(seen)} already in {a.out})", flush=True)
    t_start = time.monotonic()
    for i, (p, s) in enumerate(todo, 1):
        if a.stop_file and a.stop_file.exists():
            print(f"[{a.arm}] stop file {a.stop_file} found: stopping before request {i}/{len(todo)} ({p.id} s{s})", flush=True)
            return 0
        rec = run_problem(p, a, s)
        with a.out.open("a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(rec) + "\n")
            f.flush()
            os.fsync(f.fileno())
        elapsed = time.monotonic() - t_start
        eta = elapsed / i * (len(todo) - i)
        print(f"[{a.arm}] {i}/{len(todo)} {p.id} s{s} {'PASS' if rec['pass'] else 'FAIL ' + str(rec.get('kind'))}"
              f"{' CAP' if rec.get('cap_hit') else ''} tokens={rec.get('completion_tokens')} tok/s={rec.get('tok_s')} wall={rec['wall_s']}s eta={eta / 60:.0f}min", flush=True)
    print(f"[{a.arm}] done in {(time.monotonic() - t_start) / 60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
