#!/usr/bin/env python3
"""A one-tool MCP server for OpenCode (phase 5, arm O30-tool). Stdlib only, Windows and Linux.

The tool, `run_tests`, runs the repository's *visible* tests and returns a short answer: `ALL TESTS PASS` or only the FIRST failing test
with its trimmed output. It is the one useful idea of the phase 4 orchestrator (`orchestrate.run_tests`, `parse_python`, `parse_node` and
`_trim_traceback` are reused as they are) placed inside OpenCode's own tool loop, where the model decides when to call it. The hidden tests
are never on disk in the agent's copy, so the tool cannot show them; the runner judges the run itself, as for every arm.

OpenCode starts the server as a child of its own process tree (MCP `type: local`), so the runner's time limit kills it with the agent.
MCP over stdio is newline-delimited JSON-RPC 2.0: `initialize`, `notifications/initialized`, `tools/list`, `tools/call`, `ping`.

    python testtool_mcp.py --repo DIR --test-cmd '["python", "-m", "unittest"]' --lang py [--log calls.jsonl]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "orchestrator"))
import orchestrate  # noqa: E402

SERVER_NAME = "forge"
TOOL_NAME = "run_tests"
SERVER_VERSION = "1"
DEFAULT_PROTOCOL = "2024-11-05"
DESCRIPTION = ("Run the repository's visible test suite and get a short answer: either `ALL TESTS PASS`, or only the FIRST failing test "
               "with its trimmed error output (the other failures are not shown). Call it after every edit instead of running the test "
               "command yourself, fix the failure it shows, then call it again.")
TOOL = {"name": TOOL_NAME, "description": DESCRIPTION, "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False}}


def render(res: "orchestrate.TestResult") -> str:
    """The text the model sees: pass, or the counts and the first failing block only."""
    if res.ok:
        return f"ALL TESTS PASS ({res.total} tests)."
    blocks = res.blocks or [("run", (res.raw or "")[-1500:])]
    head, body = blocks[0]
    if res.total is None:
        counts = "The tests did not run to the end."
    else:
        failed = res.failed or len(blocks)
        counts = f"{res.passed} of {res.total} tests pass, {failed} fail."
    more = ""
    if len(blocks) > 1:
        more = f"\n({len(blocks) - 1} more failing tests are not shown: fix this one, then run the tests again.)"
    shown = body if body.startswith(head) else f"{head}\n{body}"
    return f"{counts} First failing test:\n{shown}{more}"


class Server:
    def __init__(self, repo: Path, cmd: list[str], lang: str, log: Path | None = None, timeout: float = 120.0):
        self.repo, self.cmd, self.lang, self.log, self.timeout = repo, cmd, lang, log, timeout
        self.calls = 0

    def call_tool(self) -> str:
        t0 = time.monotonic()
        res = orchestrate.run_tests(self.repo, self.cmd, self.lang, self.timeout)
        text = render(res)
        self.calls += 1
        if self.log is not None:
            row = {"t": round(time.time(), 1), "n": self.calls, "rc": res.rc, "total": res.total, "failed": res.failed, "ok": res.ok,
                   "secs": round(time.monotonic() - t0, 2), "chars": len(text)}
            with self.log.open("a", encoding="utf-8", newline="\n") as f:
                f.write(json.dumps(row) + "\n")
        return text

    def handle(self, msg: dict[str, Any]) -> dict[str, Any] | None:
        """One JSON-RPC message in, the reply (or None for a notification) out."""
        method, mid = msg.get("method"), msg.get("id")
        if mid is None:                                   # notifications (initialized, cancelled, ...) get no reply
            return None

        def ok(result: dict[str, Any]) -> dict[str, Any]:
            return {"jsonrpc": "2.0", "id": mid, "result": result}

        def err(code: int, text: str) -> dict[str, Any]:
            return {"jsonrpc": "2.0", "id": mid, "error": {"code": code, "message": text}}
        if method == "initialize":
            wanted = (msg.get("params") or {}).get("protocolVersion") or DEFAULT_PROTOCOL
            return ok({"protocolVersion": wanted, "capabilities": {"tools": {}}, "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION}})
        if method == "ping":
            return ok({})
        if method == "tools/list":
            return ok({"tools": [TOOL]})
        if method == "tools/call":
            if (msg.get("params") or {}).get("name") != TOOL_NAME:
                return err(-32602, f"unknown tool {(msg.get('params') or {}).get('name')!r}")
            try:
                return ok({"content": [{"type": "text", "text": self.call_tool()}], "isError": False})
            except Exception as e:                         # a crashing tool must not kill the server: tell the model instead
                return ok({"content": [{"type": "text", "text": f"the test tool failed: {type(e).__name__}: {e}"[:400]}], "isError": True})
        return err(-32601, f"method not found: {method}")


def serve(server: Server, stdin: Any, stdout: Any) -> None:
    for raw in stdin:
        line = raw.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except ValueError:
            stdout.write(json.dumps({"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "parse error"}}).encode("utf-8") + b"\n")
            stdout.flush()
            continue
        if isinstance(msg, dict):
            reply = server.handle(msg)
            if reply is not None:
                stdout.write(json.dumps(reply).encode("utf-8") + b"\n")      # bytes: Windows text mode would write CRLF
                stdout.flush()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--repo", required=True, type=Path)
    ap.add_argument("--test-cmd", required=True, help="JSON list: the argv that runs the visible tests")
    ap.add_argument("--lang", required=True, choices=["py", "js"])
    ap.add_argument("--log", type=Path, default=None, help="one JSON line per call is appended here")
    ap.add_argument("--timeout", type=float, default=120.0)
    a = ap.parse_args(argv)
    serve(Server(a.repo, json.loads(a.test_cmd), a.lang, a.log, a.timeout), sys.stdin.buffer, sys.stdout.buffer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
