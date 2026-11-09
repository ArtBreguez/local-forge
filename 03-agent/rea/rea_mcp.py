#!/usr/bin/env python3
"""MCP server that gives OpenCode five REA tools (phase 6 REA benchmark, arm O). Stdlib only; Windows and Linux.

REA's own MCP server publishes its whole catalogue (over a hundred tools: Android, browsers, firmware, ...): the schemas alone would
not fit a 16K context. This server publishes the five commands the tasks can use and runs the real `rea` CLI for each; the MODEL
chooses the tool, the arguments and the order (the harness does not), so what is measured is how well the model drives REA.

Every call is validated (required arguments, enum values, the file exists) and logged as one JSON line:
`valid` false = the model gave a bad call (missing argument, unknown file or enum value); `ok` false with `valid` true = REA itself failed.
MCP over stdio is newline-delimited JSON-RPC 2.0: `initialize`, `notifications/initialized`, `tools/list`, `tools/call`, `ping`.

    python rea_mcp.py --log calls.jsonl [--rea-root C:\\Users\\arthu\\llm\\rea] [--rea-cmd '["python", "fake_rea.py"]']
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import reacli  # noqa: E402

SERVER_NAME = "rea"
SERVER_VERSION = "1"
DEFAULT_PROTOCOL = "2024-11-05"


def _tool(name: str, description: str, props: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {"name": name, "description": description,
            "inputSchema": {"type": "object", "properties": props, "required": required, "additionalProperties": False}}


_BIN = {"type": "string", "description": "path of the binary (PE executable)"}
TOOLS = [
    _tool("decompile", "Decompile ONE function of a native binary with Ghidra and return its pseudocode (C-like). Takes about half a minute.",
          {"binary": _BIN, "procedure": {"type": "string", "description": "function name, exactly as the binary has it"}}, ["binary", "procedure"]),
    _tool("decompile_all", "Decompile every function of a native binary (optionally only those whose name contains a substring) and return all the pseudocode at once.",
          {"binary": _BIN, "name_contains": {"type": "string", "description": "case-sensitive substring of the function name; empty means all functions"}}, ["binary"]),
    _tool("search", "Search the strings or the function names of a native binary for a text; use kind procedures to list functions.",
          {"binary": _BIN, "pattern": {"type": "string", "description": "literal text to look for"},
           "kind": {"type": "string", "enum": ["strings", "procedures"], "description": "what to search: strings or procedures (function names)"}}, ["binary", "pattern", "kind"]),
    _tool("imports", "List the DLLs and functions a Windows PE binary imports, with its machine type and the location of its import directory (a header field).",
          {"binary": _BIN}, ["binary"]),
    _tool("model_header", "Read the header of a GGUF model file (no weights are loaded): every metadata key and value, and the tensor count.",
          {"path": {"type": "string", "description": "path of the .gguf file"}}, ["path"]),
]
SPEC = {t["name"]: t["inputSchema"] for t in TOOLS}


def validate(name: str, args: Any) -> str | None:
    """None when the call is well formed, else the message the model gets."""
    if name not in SPEC:
        return f"unknown tool {name!r}; the tools are: {', '.join(SPEC)}"
    if not isinstance(args, dict):
        return "the arguments must be a JSON object"
    props = SPEC[name]["properties"]
    for key in SPEC[name]["required"]:
        if key not in args or args[key] in (None, ""):
            return f"missing required argument {key!r}"
    for key, val in args.items():
        if key not in props:
            return f"unknown argument {key!r}; the arguments are: {', '.join(props)}"
        if not isinstance(val, str):
            return f"argument {key!r} must be a string"
        if "enum" in props[key] and val not in props[key]["enum"]:
            return f"argument {key!r} must be one of {props[key]['enum']}"
    path = args.get("binary") or args.get("path")
    if path and not Path(path).is_file():
        return f"no such file: {path}"
    return None


class Server:
    def __init__(self, rea: reacli.Rea, log: Path | None = None):
        self.rea, self.log = rea, log
        self.calls = 0

    def run_tool(self, name: str, a: dict[str, Any]) -> reacli.ReaResult:
        if name == "decompile":
            return self.rea.decompile(a["binary"], a["procedure"])
        if name == "decompile_all":
            return self.rea.decompile_all(a["binary"], a.get("name_contains", ""))
        if name == "search":
            return self.rea.search(a["binary"], a["pattern"], a["kind"])
        if name == "imports":
            return self.rea.imports(a["binary"])
        return self.rea.model_header(a["path"])

    def call_tool(self, name: str, args: Any) -> tuple[str, bool]:
        """(text for the model, isError)."""
        t0 = time.monotonic()
        self.calls += 1
        bad = validate(name, args)
        if bad:
            text, is_err, valid, ok, rc = f"invalid call: {bad}", True, False, False, None
        else:
            res = self.run_tool(name, args)
            text, is_err, valid, ok, rc = res.text, not res.ok, True, res.ok, res.rc
        if self.log is not None:
            row = {"t": round(time.time(), 1), "n": self.calls, "tool": name, "args": args if isinstance(args, dict) else str(args)[:200],
                   "valid": valid, "ok": ok, "rc": rc, "secs": round(time.monotonic() - t0, 2), "chars": len(text)}
            with self.log.open("a", encoding="utf-8", newline="\n") as f:
                f.write(json.dumps(row) + "\n")
        return text, is_err

    def handle(self, msg: dict[str, Any]) -> dict[str, Any] | None:
        method, mid = msg.get("method"), msg.get("id")
        if mid is None:
            return None

        def ok(result: dict[str, Any]) -> dict[str, Any]:
            return {"jsonrpc": "2.0", "id": mid, "result": result}
        if method == "initialize":
            wanted = (msg.get("params") or {}).get("protocolVersion") or DEFAULT_PROTOCOL
            return ok({"protocolVersion": wanted, "capabilities": {"tools": {}}, "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION}})
        if method == "ping":
            return ok({})
        if method == "tools/list":
            return ok({"tools": TOOLS})
        if method == "tools/call":
            p = msg.get("params") or {}
            try:
                text, is_err = self.call_tool(p.get("name"), p.get("arguments") or {})
            except Exception as e:                                  # a crashing tool must not kill the server: tell the model instead
                text, is_err = f"the tool failed: {type(e).__name__}: {e}"[:400], True
            return ok({"content": [{"type": "text", "text": text}], "isError": is_err})
        return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": f"method not found: {method}"}}


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
                stdout.write(json.dumps(reply).encode("utf-8") + b"\n")        # bytes: Windows text mode would write CRLF
                stdout.flush()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--log", type=Path, default=None, help="one JSON line per call is appended here")
    ap.add_argument("--rea-root", type=Path, default=reacli.DEFAULT_ROOT)
    ap.add_argument("--rea-cmd", default=None, help="JSON list: the argv of the rea command (default: the REA install under --rea-root)")
    ap.add_argument("--timeout", type=float, default=600.0)
    a = ap.parse_args(argv)
    rea = reacli.Rea(json.loads(a.rea_cmd) if a.rea_cmd else None, None, a.timeout, a.rea_root)
    serve(Server(rea, a.log), sys.stdin.buffer, sys.stdout.buffer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
