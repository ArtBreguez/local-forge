"""A stand-in for `opencode run --format json` in tests: reads its config like OpenCode does, starts the configured MCP server,
makes the tool calls a script says, and prints OpenCode-style JSON events with a final ANSWER line."""
import json
import os
import subprocess
import sys
from pathlib import Path

cfg = json.loads((Path(os.environ["XDG_CONFIG_HOME"]) / "opencode" / "opencode.json").read_text(encoding="utf-8"))
server = cfg["mcp"]["servers"]["rea"]
assert server["type"] == "local" and server["codemode"] is False
assert cfg["permission"]["bash"] == "deny"
proc = subprocess.Popen(server["command"], stdin=subprocess.PIPE, stdout=subprocess.PIPE)


def rpc(method, params=None, mid=1):
    proc.stdin.write((json.dumps({"jsonrpc": "2.0", "id": mid, "method": method, "params": params or {}}) + "\n").encode())
    proc.stdin.flush()
    return json.loads(proc.stdout.readline())


rpc("initialize")
listed = rpc("tools/list", mid=2)["result"]["tools"]
assert len(listed) == 5
events = []
script = json.loads(os.environ["FAKE_OPENCODE_SCRIPT"])
for i, (tool, args) in enumerate(script["calls"], 3):
    res = rpc("tools/call", {"name": tool, "arguments": args}, mid=i)["result"]
    events.append({"type": "tool_use", "part": {"type": "tool", "tool": "rea_" + tool, "state": {"status": "error" if res["isError"] else "completed"}}})
for tool in script.get("unknown", []):
    events.append({"type": "tool_use", "part": {"type": "tool", "tool": tool, "state": {"status": "error"}}})
events.append({"type": "text", "part": {"type": "text", "text": script["final"]}})
proc.stdin.close()
proc.wait(timeout=20)
for e in events:
    print(json.dumps(e), flush=True)
