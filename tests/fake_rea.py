"""A stand-in for the `rea` CLI in tests: prints the shapes the real REA printed on the host (fixtures in results/rea/)."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIX = ROOT / "results" / "rea"
args = sys.argv[1:]
cmd = args[0] if args else ""
if cmd == "decompile":
    proc = args[args.index("--procedure") + 1]
    if proc == "nope":
        sys.stderr.write("procedure not found: nope\n")
        sys.exit(2)
    print(json.dumps({"evidence_id": "ev_x", "normalized_result": f"undefined4 {proc}(uint param_1)\n{{\n  return param_1 ^ 0xedb88320;\n}}\n"}))
elif cmd == "decompile-all":
    out = Path(args[args.index("--output") + 1])
    out.mkdir(parents=True)
    for name in ("rb_a", "rb_login"):
        (out / f"0040{ord(name[-1]):x}_{name}.c").write_text(f"void {name}(void)\n{{\n  strcmp(a, b);\n}}\n", encoding="utf-8")
    (out / "manifest.json").write_text("{}", encoding="utf-8")
    print(json.dumps({"normalized_result": {"functions": 2}}))
elif cmd == "search":
    print(json.dumps({"normalized_result": {"matches": [{"name": "rb_login", "address": "0x401000"}], "kind": args[args.index("--kind") + 1]}}))
elif cmd == "inspect-pe-imports":
    sys.stdout.buffer.write((FIX / "peimports.json").read_bytes())
elif cmd == "inspect-model-artifact":
    sys.stdout.buffer.write((FIX / "gguf.json").read_bytes())
else:
    sys.stderr.write(f"unknown command {cmd}\n")
    sys.exit(1)
