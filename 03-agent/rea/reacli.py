"""REA from Python (phase 6, issue #23 part C): run the `rea` CLI, keep the useful part of its JSON envelope, and turn each command's
result into a short text a 16K-context model can read. Stdlib only; Windows and Linux.

REA prints a large evidence envelope (`evidence_id`, `analysis_profile`, ...); the answer to a question is in `normalized_result`.
`view_*` functions cut that down (the imports as `DLL: fn, fn`, the GGUF metadata as `key = value`, the pseudocode as it is) and every
view ends at `MAX_CHARS`, so a tool result never fills the context. Nothing here interprets the result: the model does that.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MAX_CHARS = 6000
MAX_CHARS_ALL = 9000
DEFAULT_ROOT = Path(r"C:\Users\arthu\llm\rea")
IS_WINDOWS = os.name == "nt"


@dataclass
class ReaResult:
    ok: bool
    text: str               # what a tool returns to the model: the view, or the error message
    secs: float
    rc: int | None
    command: list[str]


def default_cmd(root: Path = DEFAULT_ROOT) -> list[str]:
    return [str(root / "app" / "node_modules" / ".bin" / ("rea.cmd" if IS_WINDOWS else "rea"))]


def default_env(root: Path = DEFAULT_ROOT) -> dict[str, str]:
    """Environment REA needs on the host: the portable Node, JDK 25 and Ghidra 12.1.4 under `root` (nothing is global there)."""
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    jdks = sorted(root.glob("jdk-*"))
    ghidras = sorted(root.glob("ghidra_*_PUBLIC"))
    nodes = sorted(root.glob("node-v*"))
    if jdks:
        env["JAVA_HOME"] = str(jdks[-1])
    if ghidras:
        env["GHIDRA_INSTALL_DIR"] = str(ghidras[-1])
    extra = [str(nodes[-1])] if nodes else []
    if jdks:
        extra.append(str(jdks[-1] / "bin"))
    env["PATH"] = os.pathsep.join(extra + [env.get("PATH", "")])
    env.setdefault("JAVA_TOOL_OPTIONS", "-Xmx1536m")           # the host has 16 GB shared with the VM and the model server
    return env


def _json_from(text: str) -> Any:
    text = text.lstrip("\ufeff \r\n\t")
    start = text.find("{")
    if start < 0:
        raise ValueError("no JSON object in the output")
    obj, _ = json.JSONDecoder().raw_decode(text[start:])
    return obj


def normalized(stdout: str) -> Any:
    """The `normalized_result` of REA's JSON output (a string for `decompile`, an object for the inspect commands)."""
    d = _json_from(stdout)
    if isinstance(d, dict) and "normalized_result" in d:
        return d["normalized_result"]
    return d


def _cap(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + f"\n... [cut: {len(text) - limit} more characters]"


def _compact(obj: Any, limit: int) -> str:
    return _cap(json.dumps(obj, ensure_ascii=False, separators=(",", ":")), limit)


def view_decompile(nr: Any) -> str:
    if isinstance(nr, str):
        return _cap(nr.strip(), MAX_CHARS)
    if isinstance(nr, dict):
        for k in ("pseudocode", "code", "text", "decompilation", "source"):
            if isinstance(nr.get(k), str):
                return _cap(nr[k].strip(), MAX_CHARS)
    return _compact(nr, MAX_CHARS)


def view_imports(nr: Any) -> str:
    """`DLL: function, function, ...` per imported DLL, plus the machine and the import directory location."""
    if not isinstance(nr, dict) or "imports" not in nr:
        return _compact(nr, MAX_CHARS)
    lines = [f"format {nr.get('format')}, machine {nr.get('machine')} ({hex(nr['machine']) if isinstance(nr.get('machine'), int) else '?'}), {nr.get('bits')}-bit"]
    d = nr.get("import_directory") or {}
    if d:
        lines.append(f"import directory: rva {d.get('rva')} ({hex(d['rva']) if isinstance(d.get('rva'), int) else '?'}), size {d.get('size')}")
    for dll in nr["imports"]:
        names = [f.get("name") or f"#{f.get('ordinal')}" for f in dll.get("functions", [])]
        lines.append(f"{dll.get('name')}: {', '.join(names)}")
    return _cap("\n".join(lines), MAX_CHARS)


def view_model(nr: Any) -> str:
    """GGUF header: the scalar metadata as `key = value`, array metadata as its length, and the tensor totals."""
    if not isinstance(nr, dict) or "metadata" not in nr:
        return _compact(nr, MAX_CHARS)
    lines = [f"format {nr.get('format')} version {nr.get('version')}"]
    for m in nr["metadata"]:
        if m.get("element_count") is not None:
            lines.append(f"{m.get('key')} = <{m.get('type')} array, {m.get('element_count')} elements>")
        else:
            lines.append(f"{m.get('key')} = {m.get('value')}")
    totals = nr.get("totals") or {}
    lines.append(f"tensors: {len(nr.get('tensors') or [])}, parameters: {totals.get('parameter_count')}")
    return _cap("\n".join(lines), MAX_CHARS)


def view_generic(nr: Any) -> str:
    return _cap(nr, MAX_CHARS) if isinstance(nr, str) else _compact(nr, MAX_CHARS)


class Rea:
    def __init__(self, cmd: list[str] | None = None, env: dict[str, str] | None = None, timeout: float = 600.0, root: Path = DEFAULT_ROOT):
        self.cmd = cmd or default_cmd(root)
        self.env = env if env is not None else default_env(root)
        self.timeout = timeout

    def raw(self, args: list[str]) -> tuple[int | None, str, str, float]:
        t0 = time.monotonic()
        try:
            r = subprocess.run([*self.cmd, *args], capture_output=True, text=True, encoding="utf-8", errors="replace", env=self.env,
                               stdin=subprocess.DEVNULL, timeout=self.timeout)
            return r.returncode, r.stdout, r.stderr, time.monotonic() - t0
        except subprocess.TimeoutExpired:
            return None, "", f"rea did not finish in {self.timeout:g} s", time.monotonic() - t0
        except OSError as e:
            return -1, "", f"could not start rea: {e}", time.monotonic() - t0

    def _run(self, args: list[str], view: Any) -> ReaResult:
        rc, out, err, secs = self.raw(args)
        cmd = [*self.cmd, *args]
        if rc != 0:
            msg = (err.strip() or out.strip() or f"rea exited with {rc}")[-600:]
            return ReaResult(False, f"REA failed: {msg}", secs, rc, cmd)
        try:
            return ReaResult(True, view(normalized(out)), secs, rc, cmd)
        except ValueError as e:
            return ReaResult(False, f"REA printed no JSON result ({e}): {out.strip()[:300]}", secs, rc, cmd)

    # one method per REA command used by the benchmark -----------------------------------------------------------------
    def decompile(self, binary: str, procedure: str) -> ReaResult:
        return self._run(["decompile", "--procedure", procedure, binary, "--format", "json", "--provider", "ghidra"], view_decompile)

    def decompile_all(self, binary: str, name: str = "") -> ReaResult:
        """Pseudocode of every function (or those whose name contains `name`), one file per function written by REA, read back here."""
        tmp = Path(tempfile.mkdtemp(prefix="rea-all-"))
        out_dir = tmp / "out"
        try:
            args = ["decompile-all", binary, "--output", str(out_dir), "--format", "json", "--provider", "ghidra"]
            if name:
                args += ["--name", name]
            rc, out, err, secs = self.raw(args)
            cmd = [*self.cmd, *args]
            if rc != 0:
                return ReaResult(False, f"REA failed: {(err.strip() or out.strip() or f'rea exited with {rc}')[-600:]}", secs, rc, cmd)
            parts = []
            for f in sorted(p for p in out_dir.rglob("*") if p.is_file() and p.suffix.lower() in (".c", ".txt", ".cpp", ".h", "")):
                if f.name.lower().startswith("manifest"):
                    continue
                parts.append(f"// ---- {f.name}\n{f.read_text(encoding='utf-8', errors='replace').strip()}")
            if not parts:
                return ReaResult(False, "REA wrote no function files: " + _cap(out.strip(), 300), secs, rc, cmd)
            return ReaResult(True, _cap("\n\n".join(parts), MAX_CHARS_ALL), secs, rc, cmd)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def search(self, binary: str, pattern: str, kind: str = "procedures") -> ReaResult:
        return self._run(["search", binary, pattern, "--kind", kind, "--format", "json", "--provider", "ghidra"], view_generic)

    def imports(self, binary: str) -> ReaResult:
        return self._run(["inspect-pe-imports", binary, "--format", "json"], view_imports)

    def model_header(self, path: str) -> ReaResult:
        return self._run(["inspect-model-artifact", path, "--format", "json"], view_model)


_ANSWER = re.compile(r"(?im)^[ \t>*_`#-]*answer[ \t*_`]*:[ \t*_`]*(.+?)[ \t*_`]*$")


def final_answer(text: str) -> str | None:
    """The text after the LAST `ANSWER:` line of a reply, or None."""
    found = _ANSWER.findall(text or "")
    return found[-1].strip() if found else None
