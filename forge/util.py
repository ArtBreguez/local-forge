"""Small helpers shared by the forge modules (stdlib only)."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

IS_WINDOWS = sys.platform.startswith("win")
EXE = ".exe" if IS_WINDOWS else ""


def run_cmd(args: list[str], timeout: float = 20.0, env: dict[str, str] | None = None) -> tuple[int, str] | None:
    """Run a command, return (returncode, stdout+stderr) or None if it cannot run at all.

    None covers a missing executable, a permission error and a timeout, so callers can
    treat "tool not available" uniformly and never crash on it.
    """
    try:
        full_env = None if env is None else {**os.environ, **env}
        p = subprocess.run(args, capture_output=True, timeout=timeout, env=full_env,
                           stdin=subprocess.DEVNULL)
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
    text = (p.stdout or b"").decode("utf-8", "replace") + (p.stderr or b"").decode("utf-8", "replace")
    return p.returncode, text


def find_tool(name: str, llama_dir: str | os.PathLike | None = None) -> str | None:
    """Locate a llama.cpp executable: explicit dir (and one level of subdirs), $FORGE_LLAMA_DIR, PATH."""
    dirs: list[Path] = []
    for d in (llama_dir, os.environ.get("FORGE_LLAMA_DIR")):
        if d:
            dirs.append(Path(d))
    for d in dirs:
        for cand in (d / (name + EXE), d / "bin" / (name + EXE)):
            if cand.is_file():
                return str(cand)
        try:
            for sub in sorted(p for p in d.iterdir() if p.is_dir()):
                cand = sub / (name + EXE)
                if cand.is_file():
                    return str(cand)
        except OSError:
            pass
    return shutil.which(name)


def default_cache_dir() -> Path:
    """Where tuned results and server logs live: $FORGE_HOME, else a per-user cache dir."""
    if os.environ.get("FORGE_HOME"):
        return Path(os.environ["FORGE_HOME"])
    if IS_WINDOWS:
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return Path(base) / "forge"
    return Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "forge"
