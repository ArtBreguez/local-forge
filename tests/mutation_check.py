"""Mutation check: break one rule at a time in a temporary copy of the repo and require the tests to fail.

    python tests/mutation_check.py [--keep]

The working tree is never modified. Every mutant is `(id, file, old, new, what it breaks)`; `old` must occur
exactly once in the file, otherwise the script aborts (so a mutant that silently did not apply can never
count as "caught"). The baseline (no mutation) must pass first. Exit status is 0 only if every mutant is killed.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

MUTANTS = [
    # --- VRAM-fit rule -------------------------------------------------------------------------
    ("vram-budget-1.5x-free", "forge/plan.py",
     "    for ngl in range(m.trunk_layers + 2):\n        c = Config(**{**asdict(base), \"ngl\": ngl})\n        if estimate(m, c).vram <= budget:",
     "    for ngl in range(m.trunk_layers + 2):\n        c = Config(**{**asdict(base), \"ngl\": ngl})\n        if estimate(m, c).vram <= free_vram_mib * 1.5:",
     "layers are chosen against 150% of the free VRAM"),
    ("vram-comparison-inverted", "forge/plan.py",
     "        if estimate(m, c).vram <= budget:\n            best = ngl",
     "        if estimate(m, c).vram >= budget:\n            best = ngl",
     "the fit comparison is inverted (picks layers that do NOT fit)"),
    ("vram-margin-removed", "forge/plan.py",
     "    return free_mib * (1 - VRAM_MARGIN_FRACTION) - VRAM_MARGIN_MIB",
     "    return free_mib",
     "no safety margin: plans to the last MiB of free VRAM"),
    ("estimate-ignores-kv-cache", "forge/plan.py",
     "            e.vram_kv += kv_tok[i] / MIB",
     "            e.vram_kv += 0",
     "KV cache not charged to VRAM"),
    ("estimate-ignores-scratch", "forge/plan.py",
     "        e.vram_compute = COMPUTE_BASE_MIB",
     "        e.vram_compute = 0.0",
     "graph scratch not charged to VRAM"),
    ("offload-first-layers-not-last", "forge/plan.py",
     "    return list(range(m.trunk_layers - n, m.trunk_layers))",
     "    return list(range(0, n))",
     "offloads blocks 0..n instead of the last n"),
    ("serve-ignores-vram-drop", "forge/serve.py",
     "            if free < floor:",
     "            if False:",
     "serve keeps the tuned config even when free VRAM dropped"),
    # --- cache key -----------------------------------------------------------------------------
    ("cache-key-drops-model", "forge/autotune.py",
     'raw = "\\x1f".join([hw_fingerprint, model_fingerprint, llama_build, variant])',
     'raw = "\\x1f".join([hw_fingerprint, llama_build, variant])',
     "cache key ignores the model file"),
    ("cache-key-drops-hardware", "forge/autotune.py",
     'raw = "\\x1f".join([hw_fingerprint, model_fingerprint, llama_build, variant])',
     'raw = "\\x1f".join([model_fingerprint, llama_build, variant])',
     "cache key ignores the hardware"),
    ("cache-key-drops-llama-build", "forge/autotune.py",
     'raw = "\\x1f".join([hw_fingerprint, model_fingerprint, llama_build, variant])',
     'raw = "\\x1f".join([hw_fingerprint, model_fingerprint, variant])',
     "cache key ignores the llama.cpp build"),
    ("hardware-fingerprint-ignores-gpu", "forge/probe.py",
     '        "gpus": [[g.get("backend"), g.get("name"), g.get("total_mib")] for g in prof.get("gpus") or []],',
     '        "gpus": [],',
     "hardware fingerprint does not include the GPU"),
    ("cache-never-hits", "forge/autotune.py",
     "        hit = cache.load(key)\n        if hit:",
     "        hit = None\n        if hit:",
     "cache is written but never read (every start re-tunes)"),
    # --- autotune search -----------------------------------------------------------------------
    ("oom-does-not-stop-the-search", "forge/autotune.py",
     "        if (fa, kv) in oom_floor and ngl >= oom_floor[(fa, kv)]:",
     "        if False:",
     "keeps probing larger layer counts after an out-of-memory failure"),
    ("floor-ignored", "forge/autotune.py",
     "    meeting = [p for p in ok if p.ppd >= pp_floor]",
     "    meeting = ok",
     "prompt-speed floor not applied"),
    # --- probe parsing -------------------------------------------------------------------------
    ("list-devices-swaps-total-and-free", "forge/probe.py",
     '"total_mib": int(total), "free_mib": int(free) if free is not None else None,',
     '"total_mib": int(free or total), "free_mib": int(total),',
     "free and total VRAM swapped"),
]


# Mutants in forge/plan.py are run against tests/test_plan.py alone: killing them must come from the arithmetic
# tests, not from some unrelated test that happens to break because the planner changed.
SCOPES = {"forge/plan.py": ["tests.test_plan"]}


def run_tests(cwd: Path, scope: list[str] | None = None) -> tuple[int, str]:
    cmd = ([sys.executable, "-m", "unittest", "-f", *scope] if scope
           else [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-t", ".", "-f"])
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=900)
    return r.returncode, r.stdout + r.stderr


def failing(output: str) -> str:
    m = re.search(r"^(FAIL|ERROR): (\S+) \((\S+)\)", output, re.M)
    return f"{m.group(3).split('.')[-2]}.{m.group(2)}" if m else "?"


def main() -> int:
    keep = "--keep" in sys.argv
    work = Path(tempfile.mkdtemp(prefix="forge-mut-"))
    try:
        def fresh() -> Path:
            dst = work / "tree"
            if dst.exists():
                shutil.rmtree(dst)
            dst.mkdir()
            for part in ("forge", "tests", "docs", "results"):
                shutil.copytree(ROOT / part, dst / part, ignore=shutil.ignore_patterns("__pycache__"))
            return dst
        rc, out = run_tests(fresh())
        print(f"baseline: {'PASS' if rc == 0 else 'FAIL'}")
        if rc != 0:
            print(out[-3000:])
            return 2
        survived = []
        for mid, rel, old, new, what in MUTANTS:
            tree = fresh()
            path = tree / rel
            src = path.read_text(encoding="utf-8")
            if src.count(old) != 1:
                print(f"ABORT: mutant {mid}: pattern occurs {src.count(old)} times in {rel} (must be exactly 1)")
                return 3
            path.write_text(src.replace(old, new), encoding="utf-8")
            rc, out = run_tests(tree, SCOPES.get(rel))
            status = "KILLED " if rc != 0 else "SURVIVED"
            print(f"{status} {mid:<36} [{'plan tests' if rel in SCOPES else 'full suite'}] {what}" + (f"   <- first failing test: {failing(out)}" if rc != 0 else ""))
            if rc == 0:
                survived.append(mid)
        print(f"\n{len(MUTANTS) - len(survived)}/{len(MUTANTS)} mutants killed" + (f"; survived: {', '.join(survived)}" if survived else ""))
        return 1 if survived else 0
    finally:
        if keep:
            print("kept", work)
        else:
            shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
