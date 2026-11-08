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
     "    for ngl in range(m.layer_slots + 2):\n        c = Config(**{**asdict(base), \"ngl\": ngl})\n        if estimate(m, c).vram <= budget:",
     "    for ngl in range(m.layer_slots + 2):\n        c = Config(**{**asdict(base), \"ngl\": ngl})\n        if estimate(m, c).vram <= free_vram_mib * 1.5:",
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
     "        e.vram_compute = compute_mib(m, cfg)",
     "        e.vram_compute = 0.0",
     "graph scratch not charged to VRAM"),
    ("offload-first-layers-not-last", "forge/plan.py",
     "    return list(range(start, m.trunk_layers))",
     "    return list(range(0, max(0, m.trunk_layers - start)))",
     "offloads the first blocks instead of the last ones"),
    ("serve-ignores-vram-shortage", "forge/serve.py",
     "        if est.vram > planmod.vram_budget(free):",
     "        if False:",
     "serve keeps a configuration whose VRAM estimate exceeds what is free now"),
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
    ("loaded-setting-keeps-descending", "forge/autotune.py",
     "        if (fa, kv, ub) in settled:",
     "        if False:",
     "keeps measuring lower layer counts after a setting already ran"),
    ("floor-ignored", "forge/autotune.py",
     "    meeting = [p for p in ok if p.ppd >= pp_floor]",
     "    meeting = ok",
     "prompt-speed floor not applied"),
    # --- compute buffer / placement (the server-plan bug) ----------------------------------------
    ("compute-ignores-context", "forge/plan.py",
     "    n_kv = -(-cfg.ctx // KV_PAD) * KV_PAD",
     "    n_kv = 8704",
     "compute buffer sized for llama-bench's context, whatever the server context is (the original bug)"),
    ("compute-ignores-ubatch", "forge/plan.py",
     "    ub = min(cfg.ubatch, cfg.ctx)",
     "    ub = min(UBATCH, cfg.ctx)",
     "compute buffer always sized for ubatch 512"),
    ("flash-attention-still-pays-for-scores", "forge/plan.py",
     "    if cfg.fa or not m.attn_layers:",
     "    if not m.attn_layers:",
     "attention-score matrix charged even with flash attention on"),
    ("output-matrix-not-offloaded-first", "forge/plan.py",
     "    out_on_gpu = cfg.ngl > 0",
     "    out_on_gpu = cfg.ngl > m.trunk_layers",
     "output matrix (796 MiB) charged to the GPU only when every block is there"),
    ("mtp-slot-not-counted-as-a-layer", "forge/plan.py",
     "    start = max(0, m.layer_slots + 1 - ngl)",
     "    start = max(0, m.trunk_layers + 1 - ngl)",
     "-ngl N puts N-1 blocks on the GPU instead of N-2 (the empty MTP slot is ignored)"),
    ("reserve-too-small", "forge/plan.py",
     "VRAM_MARGIN_MIB = 152 ",
     "VRAM_MARGIN_MIB = 100 ",
     "reserve outside the window the six load/fail facts allow"),
    # --- ubatch / speculative decoding -----------------------------------------------------------
    ("server-ignores-ubatch", "forge/plan.py",
     '"-ub", str(cfg.ubatch)',
     '"-ub", str(UBATCH)',
     "llama-server always gets -ub 512, whatever was planned or tuned"),
    ("speculation-on-by-default", "forge/plan.py",
     'a += ["--spec-type", cfg.spec_type or "none"]',
     'a += ["--spec-type", cfg.spec_type or "ngram-mod"]',
     "ngram-mod switched on unless the user asked for something else"),
    ("tuner-has-no-ubatch-axis", "forge/autotune.py",
     "    ubs = list(dict.fromkeys([base.ubatch, *(u for u in ubatches if 0 < u <= planmod.COMPUTE_MAX_UBATCH)])) if gpu_plan else [base.ubatch]",
     "    ubs = [base.ubatch]",
     "tune only measures the planned ubatch"),
    ("tuner-measures-layers-the-server-cannot-hold", "forge/autotune.py",
     "                centers[(fa, kv, ub)] = min(planmod.max_layers(model, replace(base, fa=fa, kv=kv, ubatch=ub), free_vram), top)",
     "                centers[(fa, kv, ub)] = min(planmod.max_layers(model, replace(base, fa=fa, kv=kv, ubatch=ub), free_vram) + 2, top)",
     "tune measures 2 layers above the server's limit (what llama-bench can load at a smaller context)"),
    ("cache-variant-ignores-search-version", "forge/autotune.py",
     ";ub={'/'.join(map(str, ubs))};search={SEARCH_VERSION}\"",
     "\"",
     "results of an older search space are reused"),
    # --- serve: re-check and fallback --------------------------------------------------------------
    ("serve-trusts-the-tuned-layer-count", "forge/serve.py",
     "        ok, detail = fits_now(model, cfg, profile)\n        if ok:\n            return Selection(name, cfg, \"tuned\"",
     "        ok, detail = True, \"\"\n        if ok:\n            return Selection(name, cfg, \"tuned\"",
     "a tune measured by llama-bench is served without re-checking it against the server's own memory estimate"),
    ("replan-drops-the-tuned-ubatch", "forge/serve.py",
     "                                       ubatch=min(tuned.config.ubatch, planmod.COMPUTE_MAX_UBATCH), enable_mtp",
     "                                       ubatch=planmod.UBATCH, enable_mtp",
     "re-planning after a failed re-check forgets the tuned ubatch"),
    ("fallback-jumps-four-layers", "forge/serve.py",
     "        if fine < planmod.FALLBACK_FINE_STEPS:",
     "        if False:",
     "fallback ladder skips straight to steps of 4 layers (15 -> 11)"),
    ("pinned-device-checked-against-the-roomiest-card", "forge/serve.py",
     "        if cfg.device:       # a config pinned to one card",
     "        if False:       # a config pinned to one card",
     "a config pinned to one GPU is judged by another GPU's free VRAM"),
    ("ladder-keeps-duplicate-rungs", "forge/serve.py",
     "        if _placement(model, n) in seen:",
     "        if False:",
     "fallback ladder retries a placement it has already tried"),
    ("mtp-block-not-charged", "forge/plan.py",
     "    if m.mtp_bytes:     # the MTP slot",
     "    if False:     # the MTP slot",
     "weights of a present MTP block are not charged anywhere"),
    ("fallback-never-reaches-smaller-contexts", "forge/serve.py",
     "    while ctx > planmod.CTX_MIN:",
     "    while False:",
     "fallback ladder has no smaller-context rungs"),
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
