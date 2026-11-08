"""Refine a plan by measuring with llama-bench, and cache the answer.

Search space (each point is one separate llama-bench process, so an out-of-memory crash
loses only that point):
    GPU layers   plan.ngl + {-4, -2, 0, +2, +4}
    attention/KV (flash attn off, f16) | (on, f16) | (on, q8_0)   -- a quantised V cache needs FA
Each point is measured at depth 0 and at the agent depth (default 8192 tokens already in the cache).

Objective (default): maximise generation tok/s at agent depth, subject to prompt tok/s at agent
depth >= a floor. An agent re-reads its context every turn, so a configuration that generates
fast but prefills slowly loses on wall-clock time. If nothing meets the floor, the one with the
best prompt speed wins.

Order of evaluation: nearest to the plan first (a different attention/KV setting counts as 2 steps),
upward before downward. A failure at N layers for a setting skips every larger count for that setting.
The budget is wall-clock: when it runs out the best point so far is returned, flagged incomplete.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Callable

from . import plan as planmod
from . import util
from .plan import Caps, Config, ModelInfo, Plan

CACHE_FILE = "tune-cache.json"
DEFAULT_DEPTH = 8192
DEFAULT_PP_FLOOR = 100.0      # tok/s at agent depth: re-reading 8K tokens takes <= ~80 s
LAYER_OFFSETS = (-4, -2, 0, 2, 4)
COMBOS = ((False, "f16"), (True, "f16"), (True, "q8_0"))   # (flash attention, KV type)
BENCH_CTX_OVERHEAD = 640      # llama-bench context = depth + 512 prompt + 128 generated

_OOM_PATTERNS = re.compile(
    r"out of (device |host )?memory|OutOfDeviceMemory|OutOfHostMemory|failed to allocate|unable to allocate|"
    r"allocation of size .* failed|cudaMalloc failed|failed to create context|failed to load model", re.I)


# ---------------------------------------------------------------------------------
# parsing llama-bench output
# ---------------------------------------------------------------------------------

def parse_bench_jsonl(text: str) -> dict[tuple[int, str], tuple[float, float]]:
    """Map (depth, 'pp'|'tg') -> (mean tok/s, stddev) from llama-bench `-o jsonl` output.

    Non-JSON lines (backend chatter on the same stream) are ignored.
    """
    out: dict[tuple[int, str], tuple[float, float]] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if "avg_ts" not in r:
            continue
        kind = "pp" if r.get("n_prompt", 0) > 0 and r.get("n_gen", 0) == 0 else "tg" if r.get("n_gen", 0) > 0 and r.get("n_prompt", 0) == 0 else None
        if kind:
            out[(int(r.get("n_depth", 0)), kind)] = (float(r["avg_ts"]), float(r.get("stddev_ts") or 0.0))
    return out


def classify_failure(rc: int | None, text: str, ngl: int) -> str:
    """'oom' for a memory failure, 'timeout' if the run was killed, else 'error'.

    llama-bench reports a failed GPU allocation as 'failed to load model' (weights) or 'failed to create
    context with model' (KV/scratch) -- see tests/fixtures/llama-bench-oom-stderr.txt. Those two phrases
    are only trusted as OOM when layers are on the GPU (a CPU-only load failing that way is a bad file).
    """
    if rc is None:
        return "timeout"
    if _OOM_PATTERNS.search(text):
        specific = re.search(r"out of (device |host )?memory|OutOfDeviceMemory|OutOfHostMemory|failed to allocate|unable to allocate|"
                             r"allocation of size .* failed|cudaMalloc failed", text, re.I)
        if specific or ngl > 0:
            return "oom"
    return "error"


# ---------------------------------------------------------------------------------
# measuring
# ---------------------------------------------------------------------------------

@dataclass
class Point:
    ngl: int
    fa: bool
    kv: str
    status: str                      # ok | oom | error | timeout | skipped
    pp0: float | None = None
    tg0: float | None = None
    ppd: float | None = None
    tgd: float | None = None
    ppd_sd: float | None = None
    tgd_sd: float | None = None
    seconds: float = 0.0
    note: str = ""

    def combo(self) -> tuple[bool, str]:
        return (self.fa, self.kv)


def bench_args(bench: list[str], model_path: str, cfg: Config, depth: int, reps: int, fa_values: bool, n_prompt: int = 512, n_gen: int = 128) -> list[str]:
    a = list(bench) + ["-m", model_path, "-ngl", str(cfg.ngl), "-t", str(cfg.threads), "-ub", str(cfg.ubatch),
                       "-ctk", cfg.kv, "-ctv", cfg.kv, "-p", str(n_prompt), "-n", str(n_gen),
                       "-d", f"0,{depth}", "-r", str(reps), "-o", "jsonl"]
    a += ["-fa", ("on" if cfg.fa else "off") if fa_values else ("1" if cfg.fa else "0")]
    if cfg.device:
        a += ["-dev", cfg.device]
    return a


@dataclass
class TuneResult:
    config: Config
    points: list[Point]
    best: Point | None
    objective: str
    complete: bool                    # False if the time budget ran out before the search finished
    elapsed_s: float
    notes: list[str] = field(default_factory=list)
    from_cache: bool = False
    conditions: dict[str, Any] = field(default_factory=dict)   # free memory when the measurements were taken
    plan_name: str = ""                                        # gpu | cpu | low-power: which plan's neighbourhood was measured

    def to_dict(self) -> dict[str, Any]:
        return {"config": asdict(self.config), "points": [asdict(p) for p in self.points],
                "best": asdict(self.best) if self.best else None, "objective": self.objective,
                "complete": self.complete, "elapsed_s": round(self.elapsed_s, 1), "notes": self.notes,
                "conditions": self.conditions, "plan_name": self.plan_name}

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "TuneResult":
        return TuneResult(config=Config(**d["config"]), points=[Point(**p) for p in d["points"]],
                          best=Point(**d["best"]) if d.get("best") else None, objective=d["objective"],
                          complete=d["complete"], elapsed_s=d.get("elapsed_s", 0.0), notes=d.get("notes", []),
                          conditions=d.get("conditions", {}), plan_name=d.get("plan_name", ""))


def candidate_order(center_ngl: int, center_combo: tuple[bool, str], trunk_layers: int, gpu: bool) -> list[tuple[int, bool, str]]:
    """Points to measure, nearest to the plan first (see module docstring)."""
    top = trunk_layers + 1
    seen: dict[tuple[int, bool, str], tuple[int, int]] = {}
    for fa, kv in COMBOS:
        for d in (LAYER_OFFSETS if gpu else (0,)):
            n = min(max(center_ngl + d, 0), top)
            cost = abs(n - center_ngl) + (2 if (fa, kv) != center_combo else 0)
            key = (n, fa, kv)
            if key not in seen or cost < seen[key][0]:
                seen[key] = (cost, 0 if n >= center_ngl else 1)
    return [k for k, _ in sorted(seen.items(), key=lambda kv_: (kv_[1][0], kv_[0][1:] != center_combo, kv_[1][1], kv_[0][0]))]


def pick_best(points: list[Point], pp_floor: float) -> tuple[Point | None, str]:
    ok = [p for p in points if p.status == "ok" and p.tgd is not None and p.ppd is not None]
    if not ok:
        return None, "no point was measured successfully"
    meeting = [p for p in ok if p.ppd >= pp_floor]
    if meeting:
        best = max(meeting, key=lambda p: (p.tgd, p.ppd))
        return best, f"max generation tok/s at agent depth subject to prompt tok/s >= {pp_floor:g}"
    best = max(ok, key=lambda p: p.ppd)
    return best, f"no point reached the prompt floor {pp_floor:g} tok/s: best prompt speed chosen"


def tune(profile: dict[str, Any], model: ModelInfo, plan: Plan, bench: list[str], *, depth: int = DEFAULT_DEPTH,
         reps: int = 1, budget_s: float = 1200.0, pp_floor: float = DEFAULT_PP_FLOOR, per_point_timeout: float = 900.0,
         caps: Caps | None = None, clock: Callable[[], float] = time.monotonic,
         run: Callable[..., tuple[int, str] | None] = util.run_cmd,
         progress: Callable[[str], None] = lambda s: None) -> TuneResult:
    """Measure around `plan`. `bench` is the llama-bench command prefix, e.g. ['C:/.../llama-bench.exe']."""
    caps = caps or Caps()
    base = plan.config
    start = clock()
    gpu_plan = base.ngl > 0
    top = model.trunk_layers + 1
    center = min(base.ngl, top)
    depth = max(512, min(depth, base.ctx - BENCH_CTX_OVERHEAD))
    gpu, free_vram, _ = planmod.pick_gpu(profile)
    total_vram = float(gpu["total_mib"]) if gpu and gpu.get("total_mib") else None
    points: list[Point] = []
    notes: list[str] = []
    oom_floor: dict[tuple[bool, str], int] = {}
    complete = True
    queue = candidate_order(center, (base.fa, base.kv), model.trunk_layers, gpu_plan)
    queued = set(queue)
    while queue:
        ngl, fa, kv = queue.pop(0)
        if (fa, kv) in oom_floor and ngl >= oom_floor[(fa, kv)]:
            points.append(Point(ngl, fa, kv, "skipped", note=f"a smaller layer count already failed for fa={'on' if fa else 'off'} kv={kv}"))
            continue
        cfg = replace(base, ngl=planmod.all_layers(model) if ngl >= top else ngl, fa=fa, kv=kv)
        if ngl > 0 and total_vram is not None and planmod.estimate(model, cfg).vram > total_vram:
            points.append(Point(ngl, fa, kv, "skipped", note="estimate exceeds total VRAM at the server context"))
            continue
        remaining = budget_s - (clock() - start)
        if remaining < 30:
            complete = False
            notes.append(f"time budget of {budget_s:g}s exhausted before {ngl} layers fa={'on' if fa else 'off'} kv={kv}")
            break
        t0 = clock()
        args = bench_args(bench, model.path, cfg, depth, reps, caps.fa_takes_value)
        res = run(args, timeout=min(per_point_timeout, remaining))
        secs = clock() - t0
        if res is None:
            pt = Point(ngl, fa, kv, "timeout", seconds=secs, note="llama-bench did not finish in time or could not start")
        else:
            rc, text = res
            meas = parse_bench_jsonl(text)
            if rc == 0 and (depth, "tg") in meas and (depth, "pp") in meas:
                pt = Point(ngl, fa, kv, "ok", pp0=meas.get((0, "pp"), (None, 0))[0], tg0=meas.get((0, "tg"), (None, 0))[0],
                           ppd=meas[(depth, "pp")][0], tgd=meas[(depth, "tg")][0],
                           ppd_sd=meas[(depth, "pp")][1], tgd_sd=meas[(depth, "tg")][1], seconds=secs)
            else:
                kind = classify_failure(rc, text, ngl)
                last = next((l.strip() for l in reversed(text.splitlines()) if l.strip()), "")
                pt = Point(ngl, fa, kv, kind, seconds=secs, note=f"rc={rc}: {last[:160]}")
        points.append(pt)
        if pt.status in ("oom", "error"):
            oom_floor[(fa, kv)] = min(ngl, oom_floor.get((fa, kv), ngl))
            # boundary refinement: the grid steps by 2, so the count just below a failure may never have been tried.
            # Probe it next, but only for the setting that is currently winning (the others are not worth the time).
            cur, _ = pick_best(points, pp_floor)
            below = ngl - 1
            ok_max = max((p.ngl for p in points if p.status == "ok" and p.combo() == (fa, kv)), default=-1)
            if cur is not None and cur.combo() == (fa, kv) and below > ok_max and (below, fa, kv) not in queued:
                queued.add((below, fa, kv))
                queue.insert(0, (below, fa, kv))
        progress(f"ngl={ngl:>2} fa={'on ' if fa else 'off'} kv={kv:<4} {pt.status:<7}"
                 + (f" pp@d={pt.ppd:.1f} tg@d={pt.tgd:.2f} tok/s" if pt.status == "ok" else f" {pt.note}"))
    best, why = pick_best(points, pp_floor)
    if best is None:
        notes.append("falling back to the planned configuration")
        chosen = base
    else:
        chosen = replace(base, ngl=planmod.all_layers(model) if best.ngl >= top else best.ngl, fa=best.fa, kv=best.kv)
    cond = {"free_vram_mib": free_vram, "ram_available_mib": (profile.get("ram") or {}).get("available_mib"),
            "bench_ctx": depth + BENCH_CTX_OVERHEAD, "budget_s": budget_s}
    return TuneResult(chosen, points, best, why, complete, clock() - start, notes, conditions=cond, plan_name=plan.name)


# ---------------------------------------------------------------------------------
# cache
# ---------------------------------------------------------------------------------

def detect_build(bench: list[str], run: Callable[..., tuple[int, str] | None] = util.run_cmd) -> str:
    """llama.cpp build id: `llama-bench --version` ('version: 12207 (998200491)'), else binary size+mtime.

    `bench` is the full command prefix (it may be a launcher + script), not just the executable."""
    r = run(list(bench) + ["--version"], timeout=60)
    if r:
        # b12207 prints "version: 0.9.0-dev (build 12207, commit 998200491)"; older builds "version: 4567 (abcdef0)"
        m = re.search(r"\(build\s+(\d+)(?:,\s*commit\s+([0-9a-f]+))?\)", r[1]) or re.search(r"version:\s*(\d+)\s*\(([0-9a-f]+)\)", r[1])
        if m:
            return f"b{m.group(1)}" + (f"-{m.group(2)}" if m.group(2) else "")
    try:
        st = Path(bench[-1]).stat()
        return f"bin-{st.st_size}-{int(st.st_mtime)}"
    except OSError:
        return "unknown"


def cache_key(hw_fingerprint: str, model_fingerprint: str, llama_build: str, variant: str = "") -> str:
    """Identity of a tuning result: this hardware, this model file, this llama.cpp build (+ the requested variant)."""
    raw = "\x1f".join([hw_fingerprint, model_fingerprint, llama_build, variant])
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


class Cache:
    def __init__(self, directory: str | Path | None = None):
        self.dir = Path(directory) if directory else util.default_cache_dir()
        self.path = self.dir / CACHE_FILE

    def _read(self) -> dict[str, Any]:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def load(self, key: str) -> dict[str, Any] | None:
        return self._read().get(key)

    def save(self, key: str, entry: dict[str, Any]) -> None:
        data = self._read()
        data[key] = entry
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        tmp.replace(self.path)

    def find(self, hw: str, model_fp: str, build: str, variant: str = "") -> dict[str, Any] | None:
        return self.load(cache_key(hw, model_fp, build, variant))


def variant_of(plan: Plan, depth: int = DEFAULT_DEPTH, pp_floor: float = DEFAULT_PP_FLOOR) -> str:
    """Parts of the request that change what 'best' means besides hardware/model/build.

    Uses the context that was *requested*, not the one the plan ended up with, so that a RAM
    squeeze at serve time still finds the entry that was tuned for the same request."""
    return f"ctx={plan.requested_ctx};depth={depth};floor={pp_floor:g};plan={plan.name}"


def tune_cached(profile: dict[str, Any], model: ModelInfo, plan: Plan, bench: list[str], cache: Cache, *, force: bool = False,
                depth: int = DEFAULT_DEPTH, pp_floor: float = DEFAULT_PP_FLOOR, run: Callable[..., tuple[int, str] | None] = util.run_cmd,
                **kw: Any) -> tuple[TuneResult, str]:
    """Return the cached result for (hardware, model, build) or measure and store it. Returns (result, key)."""
    build = detect_build(bench, run)
    key = cache_key(profile["fingerprint"], model.fingerprint, build, variant_of(plan, depth, pp_floor))
    if not force:
        hit = cache.load(key)
        if hit:
            res = TuneResult.from_dict(hit["result"])
            # a result cut short by its time budget is only reused for requests that allow no more time than it had
            if res.complete or res.conditions.get("budget_s", 0) >= kw.get("budget_s", 0):
                res.from_cache = True
                return res, key
    res = tune(profile, model, plan, bench, depth=depth, pp_floor=pp_floor, run=run, **kw)
    if res.best is not None:
        cache.save(key, {"created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "hardware": profile["fingerprint"],
                         "model": model.fingerprint, "model_path": model.path, "llama_build": build,
                         "variant": variant_of(plan, depth, pp_floor), "result": res.to_dict(),
                         "command": " ".join(planmod.server_command("llama-server", model.path, res.config))})
    return res, key
