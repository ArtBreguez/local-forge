"""Start llama-server with the tuned configuration and hand back an OpenAI-compatible endpoint.

On every start the machine is probed again, because the conditions that were true when the
configuration was tuned may not hold now: another program may hold VRAM or RAM, or the machine
may be on battery. `select_config` picks, in this order:
  1. on battery/UPS                      -> the low-power plan
  2. no GPU, or no layer fits            -> the CPU plan
  3. tuned config, if the SERVER's memory estimate (its own context and ubatch, not llama-bench's) fits the free VRAM/RAM now
  4. otherwise a re-plan on the current free memory, keeping the tuned attention/KV settings and ubatch
If the server still fails to load (e.g. the estimate was optimistic), `fallback_ladder` retries with
one GPU layer fewer at a time, then bigger steps, then a smaller context, instead of giving up.
"""
from __future__ import annotations

import json
import socket
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable

from . import autotune, plan as planmod, probe as probemod, util
from .plan import Caps, Config, ModelInfo, Plan

MAX_ATTEMPTS = 20          # the longest ladder on this model is 16 rungs (see test_fallback_ladder_reaches_the_smaller_contexts)


@dataclass
class Selection:
    name: str                  # plan name: gpu | cpu | low-power
    config: Config
    source: str                # tuned | plan
    reasons: list[str] = field(default_factory=list)
    plan: Plan | None = None


def fits_now(model: ModelInfo, cfg: Config, profile: dict[str, Any]) -> tuple[bool, str]:
    """Will llama-server load `cfg` on the memory that is free right now? (H-SERVER-RECHECK)

    The planner's estimate for the SERVER (its context, its ubatch) against the current budget. A tuned result
    does not get a pass for having been measured: llama-bench ran at a smaller context than the server allocates
    for, so a layer count that loaded there can still fail here (15 layers did: 673 MiB compute buffer at 16K).
    """
    try:
        est = planmod.estimate(model, cfg)
    except ValueError as e:                       # e.g. a ubatch the compute-buffer model does not cover (hand-edited cache)
        return False, str(e)
    ram = (profile.get("ram") or {}).get("available_mib")
    if ram is not None and est.ram > planmod.ram_budget(ram):
        return False, f"needs ~{est.ram:.0f} MiB RAM, budget now {planmod.ram_budget(ram):.0f} MiB of {ram:.0f} MiB available"
    if cfg.ngl > 0:
        if cfg.device:       # a config pinned to one card is checked against that card, not the roomiest one
            gpu = next((g for g in profile.get("gpus") or [] if g.get("device") == cfg.device and g.get("backend")), None)
            if gpu is None:
                return False, f"pinned device {cfg.device} is not present now"
            free = gpu.get("free_mib")
            if free is None and gpu.get("total_mib") is not None:
                free = float(gpu["total_mib"]) * planmod.UNKNOWN_FREE_VRAM_FRACTION
        else:
            gpu, free, _ = planmod.pick_gpu(profile)
        if gpu is None or free is None:
            return False, "no usable GPU now"
        if est.vram > planmod.vram_budget(free):
            return False, (f"needs ~{est.vram:.0f} MiB VRAM at ctx {cfg.ctx}, ubatch {cfg.ubatch} (weights {est.vram_weights:.0f}, KV+state "
                           f"{est.vram_kv + est.vram_state:.0f}, compute {est.vram_compute:.0f}), budget now {planmod.vram_budget(free):.0f} MiB of {free:.0f} MiB free")
    return True, "fits"


def select_config(profile: dict[str, Any], model: ModelInfo, plans: dict[str, Plan], tuned: autotune.TuneResult | None) -> Selection:
    name, why = planmod.choose_plan(profile, plans)
    reasons = [f"plan '{name}': {why}"]
    if tuned is not None and tuned.plan_name and tuned.plan_name != name:
        reasons.append(f"tuned entry ignored: it was measured for the '{tuned.plan_name}' plan, not '{name}'")
        tuned = None
    if tuned is not None:
        spec = plans[name].config.spec_type          # speculation is a serve-time choice; llama-bench never measures it
        cfg = replace(tuned.config, spec_type=spec)
        if cfg.ctx < plans[name].config.ctx:      # the tuning was made under memory pressure that has passed
            cfg = replace(cfg, ctx=plans[name].config.ctx)
            reasons.append(f"context {tuned.config.ctx} -> {cfg.ctx}: more memory is free than when it was tuned")
        ok, detail = fits_now(model, cfg, profile)
        if ok:
            return Selection(name, cfg, "tuned", reasons + ["tuned configuration fits the server's memory estimate"], plans[name])
        reasons.append(f"tuned configuration skipped: {detail}")
        # keep what the tuning learned about attention/KV/ubatch, shrink the layers to what the server can hold now
        replanned = planmod.build_plan(profile, model, mode=name, ctx=plans[name].requested_ctx, fa=tuned.config.fa, kv=tuned.config.kv,
                                       ubatch=min(tuned.config.ubatch, planmod.COMPUTE_MAX_UBATCH), enable_mtp=spec == "draft-mtp", spec=spec if spec in planmod.SPEC_OPT_IN else None)
        return Selection(name, replanned.config, "plan", reasons + [f"re-planned for the server with fa={'on' if tuned.config.fa else 'off'} "
                                                                   f"kv={tuned.config.kv} ubatch={tuned.config.ubatch}: {replanned.config.ngl} layers"], replanned)
    return Selection(name, plans[name].config, "plan", reasons, plans[name])


def _placement(model: ModelInfo, ngl: int) -> tuple[bool, tuple[int, ...]]:
    """What a given -ngl puts on the GPU: (output matrix there?, blocks there). Equal placements are the same server start."""
    return ngl > 0, tuple(planmod.gpu_layer_ids(model, ngl))


def fallback_ladder(cfg: Config, model: ModelInfo) -> list[Config]:
    """The configurations to try, in order, when loading fails (H-FALLBACK-STEP).

    One GPU layer fewer at a time for FALLBACK_FINE_STEPS steps (when the estimate is wrong it is wrong by about one layer:
    the first version of this ladder jumped 15 -> 11 although 14 loads), then steps of 4 layers down to 0, then a smaller
    context with the CPU only. `-ngl` above the layer count (99) means "everything" and counts as the full model.
    """
    ladder = [cfg]
    n = min(cfg.ngl, model.layer_slots + 1)
    fine = 0
    step = 0
    seen = {_placement(model, cfg.ngl)}
    while n > 0:
        if fine < planmod.FALLBACK_FINE_STEPS:
            n -= 1
            fine += 1
        else:
            step = step or max(4, -(-n // 7))     # at most ~7 coarse rungs however deep the model is, so the context rungs stay reachable
            n = max(0, n - step)
        if _placement(model, n) in seen:          # e.g. -ngl 2 and 1 both hold only the output matrix on a model with an MTP slot
            continue
        seen.add(_placement(model, n))
        ladder.append(replace(cfg, ngl=n))
    ctx = cfg.ctx
    while ctx > planmod.CTX_MIN:
        ctx = max(planmod.CTX_MIN, ctx // 2)
        ladder.append(replace(cfg, ngl=0, ctx=ctx))
    return ladder


# ---------------------------------------------------------------------------------
# process handling
# ---------------------------------------------------------------------------------

def http_status(url: str, timeout: float = 3.0) -> int | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:  # noqa: S310 - local URL
            return r.status
    except urllib.error.HTTPError as e:
        return e.code
    except (urllib.error.URLError, OSError, ValueError):
        return None


def connect_host(host: str) -> str:
    """Address to dial for a server bound to `host` (0.0.0.0 / :: cannot be connected to on Windows)."""
    return {"0.0.0.0": "127.0.0.1", "": "127.0.0.1", "::": "::1"}.get(host, host)


def port_in_use(host: str, port: int) -> bool:
    try:
        with socket.create_connection((connect_host(host), port), timeout=1):
            return True
    except OSError:
        return False


def wait_healthy(proc: subprocess.Popen, host: str, port: int, timeout: float, poll: float = 1.0,
                 clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep) -> tuple[bool, str]:
    """Poll GET /health until 200. llama-server answers 503 while the model loads."""
    deadline = clock() + timeout
    while clock() < deadline:
        rc = proc.poll()
        if rc is not None:
            return False, f"server exited with code {rc} before becoming healthy"
        if http_status(f"http://{connect_host(host)}:{port}/health") == 200:
            return True, "healthy"
        sleep(poll)
    return False, f"not healthy after {timeout:g}s"


def tail(path: Path, n: int = 12) -> str:
    try:
        return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-n:])
    except OSError:
        return ""


@dataclass
class Server:
    proc: subprocess.Popen
    config: Config
    command: list[str]
    endpoint: str
    log_path: Path
    attempts: list[dict[str, Any]]

    def stop(self) -> None:
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()


def start_server(server: list[str], model: ModelInfo, cfg: Config, *, host: str, port: int, log_dir: Path, caps: Caps,
                 health_timeout: float = 300.0, max_attempts: int = MAX_ATTEMPTS,
                 progress: Callable[[str], None] = lambda s: None) -> Server:
    """Launch with `cfg`; on a failed load walk down `fallback_ladder`. Raises RuntimeError if nothing starts."""
    log_dir.mkdir(parents=True, exist_ok=True)
    if port_in_use(host, port):
        raise RuntimeError(f"port {port} on {host} is already in use (another server, or a previous llama-server): not starting a second one")
    attempts: list[dict[str, Any]] = []
    for i, c in enumerate(fallback_ladder(cfg, model)[:max_attempts]):
        cmd = planmod.server_command(server, model.path, c, host, port, caps)
        log = log_dir / f"llama-server-{time.strftime('%Y%m%d-%H%M%S')}-{i}.log"
        progress(f"attempt {i + 1}: ngl={c.ngl} ctx={c.ctx} fa={'on' if c.fa else 'off'} kv={c.kv} ub={c.ubatch} spec={c.spec_type or 'none'}")
        with open(log, "wb") as lf:
            proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        ok, detail = wait_healthy(proc, host, port, health_timeout)
        attempts.append({"ngl": c.ngl, "ctx": c.ctx, "ok": ok, "detail": detail, "log": str(log)})
        if ok:
            return Server(proc, c, cmd, f"http://{connect_host(host)}:{port}/v1", log, attempts)
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        progress(f"  failed: {detail}\n  {tail(log, 3)}")
    raise RuntimeError("llama-server did not start with any fallback configuration:\n" + json.dumps(attempts, indent=2))


def serve(model_path: str, *, llama_dir: str | None = None, host: str = "127.0.0.1", port: int = 8080, ctx: int = planmod.CTX_TARGET,
          cache_dir: str | None = None, health_timeout: float = 300.0, server_prefix: list[str] | None = None,
          profile: dict[str, Any] | None = None, depth: int = autotune.DEFAULT_DEPTH, pp_floor: float = autotune.DEFAULT_PP_FLOOR,
          spec: str | None = None, progress: Callable[[str], None] = print) -> tuple[Server, Selection]:
    """Probe, select, launch. `server_prefix` replaces the executable lookup (tests use it for a fake server)."""
    model = planmod.load_model(model_path)
    cache = autotune.Cache(cache_dir)
    server_exe = server_prefix or [util.find_tool("llama-server", llama_dir) or ""]
    if not server_exe[0]:
        raise RuntimeError("llama-server not found (use --llama-dir or $FORGE_LLAMA_DIR)")
    profile = profile or probemod.probe(llama_dir, str(Path(model_path).parent))
    plans = planmod.build_all(profile, model, ctx=ctx, spec=spec)
    name, _ = planmod.choose_plan(profile, plans)
    tuned = None
    bench = util.find_tool("llama-bench", llama_dir)
    if bench:
        entry = cache.find(profile["fingerprint"], model.fingerprint, autotune.detect_build([bench]), autotune.variant_of(plans[name], depth, pp_floor))
        if entry:
            tuned = autotune.TuneResult.from_dict(entry["result"])
    sel = select_config(profile, model, plans, tuned)
    for r in sel.reasons:
        progress(f"select: {r}")
    for w in (sel.plan.warnings if sel.plan else []):
        progress(f"warning: {w}")
    help_run = util.run_cmd(server_exe + ["--help"], timeout=60)
    caps = Caps.from_help(help_run[1]) if help_run else Caps()
    srv = start_server(server_exe, model, sel.config, host=host, port=port, log_dir=cache.dir / "logs", caps=caps,
                       health_timeout=health_timeout, progress=progress)
    return srv, sel
