"""`python -m forge <probe|plan|tune|serve|heuristics>`"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from . import autotune, plan as planmod, probe as probemod, util
from .plan import Caps


def _common(p: argparse.ArgumentParser, model: bool = True) -> None:
    p.add_argument("--llama-dir", default=os.environ.get("FORGE_LLAMA_DIR"), help="folder with llama-bench/llama-server (or $FORGE_LLAMA_DIR; PATH is searched too)")
    p.add_argument("--cache-dir", default=None, help="where tuned results and logs go (default: $FORGE_HOME or a per-user cache dir)")
    if model:
        p.add_argument("--model", default=os.environ.get("FORGE_MODEL"), help="path to the GGUF file (or $FORGE_MODEL)")
        p.add_argument("--ctx", type=int, default=planmod.CTX_TARGET, help="context length to plan for (default %(default)s)")
        p.add_argument("--profile", help="use this saved profile JSON instead of probing (for planning another machine or testing)")


def _add_spec(p: argparse.ArgumentParser) -> None:
    p.add_argument("--spec", choices=["none", *planmod.SPEC_OPT_IN], default="none",
                   help="speculative decoding: off by default (H-SPEC-NONE); ngram-mod is opt-in, it was faster on repeated code but not output-identical at temperature 0")


def _need_model(a: argparse.Namespace) -> str:
    if not a.model:
        sys.exit("error: --model (or $FORGE_MODEL) is required")
    if not Path(a.model).is_file():
        sys.exit(f"error: model file not found: {a.model}")
    return a.model


def _profile(a: argparse.Namespace) -> dict[str, Any]:
    if getattr(a, "profile", None):
        return json.loads(Path(a.profile).read_text(encoding="utf-8"))
    mdir = str(Path(a.model).parent) if getattr(a, "model", None) else None
    return probemod.probe(a.llama_dir, mdir)


def cmd_probe(a: argparse.Namespace) -> int:
    print(json.dumps(probemod.probe(a.llama_dir, a.model_dir), indent=2))
    return 0


def cmd_plan(a: argparse.Namespace) -> int:
    model = planmod.load_model(_need_model(a))
    if a.dump_model:
        print(json.dumps(asdict(model)))
        return 0
    prof = _profile(a)
    kw: dict[str, Any] = {"ctx": a.ctx, "kv": a.kv, "fa": {"on": True, "off": False, None: None}[a.fa], "enable_mtp": a.mtp,
                          "ubatch": a.ubatch, "spec": None if a.spec == "none" else a.spec}
    plans = planmod.build_all(prof, model, **kw)
    name, why = planmod.choose_plan(prof, plans)
    if a.json:
        print(json.dumps({"applies": name, "reason": why, "plans": {k: v.to_dict() for k, v in plans.items()}}, indent=2))
        return 0
    print(f"model: {model.arch} {model.quant} {model.file_mib:.0f} MiB, {model.trunk_layers} blocks "
          f"({len(model.attn_layers)} with KV cache), KV {model.kv_bytes_per_token('f16') / 1024:.1f} KiB/token f16; "
          f"-ngl N = N-{model.layer_slots + 1 - model.trunk_layers} blocks + the output matrix on the GPU")
    for k, v in plans.items():
        print(("* " if k == name else "  ") + (planmod.explain(v) if a.explain else v.summary()))
    print(f"applies now: {name} ({why})")
    return 0


def cmd_tune(a: argparse.Namespace) -> int:
    model = planmod.load_model(_need_model(a))
    bench = util.find_tool("llama-bench", a.llama_dir)
    if not bench:
        sys.exit("error: llama-bench not found (use --llama-dir or $FORGE_LLAMA_DIR)")
    prof = _profile(a)
    plans = planmod.build_all(prof, model, ctx=a.ctx)
    name = a.mode or planmod.choose_plan(prof, plans)[0]
    p = plans[name]
    help_run = util.run_cmd([bench, "--help"], timeout=60)
    caps = Caps.from_help(help_run[1]) if help_run else Caps()    # unknown build: assume the current flag style
    print(f"tuning plan '{name}': {p.summary()}\nbudget {a.budget:g}s, depth {a.depth}, reps {a.reps}, prompt floor {a.pp_floor:g} tok/s", flush=True)
    cache = autotune.Cache(a.cache_dir)
    res, key = autotune.tune_cached(prof, model, p, [bench], cache, force=a.force, depth=a.depth, pp_floor=a.pp_floor,
                                    reps=a.reps, budget_s=a.budget, caps=caps, ubatches=tuple(int(x) for x in a.ubatches.split(",")),
                                    progress=lambda s: print("  " + s, flush=True))
    print(("cached result" if res.from_cache else f"measured in {res.elapsed_s:.0f}s") + f" (key {key}, cache {cache.path})")
    if res.best is None:
        print("no configuration could be measured; using the plan:", p.summary())
    else:
        b = res.best
        print(f"best: ngl={b.ngl} fa={'on' if b.fa else 'off'} kv={b.kv} ub={b.ub}: {b.tgd:.2f} tok/s generation, {b.ppd:.1f} tok/s prompt at depth ({res.objective})")
    if not res.complete:
        print("note: time budget ran out; result is the best of a partial search")
    server = util.find_tool("llama-server", a.llama_dir) or "llama-server"
    print("llama-server command line (`forge serve` re-checks it against the server's own memory estimate before using it):\n  "
          + " ".join(planmod.server_command(server, model.path, res.config, caps=Caps())))
    return 0


def cmd_serve(a: argparse.Namespace) -> int:
    from . import serve as servemod
    model_path = _need_model(a)
    try:
        srv, sel = servemod.serve(model_path, llama_dir=a.llama_dir, host=a.host, port=a.port, ctx=a.ctx, cache_dir=a.cache_dir,
                                  health_timeout=a.health_timeout, profile=json.loads(Path(a.profile).read_text()) if a.profile else None,
                                  depth=a.depth, pp_floor=a.pp_floor, spec=None if a.spec == "none" else a.spec,
                                  progress=lambda s: print(s, flush=True))
    except RuntimeError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(json.dumps({"endpoint": srv.endpoint, "plan": sel.name, "source": sel.source, "config": asdict(srv.config),
                      "attempts": [{k: v for k, v in at.items() if k != "log"} for at in srv.attempts],
                      "command": srv.command, "log": str(srv.log_path)}, indent=2), flush=True)
    if a.once:
        srv.stop()
        return 0
    try:
        return srv.proc.wait()
    except KeyboardInterrupt:
        srv.stop()
        return 0


def cmd_heuristics(a: argparse.Namespace) -> int:
    for h in planmod.HEURISTICS.values():
        print(f"{h.id}  [{h.status}]\n  rule: {h.rule}")
        for path, note in h.evidence:
            print(f"  evidence: {path or 'NONE'}: {note}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="forge", description="hardware-adaptive llama.cpp serving")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("probe", help="print the machine profile as JSON")
    _common(p, model=False)
    p.add_argument("--model-dir")
    p.set_defaults(fn=cmd_probe)

    p = sub.add_parser("plan", help="compute starting configurations (gpu, cpu, low-power) without running anything")
    _common(p)
    p.add_argument("--kv", choices=["f16", "q8_0"], default=None, help="force the KV cache type")
    p.add_argument("--fa", choices=["on", "off"], default=None, help="force flash attention")
    p.add_argument("--mtp", action="store_true", help="enable MTP speculative decoding if the model has it")
    p.add_argument("--ubatch", type=int, default=planmod.UBATCH, help="physical batch size (-ub) to plan for (default %(default)s; `tune` also tries 256)")
    _add_spec(p)
    p.add_argument("--explain", action="store_true", help="show the heuristic behind every decision")
    p.add_argument("--json", action="store_true")
    p.add_argument("--dump-model", action="store_true", help="print the parsed model info as JSON and exit")
    p.set_defaults(fn=cmd_plan)

    p = sub.add_parser("tune", help="measure around the plan with llama-bench and cache the best configuration")
    _common(p)
    p.add_argument("--mode", choices=["gpu", "cpu", "low-power"], help="tune this plan instead of the one that applies now")
    p.add_argument("--budget", type=float, default=1200.0, help="seconds (default %(default)s)")
    p.add_argument("--depth", type=int, default=autotune.DEFAULT_DEPTH, help="agent depth: tokens already in the cache (default %(default)s)")
    p.add_argument("--reps", type=int, default=1, help="llama-bench repetitions per point (default %(default)s; the protocol uses 3 for reported numbers)")
    p.add_argument("--pp-floor", type=float, default=autotune.DEFAULT_PP_FLOOR, help="minimum prompt tok/s at depth (default %(default)s)")
    p.add_argument("--ubatches", default=",".join(map(str, planmod.UBATCH_CHOICES)), help="physical batch sizes to try, comma-separated (default %(default)s; `serve` only finds a tuning made with the default list)")
    p.add_argument("--force", action="store_true", help="ignore the cache")
    p.set_defaults(fn=cmd_tune)

    p = sub.add_parser("serve", help="start llama-server with the tuned (or planned) configuration")
    _common(p)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--health-timeout", type=float, default=300.0)
    p.add_argument("--depth", type=int, default=autotune.DEFAULT_DEPTH, help="use the tuning made with this agent depth (default %(default)s)")
    p.add_argument("--pp-floor", type=float, default=autotune.DEFAULT_PP_FLOOR, help="use the tuning made with this prompt floor (default %(default)s)")
    p.add_argument("--once", action="store_true", help="start, check health, print the endpoint, stop (smoke test)")
    _add_spec(p)
    p.set_defaults(fn=cmd_serve)

    p = sub.add_parser("heuristics", help="list every planning rule with its evidence")
    p.set_defaults(fn=cmd_heuristics)

    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
