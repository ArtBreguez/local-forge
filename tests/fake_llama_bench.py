"""A stand-in for llama-bench used by the autotune tests (runs anywhere Python runs).

Behaviour is driven by environment variables so tests can vary it:
  FAKE_BENCH_LOG       append one line per measurement call: "ngl fa ctk depth ubatch"
  FAKE_BENCH_OOM_NGL   calls with -ngl >= this fail like a real Vulkan out-of-memory (default 18)
  FAKE_BENCH_BUILD     build number printed by --version (default 12207)
  FAKE_BENCH_FA_FAST   if "1", flash attention gives higher generation speed but a much lower prompt speed
  FAKE_BENCH_SLEEP     seconds to sleep before answering (timeout test)
Output format = real `llama-bench -o jsonl` rows (subset of fields), plus backend chatter on stderr.
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def arg(args, flag, default=None):
    return args[args.index(flag) + 1] if flag in args else default


def main(args):
    if "--version" in args:
        print("version: 0.9.0-dev (build %s, commit 998200491)" % os.environ.get("FAKE_BENCH_BUILD", "12207"))
        return 0
    if "--list-devices" in args:
        sys.stdout.write(open(os.path.join(HERE, "fixtures", "list-devices-rx6500xt-idle.txt")).read())
        return 0
    ngl = int(arg(args, "-ngl", "0"))
    fa = arg(args, "-fa", "off") in ("on", "1")
    ctk = arg(args, "-ctk", "f16")
    depths = [int(x) for x in arg(args, "-d", "0").split(",")]
    ub = int(arg(args, "-ub", "512"))
    if os.environ.get("FAKE_BENCH_LOG"):
        with open(os.environ["FAKE_BENCH_LOG"], "a") as f:
            f.write("%d %s %s %s %s\n" % (ngl, "on" if fa else "off", ctk, arg(args, "-d"), arg(args, "-ub", "512")))
    if os.environ.get("FAKE_BENCH_SLEEP"):
        time.sleep(float(os.environ["FAKE_BENCH_SLEEP"]))
    sys.stderr.write("ggml_vulkan: Found 1 Vulkan devices:\n")
    if ngl >= int(os.environ.get("FAKE_BENCH_OOM_NGL", "18")):
        sys.stderr.write(open(os.path.join(HERE, "fixtures", "llama-bench-oom-stderr.txt"), encoding="utf-8").read())
        return 1
    fa_fast = os.environ.get("FAKE_BENCH_FA_FAST") == "1"
    for d in depths:
        base_tg = 5.5 + 0.25 * min(ngl, 16) - (0.0 if d == 0 else 0.8)
        tg = base_tg + ((0.6 if fa_fast else -1.2) if fa and d else 0) - (0.3 if ctk == "q8_0" else 0)
        pp = (190.0 if not (fa and d) else (60.0 if fa_fast else 85.0)) * (1.0 if ub >= 512 else 0.78)   # sweep D: ubatch 256 prefills at 151 tok/s, 512 at 195
        for n_prompt, n_gen, v in ((512, 0, pp), (0, 128, tg)):
            print(json.dumps({"build_number": int(os.environ.get("FAKE_BENCH_BUILD", "12207")), "n_gpu_layers": ngl,
                              "flash_attn": int(fa), "type_k": ctk, "n_prompt": n_prompt, "n_gen": n_gen, "n_depth": d,
                              "avg_ts": v, "stddev_ts": 0.1}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
