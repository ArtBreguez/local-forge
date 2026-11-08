"""Put a tuned result into a scratch forge cache so `forge serve` finds it. Used only for the post-fix host check; the entry says it is a replay.

usage: replay_tune.py CACHE_DIR [NGL UBATCH]
  no NGL/UBATCH: replay the real result of the first host tune (results/forge-tune-host-20261108.json: 15 layers, ubatch 512, measured by llama-bench)
  NGL UBATCH   : a SYNTHETIC entry for that layer count / ubatch (not a tune: no measurement behind it), to check the planner's prediction on the server
"""
import json
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

sys.path.insert(0, r"C:\Users\arthu\llm\forge\repo")
from forge import autotune, plan as planmod, probe as probemod, util

LLAMA = r"C:\Users\arthu\llm\llama-b12207"
MODEL = r"C:\Users\arthu\.ollama\models\blobs\sha256-55f51ef810a801849ab84f2ae029d1e6142bf83f563983f550d3ce4898750312"
cache_dir = sys.argv[1]
model = planmod.load_model(MODEL)
prof = probemod.probe(LLAMA, str(Path(MODEL).parent))
plans = planmod.build_all(prof, model)
bench = util.find_tool("llama-bench", LLAMA)
build = autotune.detect_build([bench])
variant = autotune.variant_of(plans["gpu"])
key = autotune.cache_key(prof["fingerprint"], model.fingerprint, build, variant)
raw = json.loads(Path(r"C:\Users\arthu\llm\forge\repo\results\forge-tune-host-20261108.json").read_text(encoding="utf-8"))
entry = next(v for k, v in raw.items() if not k.startswith("_"))
result = entry["result"]
label = "REPLAY of the real result of the first host tune (llama-bench, 15 layers, ubatch 512)"
if len(sys.argv) == 4:
    ngl, ub = int(sys.argv[2]), int(sys.argv[3])
    result["config"] = {**result["config"], "ngl": ngl, "ubatch": ub}
    result["best"] = {**result["best"], "ngl": ngl, "ub": ub}
    label = f"SYNTHETIC entry: {ngl} layers at ubatch {ub}, no measurement behind it (checks the planner's prediction for the server)"
Path(cache_dir).mkdir(parents=True, exist_ok=True)
autotune.Cache(cache_dir).save(key, {"created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "hardware": prof["fingerprint"], "model": model.fingerprint,
                                     "model_path": MODEL, "llama_build": build, "variant": variant, "replay": label, "result": result})
print(f"cache {cache_dir}: key {key} build {build}\nvariant {variant}\n{label}\nconfig {result['config']}")
