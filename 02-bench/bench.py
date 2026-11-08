"""Benchmark dsv4-9b-heretic on Ollama across GPU layers / context / threads.
Writes one JSON line per config to bench-<tag>.jsonl. Usage: python bench.py <tag>"""
import json, sys, time, urllib.request, itertools, pathlib, subprocess

API = "http://127.0.0.1:11434"
MODEL = "dsv4-9b-heretic"
TAG = sys.argv[1] if len(sys.argv) > 1 else "a"
OUT = pathlib.Path(__file__).with_name(f"bench-{TAG}.jsonl")
SRC = pathlib.Path(__file__).with_name("agent-test") / "test" / "duration.test.js"
# ~2K-token prompt: what an agent turn looks like (instructions + a file + a test log)
PROMPT = ("You are a coding agent. Read the file and the failing test output, then write the fixed "
          "function only.\n\n" + (SRC.read_text() + "\n") * 6 +
          "\nnode --test output:\n" + "not ok 1 - combined units\n  expected 5400, got RangeError\n" * 20)

def post(path, body, timeout=900):
    req = urllib.request.Request(API + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())

def ps():
    try:
        out = subprocess.run([r"C:\Users\arthu\AppData\Local\Programs\Ollama\ollama.exe", "ps"],
                             capture_output=True, text=True, timeout=30).stdout.splitlines()
        return " ".join(out[1].split()[2:6]) if len(out) > 1 else ""
    except Exception as e:
        return f"ps error {e}"

def run(opts):
    post("/api/generate", {"model": MODEL, "keep_alive": 0})  # unload so options apply
    time.sleep(3)
    t0 = time.time()
    try:
        r = post("/api/generate", {"model": MODEL, "prompt": PROMPT, "stream": False, "think": False,
                                   "keep_alive": "2m", "options": {**opts, "num_predict": 192,
                                   "temperature": 0, "seed": 1}})
    except Exception as e:
        return {"opts": opts, "error": str(e)[:200], "wall": round(time.time() - t0, 1)}
    split = ps()
    ns = 1e9
    return {"opts": opts, "split": split,
            "prompt_tokens": r.get("prompt_eval_count"),
            "pp_tps": round(r["prompt_eval_count"] / (r["prompt_eval_duration"] / ns), 1) if r.get("prompt_eval_duration") else None,
            "gen_tokens": r.get("eval_count"),
            "tg_tps": round(r["eval_count"] / (r["eval_duration"] / ns), 2) if r.get("eval_duration") else None,
            "load_s": round(r.get("load_duration", 0) / ns, 1), "wall": round(time.time() - t0, 1)}

if TAG == "a":
    grid = [dict(num_ctx=c, num_gpu=g, num_thread=t)
            for c, g, t in itertools.product([16384, 32768], [8, 12, 16, 20, 24], [6])]
    grid += [dict(num_ctx=16384, num_gpu=16, num_thread=12), dict(num_ctx=16384, num_gpu=16, num_thread=5)]
else:  # after server-level changes (flash attention, KV quant): re-run the neighbourhood of the best
    grid = [dict(num_ctx=c, num_gpu=g, num_thread=6) for c, g in itertools.product([16384, 32768], [12, 16, 20, 24, 28])]

OUT.write_text("")
for opts in grid:
    res = run(opts)
    with OUT.open("a") as f:
        f.write(json.dumps(res) + "\n")
post("/api/generate", {"model": MODEL, "keep_alive": 0})
with OUT.open("a") as f:
    f.write('{"done": true}\n')
