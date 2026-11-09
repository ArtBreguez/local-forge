"""Phase 3 quality benchmark on Modal GPUs: same GGUF files, same runner, many models in parallel.

Why: answer quality depends on the model file and the sampling, not on the card, so the A/B between models (and pass@k for best-of-N)
can run on rented GPUs while the host keeps measuring speed and the agent. Speed numbers from here mean nothing for the host.

    modal run tools/modal_qbench.py                    # every model in MODELS, both arms
    modal run tools/modal_qbench.py --only heretic-9b  # one model

Arms (03-agent/quality/run_quality.py, unchanged):
  greedy : temperature 0, seed 1234, thinking off, 1 sample  (= phase 3 arm A)
  sample : temperature 0.7, seeds 1234.., thinking off, 10 samples -> pass@1/5/10 (unbiased estimator, hidden tests = oracle;
           an upper bound for best-of-N, which phase 6 selects with visible tests only)
Results land in results/phase6/modal/<model>.<arm>.jsonl on this machine.
"""
from __future__ import annotations

import json
import pathlib
import subprocess
import time

import modal

ROOT = pathlib.Path(__file__).resolve().parents[1]
# ghcr no longer publishes per-build CUDA tags (b12207 has none); pinned by digest, version printed in the meta line
LLAMA_TAG = "server-cuda12"
LLAMA_DIGEST = "sha256:f28803153f5ce48a6c8deaef694f58fca09b5dd79baca75fced670ec23b7bd71"

# (name, repo, file, gpu, sha256 or None) -- licences: all Apache-2.0 except the heretic merge (see its card)
MODELS = [
    ("heretic-9b", "mradermacher/DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic-i1-GGUF",
     "DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic.i1-IQ4_XS.gguf", "L4"),
    ("qwen35-9b", "unsloth/Qwen3.5-9B-GGUF", "Qwen3.5-9B-IQ4_XS.gguf", "L4"),
    ("qwen25-coder-7b", "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF", "qwen2.5-coder-7b-instruct-q4_k_m.gguf", "L4"),
    ("qwen3-coder-30b-a3b", "unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF", "Qwen3-Coder-30B-A3B-Instruct-UD-IQ3_XXS.gguf", "L4"),
    ("qwen35-35b-a3b", "unsloth/Qwen3.5-35B-A3B-GGUF", "Qwen3.5-35B-A3B-UD-IQ3_XXS.gguf", "L4"),
    ("qwen35-27b", "unsloth/Qwen3.5-27B-GGUF", "Qwen3.5-27B-Q4_K_M.gguf", "L40S"),
]
ARMS = {
    "greedy": ["--temperature", "0", "--seed", "1234", "--samples", "1"],
    "sample": ["--temperature", "0.7", "--seed", "1234", "--samples", "10"],
}
WORKERS = 4   # llama-server slots and parallel runner processes (each on a disjoint problem subset)

image = (
    modal.Image.from_registry(f"ghcr.io/ggml-org/llama.cpp@{LLAMA_DIGEST}", add_python="3.11")
    .entrypoint([])
    .apt_install("nodejs")
    .pip_install("huggingface_hub[hf_transfer]==0.36.0")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .add_local_dir(ROOT / "03-agent" / "quality", "/qbench", ignore=["__pycache__", "_ref"])
)
models_vol = modal.Volume.from_name("lf-models", create_if_missing=True)
app = modal.App("lf-qbench", image=image)


def _serve(path: str, slots: int) -> subprocess.Popen:
    import urllib.request
    srv = subprocess.Popen(["/app/llama-server", "-m", path, "--host", "127.0.0.1", "--port", "8080", "-ngl", "999",
                            "-c", str(16384 * slots), "--parallel", str(slots), "-fa", "off", "-ctk", "f16", "-ctv", "f16"],
                           stdout=subprocess.DEVNULL, stderr=open("/tmp/server.log", "w"))
    for _ in range(600):
        try:
            if urllib.request.urlopen("http://127.0.0.1:8080/health", timeout=2).status == 200:
                return srv
        except Exception:
            pass
        if srv.poll() is not None:
            raise RuntimeError("llama-server died:\n" + open("/tmp/server.log").read()[-3000:])
        time.sleep(1)
    raise RuntimeError("llama-server never became healthy")


def run_one(name: str, repo: str, fname: str, smoke: bool = False) -> dict:
    import hashlib
    from huggingface_hub import hf_hub_download
    t0 = time.time()
    path = hf_hub_download(repo, fname, cache_dir="/models")
    models_vol.commit()
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    sha = h.hexdigest()
    version = subprocess.run(["/app/llama-server", "--version"], capture_output=True, text=True).stderr.strip().splitlines()[-2:]
    srv = _serve(path, WORKERS)
    problems = sorted(p.name for p in pathlib.Path("/qbench/problems").iterdir())
    if smoke:
        problems = problems[:2] + problems[-2:]
    out = {}
    try:
        for arm, flags in ARMS.items():
            if smoke:
                flags = [x if x != "10" else "2" for x in flags]
            procs = []
            for w in range(WORKERS):
                subset = problems[w::WORKERS]
                procs.append(subprocess.Popen(
                    ["python", "/qbench/run_quality.py", "--endpoint", "http://127.0.0.1:8080/v1", "--arm", f"{name}.{arm}",
                     "--thinking", "off", "--model-name", name, "--problems", ",".join(subset), "--node", "/usr/bin/node",
                     "--note", f"modal {LLAMA_TAG} {repo}/{fname} sha256={sha}", "--out", f"/tmp/{arm}.{w}.jsonl", *flags],
                    cwd="/qbench", stdout=subprocess.DEVNULL, stderr=open(f"/tmp/{arm}.{w}.err", "w")))
            for p in procs:
                p.wait()
            lines = []
            for w in range(WORKERS):
                f = pathlib.Path(f"/tmp/{arm}.{w}.jsonl")
                lines += f.read_text().splitlines() if f.exists() else [json.dumps({"worker_error": open(f"/tmp/{arm}.{w}.err").read()[-2000:]})]
            out[arm] = "\n".join(lines) + "\n"
    finally:
        srv.kill()
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    out["_meta"] = json.dumps({"gpu": gpu, "model": name, "repo": repo, "file": fname, "sha256": sha, "llama": LLAMA_TAG, "llama_digest": LLAMA_DIGEST, "llama_version": version,
                               "wall_s": round(time.time() - t0)})
    return out


# one Modal function per GPU type (gpu is fixed at decoration time)
@app.function(gpu=["L4", "A10", "L40S"], volumes={"/models": models_vol}, timeout=3 * 3600, max_containers=6)
def run_l4(name: str, repo: str, fname: str, smoke: bool = False) -> dict:
    return run_one(name, repo, fname, smoke)


@app.function(gpu=["L40S", "A100-40GB"], volumes={"/models": models_vol}, timeout=3 * 3600, max_containers=2)
def run_l40s(name: str, repo: str, fname: str, smoke: bool = False) -> dict:
    return run_one(name, repo, fname, smoke)


@app.local_entrypoint()
def main(only: str = "", smoke: bool = False):
    dest = ROOT / "results" / "phase6" / ("modal-smoke" if smoke else "modal")
    dest.mkdir(parents=True, exist_ok=True)
    todo = [m for m in MODELS if not only or m[0] in only.split(",")]
    calls = [((run_l40s if gpu == "L40S" else run_l4).spawn(name, repo, fname, smoke), name) for name, repo, fname, gpu in todo]
    for call, name in calls:
        try:
            res = call.get()
        except Exception as e:  # one model failing must not lose the others
            print(f"{name}: FAILED {e!r}"[:500])
            continue
        for arm in ARMS:
            (dest / f"{name}.{arm}.jsonl").write_text(res[arm])
        (dest / f"{name}.meta.json").write_text(res["_meta"])
        print(f"{name}: done {res['_meta']}")
