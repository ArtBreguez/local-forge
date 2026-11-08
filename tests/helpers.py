"""Shared test helpers: fixtures, synthetic GGUF files, profiles, free ports."""
from __future__ import annotations

import json
import socket
import sys
from pathlib import Path

from forge import gguf, plan, probe

ROOT = Path(__file__).resolve().parent.parent
FIX = ROOT / "tests" / "fixtures"
FAKE_BENCH = [sys.executable, str(ROOT / "tests" / "fake_llama_bench.py")]
FAKE_SERVER = [sys.executable, str(ROOT / "tests" / "fake_llama_server.py")]


def fixture(name: str) -> str:
    return (FIX / name).read_text(encoding="utf-8")


def fit_params_grid() -> dict[tuple[int, int, int, str, str], tuple[int, int, int]]:
    """llama-fit-params b12207 on the host (results/fit-params-grid-host-20261108.txt): (ngl, ctx, ubatch, fa, kv) -> Vulkan0 (model, context, compute) MiB.

    llama.cpp's own estimate of what it allocates; `-fit off -fitp on`, so nothing was loaded and the numbers do not depend on free memory."""
    out = {}
    for line in (ROOT / "results" / "fit-params-grid-host-20261108.txt").read_text(encoding="utf-8-sig").splitlines():
        if line.startswith("#") or "|" not in line:
            continue
        head, vulkan, _host = line.split("|")
        ngl, ctx, ub, fa, kv = head.split()
        out[(int(ngl), int(ctx), int(ub), fa, kv)] = tuple(int(x) for x in vulkan.split()[1:4])
    return out


def real_model() -> plan.ModelInfo:
    """The DeepSeek-V4-Pro-Qwen3.5-9B heretic IQ4_XS, parsed from the real file by forge/gguf.py on the host."""
    return plan.ModelInfo(**json.loads(fixture("modelinfo-dsv4-9b.json")))


def profile(free_vram: int | None = 3443, ram_free: int = 5600, battery: bool = False, gpu: bool = True,
            cores: int | None = 6, threads: int = 12, backend: str = "Vulkan", vram_total: int = 4080,
            gpu_name: str = "AMD Radeon RX 6500 XT") -> dict:
    """The Windows host as probed with an idle GPU (numbers from tests/fixtures), adjustable per test."""
    p = {"cpu": {"model": "AMD Ryzen 5 3600 6-Core Processor", "physical_cores": cores, "logical_threads": threads},
         "os": {"system": "Windows"},
         "ram": {"total_mib": 16300, "available_mib": ram_free},
         "gpus": ([{"index": 0, "device": backend + "0", "backend": backend, "name": gpu_name,
                    "total_mib": vram_total, "free_mib": free_vram}] if gpu else []),
         "power": {"on_battery": battery}}
    p["fingerprint"] = probe.fingerprint(p)
    return p


def tiny_gguf(path: Path, *, layers: int = 8, hybrid: bool = True, nextn: int = 0, mtp_tensors: bool = False,
              layer_kib: int = 64, tied: bool = False) -> Path:
    """A few-hundred-KiB GGUF shaped like a small hybrid (every 4th layer is attention) or dense model."""
    a = "qwen35" if hybrid else "llama"
    meta = {"general.architecture": a, "general.name": "tiny", "general.file_type": 30,
            f"{a}.block_count": layers + nextn, f"{a}.embedding_length": 64, f"{a}.context_length": 32768,
            f"{a}.attention.head_count": 4, f"{a}.attention.head_count_kv": 2,
            f"{a}.attention.key_length": 16, f"{a}.attention.value_length": 16,
            f"{a}.nextn_predict_layers": nextn,
            "tokenizer.ggml.tokens": ["t%d" % i for i in range(gguf.MAX_KEPT_ARRAY + 10)]}
    if hybrid:
        meta[f"{a}.full_attention_interval"] = 4
        meta[f"{a}.ssm.conv_kernel"] = 4
        meta[f"{a}.ssm.state_size"] = 16
        meta[f"{a}.ssm.inner_size"] = 64
        meta[f"{a}.ssm.group_count"] = 2
    tensors = [("token_embd.weight", (64, 100), 23, 16384), ("output_norm.weight", (64,), 0, 256)]
    if not tied:
        tensors.insert(1, ("output.weight", (64, 100), 23, 16384))
    for i in range(layers):
        tensors.append((f"blk.{i}.ffn_up.weight", (64, 64), 23, layer_kib * 1024))
        tensors.append((f"blk.{i}.attn_norm.weight", (64,), 0, 256))
    if mtp_tensors:
        for i in range(layers, layers + nextn):
            tensors.append((f"blk.{i}.nextn.eh_proj.weight", (64, 64), 23, 8192))
    gguf.write_synthetic(str(path), meta, tensors)
    return path


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]
