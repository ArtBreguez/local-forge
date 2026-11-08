"""Compute a starting llama-server configuration from a machine profile and a GGUF header.

Nothing is run here. The plan is arithmetic over (a) what the GGUF header says the model
needs and (b) what the probe says the machine has, using the named heuristics in HEURISTICS.
Each heuristic states the measurement it comes from or says "untested". `forge tune`
(autotune.py) is what replaces guesses with measurements.

All sizes are MiB unless a name says bytes.
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from typing import Any

from . import gguf

MIB = 1024 * 1024
KV_BYTES_PER_ELEM = {"f16": 2.0, "q8_0": 34 / 32, "q4_0": 18 / 32}  # ggml block layouts: q8_0 = 32 values + fp16 scale

# ---------------------------------------------------------------------------------
# Heuristics registry. evidence = [(repo path or None, what it shows)].
# status: measured | partially measured | derived | untested
# ---------------------------------------------------------------------------------

@dataclass(frozen=True)
class Heuristic:
    id: str
    rule: str
    status: str
    evidence: tuple[tuple[str | None, str], ...]


def _h(id: str, rule: str, status: str, *evidence: tuple[str | None, str]) -> Heuristic:
    return Heuristic(id, rule, status, tuple(evidence))


# Constants used by the rules (named so tests and docs can refer to them).
VRAM_MARGIN_FRACTION = 0.05   # of free VRAM, kept unused
VRAM_MARGIN_MIB = 128         # plus this much
COMPUTE_BASE_MIB = 1100       # graph scratch buffers on the GPU (fitted, see H-LAYERS-VRAM-FIT)
RAM_USE_FRACTION = 0.95       # of available RAM the plan may commit (see H-CTX-RAM: a CPU-only run completed at 93%)
EMBD_RESIDENT_FRACTION = 0.0  # share of token_embd counted as resident: it is read by row, so mmap only pages in what is used
HOST_OVERHEAD_MIB = 512       # process, tokenizer, runtime, staging buffers
CTX_TARGET = 16384
CTX_MIN = 4096
Q8_KV_MIN_GAIN_LAYERS = 6     # q8_0 KV (needs flash attention) must buy at least this many GPU layers
UBATCH = 512
UNKNOWN_FREE_VRAM_FRACTION = 0.80

HEURISTICS: dict[str, Heuristic] = {h.id: h for h in [
    _h("H-LAYERS-VRAM-FIT",
       f"GPU layers = the largest count whose estimated VRAM (weights of the offloaded layers + KV of their attention layers + "
       f"recurrent state + {COMPUTE_BASE_MIB} MiB graph scratch) stays below free VRAM x (1-{VRAM_MARGIN_FRACTION}) - {VRAM_MARGIN_MIB} MiB. "
       "The scratch size is a fit, not a derivation. At the default ubatch (512) on 3443 MiB free: 16 layers load and run and 17 do not "
       "(weights+KV+state: 1978 and 2119 MiB), so scratch lies in (1324, 1465] MiB. The plan reserves scratch + margin = about 1400 MiB, inside that window.",
       "partially measured",
       ("tests/fixtures/llama-bench-sweepc-snapshot.log",
        "sweep C, llama.cpp b12207 Vulkan, llama-bench at ~8.7K context: ngl=16 loads for FA off/on and f16/q8_0 KV, ngl=20 fails to create the "
        "context for all three, ngl>=24 fails to load"),
       ("tests/fixtures/llama-bench-sweepd-snapshot.log",
        "sweep D, ubatch 512, FA off, f16: ngl=16 runs at depth 8192; ngl=17 fails while filling the 8192 depth; ngl 18-20 fail to create the context"),
       ("tests/fixtures/llama-bench-oom-stderr.txt", "the allocation failure text (Vulkan0 buffer, ErrorOutOfDeviceMemory)"),
       ("docs/JOURNAL.md", "sweep A/B (Ollama): 16 layers at 16K fit; Ollama also fitted 20 with FA + q8_0, which llama.cpp did not reproduce"),
       (None, "the scratch size depends on the ubatch (ub=256 fits 18 layers, see H-UBATCH) but the model has one constant for ub=512; "
              "one machine, one model: not checked against another GPU or architecture, and not against llama.cpp's own allocation log")),
    _h("H-VRAM-HEADROOM",
       f"Never plan to the last MiB: keep {VRAM_MARGIN_FRACTION:.0%} of free VRAM + {VRAM_MARGIN_MIB} MiB unused, because free VRAM "
       "moves while the machine is used. `forge serve` tolerates the same drift before it abandons a tuned configuration.",
       "measured",
       ("tests/fixtures/list-devices-rx6500xt-idle.txt", "3443 MiB free on the idle card"),
       ("tests/fixtures/list-devices-rx6500xt-busy.txt", "3105 MiB free minutes later with a benchmark running: 338 MiB (10%) gone")),
    _h("H-OUTPUT-LAYER",
       "The output projection (output.weight) goes to the GPU only when layers > the number of transformer blocks; "
       "with fewer layers it stays in RAM and its size is not charged to VRAM.",
       "untested",
       (None, "llama.cpp offload semantics ('offloaded N/N layers' log line); not verified on this machine")),
    _h("H-KV-ATTN-ONLY",
       "KV cache exists only in full-attention layers. Layer kinds come from the GGUF (recurrent_layers array, "
       "full_attention_interval, or per-layer head_count_kv); recurrent layers keep a fixed-size state instead.",
       "derived",
       ("tests/fixtures/gguf-dump-dsv4-9b.txt", "qwen35: 8 of 32 blocks are full attention (every 4th), "
                                                  "so f16 KV costs 32 KiB/token, not 128 KiB"),
       (None, "the allocation llama.cpp actually makes for this architecture has not been measured")),
    _h("H-FA-BACKEND",
       "Flash attention off + f16 KV on Vulkan and on CPU. On CUDA/ROCm/Metal: on (untested here).",
       "measured",
       ("tests/fixtures/llama-bench-sweepc-snapshot.jsonl",
        "llama.cpp b12207 Vulkan, depth 8192, tg128 tok/s FA off -> FA on (f16): ngl=0 5.45 -> 4.44, ngl=8 6.78 -> 5.76, "
        "ngl=12 7.65 -> 6.59, ngl=16 9.15 -> 7.83; pp512 at depth 8192 drops 2.0-2.3x (ngl=16: 195 -> 84)"),
       (None, "no CUDA, ROCm or Metal measurement exists in this repo; on those backends 'on' is the upstream default and untested here")),
    _h("H-KV-Q8-GAIN",
       f"Use q8_0 KV (which requires flash attention) only if it lets at least {Q8_KV_MIN_GAIN_LAYERS} more layers onto the GPU "
       "than f16 KV does, or if f16 KV does not fit the context in RAM at all.",
       "partially measured",
       ("tests/fixtures/llama-bench-sweepc-snapshot.jsonl",
        "FA on + q8_0 at depth 8192 is the slowest setting at every ngl (ngl=16: 7.45 tok/s vs 9.15 with FA off + f16), and it did not "
        "let more layers load (ngl=20 failed for all three)"),
       ("docs/JOURNAL.md", "sweep B: Ollama with FA + q8_0 gave 9.19 tok/s at 16 layers; 20 layers 8.96"),
       (None, f"the threshold {Q8_KV_MIN_GAIN_LAYERS} is a judgement: the FA penalty (~1.3 tok/s at ngl=16) is worth roughly the gain of 4-8 layers")),
    _h("H-THREADS-PHYSICAL",
       "Generation threads = physical cores (not SMT threads).",
       "measured",
       ("docs/JOURNAL.md", "sweep A: 6 threads (= physical cores) 8.32 tok/s, 5 threads 8.00, 12 threads 7.81"),
       ("results/bench-a.jsonl", "raw rows")),
    _h("H-CTX-TARGET",
       f"Plan for a {CTX_TARGET}-token context (override with --ctx): the agent's system prompt and tool schemas already take "
       "several thousand tokens, and PROTOCOL.md phase 3 compares 16K and 32K.",
       "partially measured",
       ("docs/JOURNAL.md", "sweep A/B: 16K allows 16 (A) / 20 (B) layers, 32K only 12 (A) / 16 (B)"),
       ("docs/PROTOCOL.md", "phase 3 arms: 16K and 32K"),
       (None, "16K itself is a judgement: no agent run has compared contexts yet (PROTOCOL.md phase 4)")),
    _h("H-CTX-RAM",
       f"RAM need = weights kept on the CPU (the token embedding is not counted: it is read by row, mmap pages in only what is used) + KV and "
       f"recurrent state of the CPU layers + {HOST_OVERHEAD_MIB} MiB. If it exceeds {RAM_USE_FRACTION:.0%} of available RAM, halve the context "
       "(down to 4096); if even that does not fit the plan is marked infeasible and the model would run from disk via mmap.",
       "partially measured",
       ("tests/fixtures/llama-bench-sweepc-snapshot.log",
        "sweep C ran CPU-only (ngl=0, ~8.7K context) to completion with 5.5 GiB free; this rule's estimate for that run is 5264 MiB = 93% of it, "
        "so the fraction cannot be lower than that without calling a configuration that worked infeasible (0.80 did, which is how this was found)"),
       (None, "one data point, at the edge: no run with less RAM, no check of resident size, and 5% left for the OS is thin; "
              "with --no-mmap or --mlock the embedding would be resident")),
    _h("H-MTP",
       "MTP speculative decoding is 'available' only if nextn_predict_layers > 0 AND the file contains the MTP block tensors; "
       "it is switched on (--spec-type draft-mtp) only when asked for, until PROTOCOL.md phase 2 measures it.",
       "derived",
       ("tests/fixtures/gguf-dump-dsv4-9b.txt",
        "this GGUF declares nextn_predict_layers=1 but has no blk.32.* tensors (427 tensors, blocks 0-31): the MTP head is not in the file"),
       ("docs/PROTOCOL.md", "phase 2 is where --spec-type draft-mtp gets measured")),
    _h("H-PARALLEL-1",
       "One server slot (-np 1): one agent, and every extra slot would multiply the recurrent state and split the context.",
       "untested",
       (None, "llama-server's own default is auto; not compared")),
    _h("H-FIT-OFF",
       "Pass --fit off so llama-server does not silently change the planned/tuned values: what is served is what was measured.",
       "derived",
       ("tests/fixtures/llama-server-help-b12207.txt", "--fit defaults to on in b12207 and adjusts unset arguments")),
    _h("H-UBATCH",
       f"Physical batch (-ub) {UBATCH}, llama.cpp's default and what the phase-1 sweeps used. It is not searched by `forge tune`.",
       "partially measured",
       ("docs/PROTOCOL.md", "phase 1 runs llama-bench -p 512 with the default ubatch"),
       ("tests/fixtures/llama-bench-sweepd-snapshot.jsonl",
        "ubatch 256 shrinks the scratch buffer enough for 18 layers (512: 16): tg 9.60 vs 9.13 tok/s at depth 8192, but prompt speed 151 vs 195 tok/s"),
       (None, "a lower ubatch would win the default objective on this machine (more generation speed, prompt speed still above the floor); "
              "it is a candidate axis for the autotuner, not yet implemented")),
    _h("H-CPU-ONLY",
       "No usable GPU, or the GPU plan is not possible: 0 GPU layers, FA off, f16 KV, physical-core threads.",
       "partially measured",
       ("tests/fixtures/llama-bench-sweepc-snapshot.jsonl", "ngl=0: 5.45 tok/s generation and 137 tok/s prompt at depth 8192 on the Ryzen 5 3600"),
       (None, "measured with llama-bench (Vulkan build, GPU idle), not as a served config on a machine without a GPU; only this CPU was measured")),
    _h("H-LOW-POWER",
       "On battery/UPS: half the physical cores, context capped at 8192, GPU layers as the normal plan. Rationale: generation is "
       "memory-bandwidth bound, so fewer threads cost little speed; a smaller context means less re-read work per agent turn.",
       "untested",
       ("docs/JOURNAL.md", "sweep A: 5 threads cost only 4% against 6 (8.00 vs 8.32 tok/s); nothing below 5 threads was measured"),
       (None, "no power draw was measured; whether the GPU is more or less energy-efficient per token than the CPU is unknown")),
    _h("H-UNKNOWN-FREE-VRAM",
       f"If only the OS (not llama.cpp) reported the GPU and free VRAM is unknown, assume {UNKNOWN_FREE_VRAM_FRACTION:.0%} of total.",
       "untested",
       (None, "guess; the llama.cpp --list-devices path always reports free VRAM")),
]}


# ---------------------------------------------------------------------------------
# model description (from the GGUF header)
# ---------------------------------------------------------------------------------

@dataclass
class ModelInfo:
    path: str
    fingerprint: str
    arch: str
    quant: str | None
    file_mib: float
    block_count: int
    trunk_layers: int
    n_embd: int
    n_head: int | None
    ctx_max: int | None
    nextn_layers: int
    mtp_tensors_present: bool
    attn_layers: list[int]
    kv_elems_per_token: list[int]      # per trunk layer: n_head_kv * (key_len + value_len), 0 for recurrent layers
    layer_bytes: list[int]
    output_bytes: int
    embd_bytes: int
    other_bytes: int
    state_bytes_per_recurrent_layer: int
    warnings: list[str] = field(default_factory=list)

    @property
    def recurrent_layers(self) -> list[int]:
        a = set(self.attn_layers)
        return [i for i in range(self.trunk_layers) if i not in a]

    def kv_bytes_per_token(self, kv: str, layers: list[int] | None = None) -> float:
        idx = range(self.trunk_layers) if layers is None else layers
        return sum(self.kv_elems_per_token[i] for i in idx) * KV_BYTES_PER_ELEM[kv]


def _per_layer(v: Any, n: int) -> list[int | None]:
    if isinstance(v, list):
        return [int(x) for x in v][:n] + [None] * max(0, n - len(v))
    return [None if v is None else int(v)] * n


def model_info(h: gguf.GGUFHeader) -> ModelInfo:
    warnings: list[str] = []
    arch = h.architecture or "unknown"
    block_count = int(h.arch_key("block_count", 0) or 0)
    nextn = int(h.arch_key("nextn_predict_layers", 0) or 0)
    blk_idx = sorted({int(t.name.split(".")[1]) for t in h.tensors if t.name.startswith("blk.") and t.name.split(".")[1].isdigit()})
    mtp_idx = [i for i in blk_idx if nextn and block_count - nextn <= i < block_count]
    trunk = block_count - nextn if nextn else block_count
    if blk_idx and not mtp_idx and max(blk_idx) + 1 != trunk:
        warnings.append(f"block_count - nextn = {trunk} but tensors cover blocks 0..{max(blk_idx)}; using {max(blk_idx) + 1}")
        trunk = max(blk_idx) + 1
    n_embd = int(h.arch_key("embedding_length", 0) or 0)
    n_head_raw = h.arch_key("attention.head_count")
    n_kv_raw = h.arch_key("attention.head_count_kv", n_head_raw)
    n_heads = _per_layer(n_head_raw, trunk)
    n_kvs = _per_layer(n_kv_raw, trunk)
    key_len = h.arch_key("attention.key_length")
    val_len = h.arch_key("attention.value_length", key_len)

    # which layers have a KV cache
    rec = h.arch_key("attention.recurrent_layers")
    interval = h.arch_key("full_attention_interval")
    if isinstance(rec, list) and rec:
        attn = [i for i in range(trunk) if i < len(rec) and not rec[i]]
    elif interval:
        attn = [i for i in range(trunk) if (i + 1) % int(interval) == 0]
    elif isinstance(n_kv_raw, list):
        attn = [i for i in range(trunk) if (n_kvs[i] or 0) > 0]
    elif arch in ("mamba", "mamba2", "rwkv6", "rwkv7") or n_kv_raw is None:
        attn = []
    else:
        attn = list(range(trunk))
    elems = []
    for i in range(trunk):
        if i not in attn or not n_kvs[i]:
            elems.append(0)
            continue
        kl = int(key_len) if key_len else (n_embd // (n_heads[i] or 1))
        vl = int(val_len) if val_len else kl
        elems.append(n_kvs[i] * (kl + vl))
    attn = [i for i in attn if elems[i]]

    # fixed-size recurrent state of one layer (qwen3next/qwen35 layout: f32 delta-net state + conv state)
    state = 0
    ssm_inner, ssm_state = h.arch_key("ssm.inner_size"), h.arch_key("ssm.state_size")
    if ssm_inner and ssm_state:
        kernel = int(h.arch_key("ssm.conv_kernel", 4))
        groups = int(h.arch_key("ssm.group_count", 1) or 1)
        state = int(ssm_state) * int(ssm_inner) * 4 + (kernel - 1) * (int(ssm_inner) + 2 * groups * int(ssm_state)) * 4

    lb = h.layer_bytes()
    layer_bytes = [lb.get(i, 0) for i in range(trunk)]
    nl = h.non_layer_bytes()
    output_bytes = sum(v for k, v in nl.items() if k.startswith("output"))
    embd_bytes = sum(v for k, v in nl.items() if k.startswith("token_embd"))
    if "output.weight" not in nl and embd_bytes:
        # tied embeddings: the output projection reuses token_embd; assume a copy lands in the output layer's buffer (conservative for VRAM)
        output_bytes += embd_bytes
        warnings.append("no output.weight (tied embeddings): charging a token_embd-sized output matrix to the GPU when the output layer is offloaded")
    other = sum(v for k, v in nl.items() if not k.startswith(("output", "token_embd")))
    return ModelInfo(
        path=h.path, fingerprint=h.fingerprint(), arch=arch, quant=h.quant_name, file_mib=h.file_size / MIB,
        block_count=block_count, trunk_layers=trunk, n_embd=n_embd,
        n_head=n_heads[0] if n_heads else None, ctx_max=h.arch_key("context_length"),
        nextn_layers=nextn, mtp_tensors_present=bool(mtp_idx) or any(".nextn." in t.name for t in h.tensors),
        attn_layers=attn, kv_elems_per_token=elems, layer_bytes=layer_bytes,
        output_bytes=output_bytes, embd_bytes=embd_bytes, other_bytes=other,
        state_bytes_per_recurrent_layer=state, warnings=warnings)


def load_model(path: str) -> ModelInfo:
    return model_info(gguf.read_header(path))


def mtp_status(m: ModelInfo) -> dict[str, Any]:
    declared = m.nextn_layers > 0
    return {"declared": declared, "tensors_present": m.mtp_tensors_present,
            "available": declared and m.mtp_tensors_present,
            "reason": ("no nextn_predict_layers in the header" if not declared else
                       "header declares %d MTP layer(s) but the file has no tensors for it" % m.nextn_layers
                       if not m.mtp_tensors_present else "declared and present")}


# ---------------------------------------------------------------------------------
# configuration + estimates
# ---------------------------------------------------------------------------------

@dataclass
class Config:
    ngl: int
    ctx: int
    threads: int
    threads_batch: int
    fa: bool
    kv: str = "f16"
    ubatch: int = UBATCH
    parallel: int = 1
    spec_type: str | None = None
    device: str | None = None     # e.g. "Vulkan0", only set when several GPUs are present

    def key(self) -> tuple:
        return (self.ngl, self.ctx, self.threads, self.fa, self.kv, self.ubatch, self.spec_type)


@dataclass
class Estimate:
    vram_weights: float = 0.0
    vram_kv: float = 0.0
    vram_state: float = 0.0
    vram_compute: float = 0.0
    ram_weights: float = 0.0
    ram_kv: float = 0.0
    ram_state: float = 0.0
    ram_overhead: float = 0.0

    @property
    def vram(self) -> float:
        return self.vram_weights + self.vram_kv + self.vram_state + self.vram_compute

    @property
    def ram(self) -> float:
        return self.ram_weights + self.ram_kv + self.ram_state + self.ram_overhead


def all_layers(m: ModelInfo) -> int:
    """The -ngl value meaning "everything, output layer included": 99, or more for models with that many blocks."""
    return max(99, m.trunk_layers + 1)


def gpu_layer_ids(m: ModelInfo, ngl: int) -> list[int]:
    """llama.cpp offloads the last `ngl` blocks (the output layer too once ngl > blocks)."""
    n = max(0, min(ngl, m.trunk_layers))
    return list(range(m.trunk_layers - n, m.trunk_layers))


def estimate(m: ModelInfo, cfg: Config) -> Estimate:
    gpu = set(gpu_layer_ids(m, cfg.ngl))
    out_on_gpu = cfg.ngl > m.trunk_layers
    e = Estimate()
    kv_tok = {i: m.kv_elems_per_token[i] * KV_BYTES_PER_ELEM[cfg.kv] * cfg.ctx for i in range(m.trunk_layers)}
    for i in range(m.trunk_layers):
        rec = (m.state_bytes_per_recurrent_layer * cfg.parallel) if m.kv_elems_per_token[i] == 0 else 0
        if i in gpu:
            e.vram_weights += m.layer_bytes[i] / MIB
            e.vram_kv += kv_tok[i] / MIB
            e.vram_state += rec / MIB
        else:
            e.ram_weights += m.layer_bytes[i] / MIB
            e.ram_kv += kv_tok[i] / MIB
            e.ram_state += rec / MIB
    e.ram_weights += (m.embd_bytes * EMBD_RESIDENT_FRACTION + m.other_bytes) / MIB
    if out_on_gpu:
        e.vram_weights += m.output_bytes / MIB
    else:
        e.ram_weights += m.output_bytes / MIB
    if gpu:
        e.vram_compute = COMPUTE_BASE_MIB
    e.ram_overhead = HOST_OVERHEAD_MIB
    return e


def vram_budget(free_mib: float) -> float:
    return free_mib * (1 - VRAM_MARGIN_FRACTION) - VRAM_MARGIN_MIB


def ram_budget(available_mib: float) -> float:
    return available_mib * RAM_USE_FRACTION


def max_layers(m: ModelInfo, base: Config, free_vram_mib: float) -> int:
    """Largest ngl (0..trunk+1) whose VRAM estimate fits the budget (H-LAYERS-VRAM-FIT)."""
    budget = vram_budget(free_vram_mib)
    best = 0
    for ngl in range(m.trunk_layers + 2):
        c = Config(**{**asdict(base), "ngl": ngl})
        if estimate(m, c).vram <= budget:
            best = ngl
    return best


# ---------------------------------------------------------------------------------
# the plan
# ---------------------------------------------------------------------------------

@dataclass
class Decision:
    rule: str
    text: str


@dataclass
class Plan:
    name: str
    config: Config
    estimate: dict[str, float]
    feasible: bool
    decisions: list[Decision]
    warnings: list[str]
    mtp: dict[str, Any]
    requested_ctx: int = CTX_TARGET

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def summary(self) -> str:
        c = self.config
        ngl = "all" if c.ngl >= 99 else str(c.ngl)
        return (f"[{self.name}] ngl={ngl} ctx={c.ctx} threads={c.threads} fa={'on' if c.fa else 'off'} kv={c.kv} "
                f"| est VRAM {self.estimate['vram']:.0f} MiB, RAM {self.estimate['ram']:.0f} MiB"
                f"{'' if self.feasible else ' | INFEASIBLE'}")


def pick_gpu(profile: dict[str, Any]) -> tuple[dict[str, Any] | None, float | None, list[Decision]]:
    """Choose the GPU with the most free VRAM; returns (gpu, free_mib, decisions)."""
    dec: list[Decision] = []
    cands = [g for g in profile.get("gpus") or [] if g.get("backend")]
    if not cands:
        return None, None, dec

    def free_of(g: dict[str, Any]) -> float | None:
        if g.get("free_mib") is not None:
            return float(g["free_mib"])
        if g.get("total_mib") is not None:
            dec.append(Decision("H-UNKNOWN-FREE-VRAM", f"{g['name']}: free VRAM unknown, assuming {UNKNOWN_FREE_VRAM_FRACTION:.0%} of {g['total_mib']} MiB"))
            return float(g["total_mib"]) * UNKNOWN_FREE_VRAM_FRACTION
        return None

    scored = [(free_of(g), g) for g in cands]
    scored = [(f, g) for f, g in scored if f is not None]
    if not scored:
        return None, None, dec
    f, g = max(scored, key=lambda x: x[0])
    return g, f, dec


def physical_threads(profile: dict[str, Any]) -> tuple[int, str]:
    cpu = profile.get("cpu") or {}
    if cpu.get("physical_cores"):
        return int(cpu["physical_cores"]), "physical cores"
    if cpu.get("logical_threads"):
        return max(1, int(cpu["logical_threads"]) // 2), "logical threads / 2 (physical core count unknown)"
    return max(1, (os.cpu_count() or 2) // 2), "os.cpu_count() / 2 (probe found no CPU info)"


def _solve(m: ModelInfo, mk: Any, fa_on: bool, kv_t: str, requested_ctx: int, free_vram: float | None,
           free_ram: float | None) -> tuple[Config, bool]:
    """Layers from VRAM, then the largest context whose RAM estimate fits (H-CTX-RAM).

    Tries contexts from the requested one downwards; for each, the GPU takes as many layers as fit, which
    gives the lowest possible RAM use at that context. Returns (config, ram_ok).
    """
    ctx = min(requested_ctx, int(m.ctx_max)) if m.ctx_max else requested_ctx
    while True:
        cfg = mk(fa_on, kv_t, ctx)
        cfg.ngl = max_layers(m, cfg, free_vram) if free_vram is not None else 0
        if free_ram is None or estimate(m, cfg).ram <= ram_budget(free_ram):
            return cfg, True
        if ctx <= CTX_MIN:
            return cfg, False
        ctx = max(CTX_MIN, ctx // 2)


def build_plan(profile: dict[str, Any], m: ModelInfo, mode: str = "gpu", ctx: int = CTX_TARGET, kv: str | None = None,
               fa: bool | None = None, enable_mtp: bool = False) -> Plan:
    """mode: 'gpu' (best use of the GPU if there is one), 'cpu', or 'low-power'."""
    if mode not in ("gpu", "cpu", "low-power"):
        raise ValueError(mode)
    if kv is not None and kv not in KV_BYTES_PER_ELEM:
        raise ValueError(f"unknown KV type {kv}")
    if kv not in (None, "f16") and fa is False:
        raise ValueError(f"a quantised V cache needs flash attention (fa=False with kv={kv})")
    dec: list[Decision] = []
    warn: list[str] = list(m.warnings)
    threads, why_t = physical_threads(profile)
    free_ram = (profile.get("ram") or {}).get("available_mib")
    gpu, free_vram, gdec = pick_gpu(profile)
    dec += gdec
    use_gpu = mode != "cpu" and gpu is not None and free_vram is not None
    backend = gpu["backend"] if gpu else None
    requested = ctx
    if mode == "low-power":
        requested = min(ctx, 8192)
        threads = max(1, threads // 2)
        dec.append(Decision("H-LOW-POWER", f"battery/UPS profile: {threads} threads, context capped at {requested}"))
    else:
        dec.append(Decision("H-THREADS-PHYSICAL", f"{threads} threads = {why_t}"))
    dec.append(Decision("H-CTX-TARGET", f"target context {requested}"))
    device = gpu["device"] if (use_gpu and len(profile.get("gpus") or []) > 1) else None

    def mk(fa_on: bool, kv_t: str, c: int) -> Config:
        return Config(ngl=0, ctx=c, threads=threads, threads_batch=threads, fa=fa_on, kv=kv_t, device=device)

    fv = free_vram if use_gpu else None
    fa_default = use_gpu and backend in ("CUDA", "ROCm", "Metal")   # H-FA-BACKEND
    if not use_gpu:
        fa_on = True if kv not in (None, "f16") else bool(fa)
        kv_t = kv or "f16"
        cfg, ram_ok = _solve(m, mk, fa_on, kv_t, requested, None, free_ram)
        dec.append(Decision("H-CPU-ONLY", "0 GPU layers" + ("" if mode == "cpu" else " (no usable GPU in the profile)")))
    else:
        if kv is not None:
            options = [(kv != "f16" or (fa_default if fa is None else fa), kv)]
        elif fa is not None:
            options = [(fa, "f16")]
        else:
            options = [(fa_default, "f16"), (True, "q8_0")]
        solved = [(_solve(m, mk, o[0], o[1], requested, fv, free_ram), o) for o in options]
        (cfg, ram_ok), (fa_on, kv_t) = solved[0]
        if len(solved) > 1:
            (cfg2, ok2), o2 = solved[1]
            if (ok2 and not ram_ok) or (ok2 and cfg2.ctx > cfg.ctx):
                cfg, ram_ok, (fa_on, kv_t) = cfg2, ok2, o2
                dec.append(Decision("H-KV-Q8-GAIN", f"f16 KV does not fit context {requested} in RAM, q8_0 does ({cfg2.ctx})"))
            elif cfg2.ngl - cfg.ngl >= Q8_KV_MIN_GAIN_LAYERS:
                cfg, ram_ok, (fa_on, kv_t) = cfg2, ok2, o2
                dec.append(Decision("H-KV-Q8-GAIN", f"q8_0 + flash attention fits {cfg2.ngl} layers vs {solved[0][0][0].ngl} with f16 (gain >= {Q8_KV_MIN_GAIN_LAYERS})"))
            else:
                dec.append(Decision("H-KV-Q8-GAIN", f"q8_0 would fit {cfg2.ngl} layers vs {cfg.ngl} with f16 (gain < {Q8_KV_MIN_GAIN_LAYERS}): keeping f16"))
        dec.append(Decision("H-FA-BACKEND", f"flash attention {'on' if fa_on else 'off'}, KV {kv_t} (backend {backend})"))
        shown = cfg.ngl
        dec.append(Decision("H-VRAM-HEADROOM", f"{free_vram - vram_budget(free_vram):.0f} MiB of {free_vram:.0f} MiB free kept unused"))
        dec.append(Decision("H-LAYERS-VRAM-FIT", f"{min(shown, m.trunk_layers)} of {m.trunk_layers} blocks"
                            f"{' + output layer' if shown > m.trunk_layers else ''} on {gpu['name']}: "
                            f"budget {vram_budget(free_vram):.0f} MiB of {free_vram:.0f} MiB free"))
        if cfg.ngl > m.trunk_layers:
            dec.append(Decision("H-OUTPUT-LAYER", f"all {m.trunk_layers} blocks + the output layer on the GPU"))
            cfg.ngl = all_layers(m)
        if cfg.ngl == 0:
            warn.append("no layer fits the free VRAM: this GPU plan is CPU-only")
    if not ram_ok:
        e = estimate(m, cfg)
        warn.append(f"RAM: needs ~{e.ram:.0f} MiB at ctx {cfg.ctx} but only {ram_budget(free_ram):.0f} MiB of {free_ram:.0f} MiB available "
                    "may be used; the model would be paged from disk (slow) or the OS may kill the server")
    if cfg.ctx != min(requested, int(m.ctx_max or requested)):
        dec.append(Decision("H-CTX-RAM", f"context reduced {requested} -> {cfg.ctx} to fit {free_ram:.0f} MiB of available RAM"))
    dec += [Decision("H-UBATCH", f"ubatch {cfg.ubatch}"), Decision("H-PARALLEL-1", "1 slot"), Decision("H-FIT-OFF", "--fit off")]
    mtp = mtp_status(m)
    if mtp["available"] and enable_mtp:
        cfg.spec_type = "draft-mtp"
    mtp["enabled"] = cfg.spec_type == "draft-mtp"
    dec.append(Decision("H-MTP", f"MTP: {mtp['reason']}; {'enabled' if mtp['enabled'] else 'not enabled'}"))
    est = estimate(m, cfg)
    return Plan(name=mode, config=cfg, estimate={**asdict(est), "vram": est.vram, "ram": est.ram}, feasible=ram_ok,
                decisions=dec, warnings=warn, mtp=mtp, requested_ctx=ctx)


def build_all(profile: dict[str, Any], m: ModelInfo, **kw: Any) -> dict[str, Plan]:
    return {mode: build_plan(profile, m, mode=mode, **kw) for mode in ("gpu", "cpu", "low-power")}


def choose_plan(profile: dict[str, Any], plans: dict[str, Plan]) -> tuple[str, str]:
    """Which named plan applies to the machine as it is now. Returns (name, reason)."""
    if (profile.get("power") or {}).get("on_battery"):
        return "low-power", "running on battery/UPS"
    gpu, free, _ = pick_gpu(profile)
    if gpu is None or plans["gpu"].config.ngl == 0:
        return "cpu", "no usable GPU" if gpu is None else "no GPU layer fits the free VRAM"
    return "gpu", "GPU available on mains power"


def explain(plan: Plan) -> str:
    lines = [plan.summary()]
    for d in plan.decisions:
        h = HEURISTICS.get(d.rule)
        lines.append(f"  - {d.rule} [{h.status if h else '?'}]: {d.text}")
    for w in plan.warnings:
        lines.append(f"  ! {w}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------------
# llama-server command line
# ---------------------------------------------------------------------------------

@dataclass
class Caps:
    """What this llama.cpp build's flags look like (read from --help; defaults = recent build)."""
    fa_takes_value: bool = True      # -fa on|off|auto (new) vs a bare -fa switch (old)
    fit: bool = True
    spec_type: bool = True
    threads_batch: bool = True

    @staticmethod
    def from_help(text: str) -> "Caps":
        import re
        return Caps(
            fa_takes_value=bool(re.search(r"--flash-attn\s+[\[<]on\|", text)) if "--flash-attn" in text else True,
            fit=bool(re.search(r"^\s*-fit,?\s", text, re.M)) or "--fit " in text,
            spec_type="--spec-type" in text,
            threads_batch="--threads-batch" in text)


def server_command(server_exe: str | list[str], model_path: str, cfg: Config, host: str = "127.0.0.1", port: int = 8080,
                   caps: Caps | None = None) -> list[str]:
    """The llama-server command line for `cfg`. `server_exe` may be a launcher prefix (list), e.g. [python, fake.py]."""
    caps = caps or Caps()
    a = ([server_exe] if isinstance(server_exe, str) else list(server_exe)) + ["-m", model_path, "--host", host, "--port", str(port),
         "-ngl", str(cfg.ngl), "-c", str(cfg.ctx), "-t", str(cfg.threads)]
    if caps.threads_batch:
        a += ["-tb", str(cfg.threads_batch)]
    if caps.fa_takes_value:
        a += ["-fa", "on" if cfg.fa else "off"]
    elif cfg.fa:
        a += ["-fa"]
    a += ["-ctk", cfg.kv, "-ctv", cfg.kv, "-ub", str(cfg.ubatch), "-np", str(cfg.parallel)]
    if cfg.device:
        a += ["--device", cfg.device]
    if caps.fit:
        a += ["--fit", "off"]
    if cfg.spec_type and caps.spec_type:
        a += ["--spec-type", cfg.spec_type]
    return a
