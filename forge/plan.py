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
VRAM_MARGIN_FRACTION = 0.05   # of free VRAM, kept unused ...
VRAM_MARGIN_MIB = 152         # ... plus this much: 324 MiB at 3443 MiB free, the middle of the (271, 378] window the load/fail facts leave
RAM_USE_FRACTION = 0.95       # of available RAM the plan may commit (see H-CTX-RAM: a CPU-only run completed at 93%)
EMBD_RESIDENT_FRACTION = 0.0  # share of token_embd counted as resident: it is read by row, so mmap only pages in what is used
HOST_OVERHEAD_MIB = 512       # process, tokenizer, runtime, staging buffers
CTX_TARGET = 16384
CTX_MIN = 4096
Q8_KV_MIN_GAIN_LAYERS = 6     # q8_0 KV (needs flash attention) must buy at least this many GPU layers
UBATCH = 512
UBATCH_CHOICES = (512, 256)   # what `forge tune` searches; 256 is the one sweep D measured
UNKNOWN_FREE_VRAM_FRACTION = 0.80

# Compute-buffer model (H-COMPUTE-SCRATCH). Fitted to `llama-fit-params` b12207 on this model (results/fit-params-grid-host-20261108.txt).
KV_PAD = 256                  # llama.cpp pads the KV length to a multiple of this
DEFAULT_N_VOCAB = 262144      # used when the header has no vocabulary size: the upper end of current models (conservative)
LOGITS_EXTRA_MIB_PER_TOKEN = 0.0625   # hidden-state scratch per ubatch token on top of the f32 logits row (n_vocab * 4 bytes)
KQ_SINGLE_MAX_MIB = 512       # attention scores up to this size are live once; above it the grid shows two copies (KQ and its softmax)
ATTN_OTHER_MIB_PER_UB_TOKEN = 0.18     # attention-phase scratch besides KQ: per ubatch token ...
ATTN_OTHER_MIB_PER_CTX_TOKEN = 0.0056  # ... and per context token
COMPUTE_MAX_UBATCH = 512      # the fit covers ubatch 128-512; at 1024 the buffer jumps (1.2-2.4 GiB) and is not modelled

# Fallback ladder (serve.py)
FALLBACK_FINE_STEPS = 6       # layer counts tried one at a time before the ladder takes bigger steps

HEURISTICS: dict[str, Heuristic] = {h.id: h for h in [
    _h("H-LAYERS-VRAM-FIT",
       f"GPU layers = the largest -ngl whose estimated VRAM (weights llama.cpp places on the GPU, H-OUTPUT-LAYER + KV of their attention layers + "
       f"recurrent state + the compute buffer of H-COMPUTE-SCRATCH, all at the SERVER context and ubatch) stays below free VRAM x (1-{VRAM_MARGIN_FRACTION}) - "
       f"{VRAM_MARGIN_MIB} MiB. The reserve (324 MiB at 3443 MiB free) is the one fitted number: six load/fail facts on the same card put it in (271, 378] MiB "
       "with the estimate above ((271, 389] with llama.cpp's own numbers), and it covers what llama-fit-params does not count (driver and per-process allocations) as well as drift: "
       "llama-bench ctx 8704 ubatch 512 loads 16 and fails 17; ubatch 256 loads 18 and fails 19; llama-server ctx 16384 ubatch 512 loads 14 and fails 15.",
       "partially measured",
       ("results/fit-params-grid-host-20261108.txt",
        "llama-fit-params b12207, -fit off -fitp on: model / context / compute MiB per device for 166 (ngl, ctx, ubatch, FA, KV) points; "
        "ngl=15 ctx=16384 ubatch=512 gives compute 673 MiB = the 705695744-byte allocation that failed in llama-server"),
       ("tests/fixtures/llama-bench-sweepd-snapshot.log",
        "sweep D FA off f16 at ~8.7K context: ubatch 512 loads 16 and fails 17 (at depth); ubatch 256 loads 17 and 18 and fails 19"),
       ("tests/fixtures/llama-bench-sweepc-snapshot.log", "sweep C: ngl=16 loads for FA off/on and f16/q8_0, ngl=20 fails to create the context, ngl>=24 fails to load"),
       ("results/phase2.log", "llama-server 16K FA off f16 ubatch 512: ngl=15 fails to load, ngl=14 loads"),
       ("results/serve-smoke-host-20261108-before-fix.log", "forge serve planned 15 layers and failed with 'Device memory allocation of size 705695744 failed'"),
       (None, "one machine, one GPU driver, one model and one llama.cpp build; the 118 MiB-wide window is not narrowed by more data on this card; "
              "the reserve is a flat fraction + constant, which is a convenience, not a derivation")),
    _h("H-COMPUTE-SCRATCH",
       "GPU compute buffer = max(logits phase, attention phase). Logits phase: ubatch x (n_vocab x 4 B + 0.0625 MiB). Attention phase (flash attention off only): "
       f"copies x n_kv x ubatch x n_head x 4 B + {ATTN_OTHER_MIB_PER_UB_TOKEN} MiB x ubatch + {ATTN_OTHER_MIB_PER_CTX_TOKEN} MiB x ctx, with n_kv = ctx padded to {KV_PAD}, "
       f"ubatch = min(ctx, ubatch), copies = 1 up to {KQ_SINGLE_MAX_MIB} MiB of scores and 2 above. Flash attention: logits phase only. It does not depend on -ngl (once at least one layer is offloaded). "
       f"Valid for ubatch <= {COMPUTE_MAX_UBATCH}.",
       "partially measured",
       ("results/fit-params-grid-host-20261108.txt",
        "ground truth from llama.cpp's own estimate on the host: ubatch 512 FA off: 517 MiB at ctx 2K-8.7K, 545 at 12K, 673 at 16K, 1745 at 24K, 2305 at 32K; "
        "ubatch 256: 258 / 293 / 377 / 593 / 777; FA on: 517 (ubatch 512) and 258 (256) at every context. The model is within -36/+36 MiB of every point with up to 20 layers on "
        "the GPU and ubatch <= 512 (+100 at -ngl 99, where llama.cpp's own buffer shrinks)"),
       ("tests/fixtures/llama-bench-sweepd-snapshot.log", "a failed allocation of 271058944 B (258.5 MiB) in sweep D = fit-params' 258 MiB for FA on, ubatch 256"),
       ("results/serve-smoke-host-20261108-before-fix.log", "the failed server allocation of 705695744 B = 673 MiB = fit-params at ctx 16384, ubatch 512, FA off"),
       (None, "the shape (logits row, KQ matrix) is read off one model (n_vocab 248320, 16 heads, f16 KV); the step from one to two score copies between 512 and 768 MiB, "
              "the 1.0-1.1 GiB jumps at ubatch 1024 and the non-monotone values past 16K come from llama.cpp's graph allocator and are not explained; "
              "other models and other backends (CUDA, Metal) untested")),
    _h("H-VRAM-HEADROOM",
       f"Never plan to the last MiB: keep {VRAM_MARGIN_FRACTION:.0%} of free VRAM + {VRAM_MARGIN_MIB} MiB unused (see H-LAYERS-VRAM-FIT for how the number was fitted). "
       "Free VRAM moves while the machine is used, and a drop larger than this is NOT covered: `forge serve` re-probes at every start, "
       "re-checks the configuration against the free VRAM it sees, and steps down one layer at a time if the load still fails (H-FALLBACK-STEP).",
       "partially measured",
       ("tests/fixtures/list-devices-rx6500xt-idle.txt", "3443 MiB free on the idle card"),
       ("tests/fixtures/list-devices-rx6500xt-busy.txt", "3105 MiB free minutes later with a benchmark running: 338 MiB (10%) gone, more than the 324 MiB reserved"),
       (None, "the reserve is fitted to load/fail facts at one free-VRAM value (3443 MiB); how it should scale with free VRAM or card size is untested")),
    _h("H-OUTPUT-LAYER",
       "llama.cpp numbers n_layer block slots (the header's block_count, an absent MTP block included) plus the output layer and offloads the last N of them: "
       "-ngl 1 puts only the output matrix (796 MiB here) on the GPU, -ngl N puts blocks >= n_layer + 1 - N. On this model -ngl N means N-2 real blocks plus the output matrix. "
       "token_embd stays in RAM at any -ngl.",
       "measured",
       ("results/fit-params-grid-host-20261108.txt",
        "Vulkan0 model MiB: ngl=1 -> 795 (output only), ngl=2 -> 795 (the empty MTP slot), ngl=4 -> 1018 (2 blocks), ngl=16 -> 2381 (14 blocks), "
        "ngl=33 -> 4314, ngl=99 -> 4429 (everything but the 515 MiB token_embd, which is listed under Host)"),
       (None, "one model and one llama.cpp build; models without an MTP slot follow the same rule in llama.cpp's source but were not run here")),
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
    _h("H-SPEC-NONE",
       "Speculative decoding is off by default (--spec-type none is passed explicitly). Opt-in with `--spec ngram-mod` (plan, serve). "
       "Measured on llama-server 16K, FA off, f16, 14 layers, 4 prompts x 2 repetitions, temperature 0: ngram-mod reached up to 17.5 tok/s on a rewrite of code it had seen "
       "(mean 10.75 vs 9.03), but 2 of 8 outputs differ from the non-speculative ones, so it is not a free speed-up; ngram-simple drafts rarely (-3%), ngram-cache is slower "
       "and differs in half the outputs; draft-mtp aborts because the GGUF has no MTP tensors.",
       "measured",
       ("results/phase2.log", "per-arm loads and the draft-mtp failure"),
       ("results/phase2-spec.jsonl", "raw rows: tok/s, drafts accepted, output hashes per spec type"),
       ("docs/JOURNAL.md", "'phase 2: speculative decoding on llama-server' table and readings"),
       (None, "whether ngram-mod lowers agent pass rates is untested (PROTOCOL.md phases 3-4); one model, one backend")),
    _h("H-FALLBACK-STEP",
       f"If llama-server fails to load, retry with ONE layer fewer, for up to {FALLBACK_FINE_STEPS} steps, then in steps of 4 layers down to 0, then with a halved context "
       "(CPU only). The estimate is off by about one layer when it is wrong at all, so the first retries must not skip layers.",
       "partially measured",
       ("results/serve-smoke-host-20261108-before-fix.log",
        "the old ladder went 15 -> 11 although 14 loads (phase 2): it gave up 3 layers, ~0.9 tok/s"),
       (None, f"the {FALLBACK_FINE_STEPS} fine steps and the step of 4 after them are judgements; no case with an error of more than 1 layer has been seen")),
    _h("H-TUNE-LAYERS-MONOTONE",
       "For a given attention/KV type and ubatch, more layers on the GPU is not slower, so `forge tune` measures the most layers the server can hold and goes "
       "lower (1, 2, 4 below) only if that count fails to load; once a setting has run, its lower counts are skipped.",
       "partially measured",
       ("tests/fixtures/llama-bench-sweepc-snapshot.jsonl",
        "generation tok/s at depth 8192, FA off f16: ngl=0 5.45, 8 6.78, 12 7.65, 16 9.15; prompt tok/s at depth 137 -> 195"),
       ("tests/fixtures/llama-bench-sweepd-snapshot.jsonl", "ubatch 256, FA off f16, generation tok/s at depth 8192: 16 layers 9.04, 17 layers 9.19, 18 layers 9.60"),
       (None, "four layer counts on one machine; a GPU slower than the CPU for this model, or a PCIe-bound card with a tiny model, could break it")),
    _h("H-SERVER-RECHECK",
       "A tuned configuration is an upper bound for the server, not a promise: llama-bench runs at depth + 640 tokens of context (8832) while llama-server allocates "
       "for the full context, so serve re-estimates the tuned layer count against the free VRAM at the server's context and ubatch, and re-plans (keeping the tuned "
       "attention, KV type and ubatch) if it does not fit. The tuner only measures layer counts the server estimate allows.",
       "measured",
       ("results/forge-tune-host-20261108.json", "the first host tune chose 15 layers (measured at 8832 tokens of context) -- the count that llama-server could not load at 16K"),
       ("results/phase2.log", "15 layers fail, 14 load, at 16K"),
       (None, "the rule is only as good as the server estimate (H-LAYERS-VRAM-FIT); a context change at serve time is re-estimated the same way")),
    _h("H-PARALLEL-1",
       "One server slot (-np 1): one agent, and every extra slot would multiply the recurrent state and split the context.",
       "untested",
       (None, "llama-server's own default is auto; not compared")),
    _h("H-FIT-OFF",
       "Pass --fit off so llama-server does not silently change the planned/tuned values: what is served is what was measured.",
       "derived",
       ("tests/fixtures/llama-server-help-b12207.txt", "--fit defaults to on in b12207 and adjusts unset arguments")),
    _h("H-UBATCH",
       f"Physical batch (-ub): the plan uses {UBATCH} (llama.cpp's default); `forge tune` also measures {UBATCH_CHOICES[1]} and the server gets whatever the "
       "tuned configuration says. A smaller ubatch shrinks the compute buffer (H-COMPUTE-SCRATCH) and buys layers.",
       "partially measured",
       ("docs/PROTOCOL.md", "phase 1 runs llama-bench -p 512 with the default ubatch"),
       ("tests/fixtures/llama-bench-sweepd-snapshot.jsonl",
        "ubatch 256 loads 18 layers (512: 16): tg 9.60 vs 9.13 tok/s at depth 8192, prompt speed 151 vs 195 tok/s"),
       ("results/fit-params-grid-host-20261108.txt", "compute buffer at ctx 16384, FA off: 673 MiB at ubatch 512, 377 MiB at 256, 229 MiB at 128"),
       (None, "ubatch 128 and 1024 were never run through llama-bench or llama-server; 256 gains 2 layers at the cost of a quarter of the prompt speed, "
              "and whether that wins depends on the objective's floor (default 100 tok/s)")),
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
    n_vocab: int = 0                   # 0 = the header did not say

    @property
    def layer_slots(self) -> int:
        """llama.cpp's n_layer: every block slot the header declares, an MTP slot included even when its tensors are absent."""
        return max(self.block_count, self.trunk_layers)

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


def _n_vocab(h: gguf.GGUFHeader) -> int:
    v = h.arch_key("vocab_size")
    if v:
        return int(v)
    t = h.metadata.get("tokenizer.ggml.tokens")
    if isinstance(t, gguf.ArrayStub):
        return t.length
    return len(t) if isinstance(t, list) else 0


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
        n_head=max((x for x in n_heads if x), default=None), ctx_max=h.arch_key("context_length"),
        nextn_layers=nextn, mtp_tensors_present=bool(mtp_idx) or any(".nextn." in t.name for t in h.tensors),
        attn_layers=attn, kv_elems_per_token=elems, layer_bytes=layer_bytes,
        output_bytes=output_bytes, embd_bytes=embd_bytes, other_bytes=other,
        state_bytes_per_recurrent_layer=state, warnings=warnings, n_vocab=_n_vocab(h))


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
    """The -ngl value meaning "everything, output layer included": 99, or more for models with that many block slots."""
    return max(99, m.layer_slots + 1)


def gpu_layer_ids(m: ModelInfo, ngl: int) -> list[int]:
    """Transformer blocks llama.cpp puts on the GPU for -ngl N (H-OUTPUT-LAYER).

    llama.cpp numbers n_layer block slots plus one output slot and offloads the LAST `ngl` of the n_layer + 1 slots:
    the output layer first, then the final block slot. A declared-but-absent MTP block still owns a slot, so on this
    model -ngl 2 puts nothing but the output matrix on the GPU. Blocks on the GPU are slots >= n_layer + 1 - ngl.
    """
    if ngl <= 0:
        return []
    start = max(0, m.layer_slots + 1 - ngl)
    return list(range(start, m.trunk_layers))


def compute_mib(m: ModelInfo, cfg: Config) -> float:
    """GPU compute (graph scratch) buffer of the server's worst-case graph, in MiB (H-COMPUTE-SCRATCH).

    llama.cpp reserves the buffer for a graph of n_tokens = min(n_ctx, n_ubatch). Its peak is either the logits phase
    (one f32 row of n_vocab per token) or, with flash attention off, the attention phase: the n_kv x n_tokens x n_head
    f32 score matrix plus scratch. Flash attention never materialises the scores, so the context length stops mattering.
    """
    if cfg.ubatch > COMPUTE_MAX_UBATCH:
        raise ValueError(f"the compute-buffer model is fitted for ubatch <= {COMPUTE_MAX_UBATCH}, got {cfg.ubatch}")
    ub = min(cfg.ubatch, cfg.ctx)
    vocab = m.n_vocab or DEFAULT_N_VOCAB
    logits_phase = ub * (vocab * 4 / MIB + LOGITS_EXTRA_MIB_PER_TOKEN)
    if cfg.fa or not m.attn_layers:
        return logits_phase
    n_kv = -(-cfg.ctx // KV_PAD) * KV_PAD
    scores = n_kv * ub * (m.n_head or 32) * 4 / MIB
    copies = 1 if scores <= KQ_SINGLE_MAX_MIB else 2
    return max(logits_phase, copies * scores + ATTN_OTHER_MIB_PER_UB_TOKEN * ub + ATTN_OTHER_MIB_PER_CTX_TOKEN * cfg.ctx)


def estimate(m: ModelInfo, cfg: Config) -> Estimate:
    gpu = set(gpu_layer_ids(m, cfg.ngl))
    out_on_gpu = cfg.ngl > 0
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
    if cfg.ngl > 0:
        e.vram_compute = compute_mib(m, cfg)
    e.ram_overhead = HOST_OVERHEAD_MIB
    return e


def vram_budget(free_mib: float) -> float:
    return free_mib * (1 - VRAM_MARGIN_FRACTION) - VRAM_MARGIN_MIB


def ram_budget(available_mib: float) -> float:
    return available_mib * RAM_USE_FRACTION


def max_layers(m: ModelInfo, base: Config, free_vram_mib: float) -> int:
    """Largest ngl (0..layer slots + 1) whose VRAM estimate fits the budget (H-LAYERS-VRAM-FIT)."""
    budget = vram_budget(free_vram_mib)
    best = 0
    for ngl in range(m.layer_slots + 2):
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
        return (f"[{self.name}] ngl={ngl} ctx={c.ctx} threads={c.threads} fa={'on' if c.fa else 'off'} kv={c.kv} ub={c.ubatch} "
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


SPEC_OPT_IN = ("ngram-mod",)   # speculative-decoding types a user may ask for with --spec (H-SPEC-NONE)


def build_plan(profile: dict[str, Any], m: ModelInfo, mode: str = "gpu", ctx: int = CTX_TARGET, kv: str | None = None,
               fa: bool | None = None, enable_mtp: bool = False, ubatch: int = UBATCH, spec: str | None = None) -> Plan:
    """mode: 'gpu' (best use of the GPU if there is one), 'cpu', or 'low-power'.

    `spec` is the opt-in speculative-decoding type (None = off, which is the default); `ubatch` the physical batch."""
    if mode not in ("gpu", "cpu", "low-power"):
        raise ValueError(mode)
    if spec is not None and spec not in SPEC_OPT_IN:
        raise ValueError(f"unsupported speculative type {spec!r} (opt-in types: {', '.join(SPEC_OPT_IN)})")
    if not 1 <= ubatch <= COMPUTE_MAX_UBATCH:
        raise ValueError(f"ubatch must be 1..{COMPUTE_MAX_UBATCH} (the compute-buffer model is not fitted above)")
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
        return Config(ngl=0, ctx=c, threads=threads, threads_batch=threads, fa=fa_on, kv=kv_t, ubatch=ubatch, device=device)

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
        dec.append(Decision("H-LAYERS-VRAM-FIT", f"-ngl {shown}: {len(gpu_layer_ids(m, shown))} of {m.trunk_layers} blocks on {gpu['name']}: "
                            f"budget {vram_budget(free_vram):.0f} MiB of {free_vram:.0f} MiB free, ctx {cfg.ctx}, ubatch {cfg.ubatch}"))
        dec.append(Decision("H-COMPUTE-SCRATCH", f"compute buffer {compute_mib(m, cfg):.0f} MiB at ctx {cfg.ctx}, ubatch {cfg.ubatch}, "
                            f"flash attention {'on' if cfg.fa else 'off'}"))
        if cfg.ngl > m.layer_slots:
            dec.append(Decision("H-OUTPUT-LAYER", f"all {m.trunk_layers} blocks + the output layer on the GPU"))
            cfg.ngl = all_layers(m)
        elif cfg.ngl > 0:
            dec.append(Decision("H-OUTPUT-LAYER", f"-ngl {cfg.ngl} = {len(gpu_layer_ids(m, cfg.ngl))} blocks + the output matrix "
                                f"({m.output_bytes / MIB:.0f} MiB, offloaded first)"))
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
    elif spec:
        cfg.spec_type = spec
    mtp["enabled"] = cfg.spec_type == "draft-mtp"
    dec.append(Decision("H-MTP", f"MTP: {mtp['reason']}; {'enabled' if mtp['enabled'] else 'not enabled'}"))
    dec.append(Decision("H-SPEC-NONE", f"--spec-type {cfg.spec_type or 'none'}" + (" (opt-in)" if cfg.spec_type else " (default)")))
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
    if caps.spec_type:
        a += ["--spec-type", cfg.spec_type or "none"]       # explicit: what is served is what was measured (H-SPEC-NONE)
    return a
