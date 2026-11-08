# Journal

Times are the host's local time (UTC-3).

## 2026-11-08 — getting the model

- The Hugging Face repo the user picked holds BF16 safetensors (19 GB). Ollama cannot run that directly and the disk had 7.8 GB free, so we used the community GGUF quant of the same model.
- Quant choice: i1-IQ4_XS (5.2 GB), imatrix-weighted, the best quality per byte at this size. Q4_K_M would be ~5.6 GB for marginal gain.
- `01-model/setup.cmd` runs as a Windows scheduled task, so it survives an SSH drop: pull, `ollama create dsv4-9b-heretic -f Modelfile` (32K context, temperature 0.6, top_p 0.95), `npm i -g opencode-ai`.
- `ollama show` reports capabilities `tools`, `thinking`, `completion`: the model can be driven by an agent.

## 2026-11-08 — first agent run (Ollama defaults, 32K context)

Task: `03-agent/agent-test`, implement `parseDuration("1h30m") -> 5400` with TypeError/RangeError cases; 5 tests. A reference implementation passes 5/5, the stub fails 5/5.

Two false starts, both harness problems, not model problems:
1. PowerShell `Start-Process` mangled the quoted `opencode.cmd` path ("O sistema não pode encontrar o caminho especificado"). Fix: a plain `agent.cmd`.
2. An older OpenCode install on the same user had a database from another version ("Database is not empty and has no session table"). Fix: isolate the run with `XDG_DATA_HOME`/`XDG_CONFIG_HOME`/`XDG_CACHE_HOME`/`XDG_STATE_HOME` under `C:\Users\arthu\llm\ocdata`, leaving the user's own OpenCode data untouched.

Result (`results/agent-run1-dsv4.log`, code in `results/agent-run1-dsv4-duration.js`):

- 18 agent steps in ~60 min: 11 edits to `src/duration.js`, 5 runs of `node --test`, test file untouched (same SHA-256).
- Final state: **3/5 tests pass**. The model split the input on whitespace, so `1h30m` and a bare `120` are rejected, both of which the file's own header comment spells out. It ran the tests, saw the failures, and re-edited without fixing the cause.
- Each step took 3–5 min. Ollama placed 10 of 34 layers on the GPU (65%/35% CPU/GPU, 6.7 GB), 6 threads.
- Raw speed probe: 7.90 tok/s generation, 7.96 tok/s prompt eval on a short prompt.
- The run did not end on its own; the watchdog that enforced the 45-min limit had died with the gateway restart, and the process was killed by hand. Lesson: the time limit must live in the same process tree as the agent (the scheduled task), not in a remote poller.

Takeaway: tool use works; speed and debugging quality are the bottlenecks.

## 2026-11-08 — sweep A: GPU layers × context × threads (Ollama defaults otherwise)

`02-bench/bench.py a`, prompt of 2,624 tokens (an agent-sized turn), 192 generated tokens, temperature 0, thinking off. Model unloaded between configs so the options apply. Partial results (`results/bench-a.jsonl`):

| ctx | GPU layers | threads | split | prompt tok/s | gen tok/s |
|---|---|---|---|---|---|
| 16K | 8 | 6 | 69/31 | 142.8 | 7.29 |
| 16K | 12 | 6 | 60/40 | 154.4 | 7.60 |
| 16K | 16 | 6 | 51/49 | 165.3 | **8.32** |
| 16K | 20 | 6 | — | out of VRAM (HTTP 500) | |
| 16K | 24 | 6 | — | out of VRAM | |
| 32K | 8 | 6 | 68/32 | 142.6 | 6.98 |
| 32K | 12 | 6 | 59/41 | 150.4 | 8.07 |
| 32K | 16+ | 6 | — | out of VRAM | |
| 16K | 16 | 12 | 51/49 | 163.9 | 7.81 |
| 16K | 16 | 5 | 51/49 | 160.9 | 8.00 |

Readings so far:
- More layers on the GPU help until VRAM runs out: 16 layers at 16K, 12 at 32K. Past that Ollama fails to load instead of falling back.
- 6 threads (one per physical core) is the peak: 5 gives 8.00, 12 (SMT) gives 7.81, 6 gives 8.32 tok/s.
- Best so far: **16K context, 16 GPU layers, 6 threads: 8.32 tok/s generation, 165 tok/s prompt**, ~9% over Ollama's own pick (10 layers).
- The gain from 8 to 16 layers is only ~14% on generation: with PCIe x4 and half the model on the CPU, DDR4 bandwidth dominates.

Next: flash attention + q8_0 KV cache at the server level (frees VRAM for more layers or more context), then llama.cpp's `llama-server` directly to use the model's MTP layer for speculative decoding, which Ollama does not.

## 2026-11-08 — sweep B: flash attention + q8_0 KV cache (Ollama, server env vars)

`02-bench/sweep-b.ps1` sets `OLLAMA_FLASH_ATTENTION=1` and `OLLAMA_KV_CACHE_TYPE=q8_0`, restarts Ollama, re-runs the layer grid. Server log confirms `flash_attn = enabled`. Raw: `results/bench-b.jsonl`.

| ctx | GPU layers | split | prompt tok/s | gen tok/s |
|---|---|---|---|---|
| 16K | 12 | 60/40 | 33.2 | 7.13 |
| 16K | 16 | 51/49 | 125.6 | **9.19** |
| 16K | 20 | 42/58 | 76.8 | 8.96 |
| 16K | 24+ | — | out of VRAM | |
| 32K | 12 | 60/40 | 150.4 | 7.53 |
| 32K | 16 | 50/50 | 37.5 | 7.57 |
| 32K | 20+ | — | out of VRAM | |

Readings:
- Memory drops (6.0 → 5.7 GB at 16K) and one more layer group fits (20 at 16K, 16 at 32K).
- Best generation so far: 9.19 tok/s (16K, 16 layers), +10% over sweep A's best.
- Prompt speed is erratic within the same sweep: 33, 126, 77, 150, 37 tok/s with no monotone pattern. One sample per point cannot separate the configuration from noise (first-load effects, the host's other load). This is why phase 1 moves to `llama-bench` with 3 repetitions and a standard deviation.

## 2026-11-08 — the hardware-adaptive layer (`forge/`)

Goal (VISION.md): the agent must keep working on whatever hardware is there, offline. `forge/` is four stages in Python's standard library, run on both Windows and Linux: `probe` (OS, CPU, RAM, GPUs with backend and free VRAM, disk, battery), `plan` (a starting configuration from the GGUF header, nothing run), `tune` (llama-bench around the plan, cached by hardware + model file + llama.cpp build) and `serve` (llama-server, health check, fall back on failure). Every planning rule is a named heuristic with the measurement behind it or the word "untested" (`python -m forge heuristics`).

What the real model's header says (`tests/fixtures/gguf-dump-dsv4-9b.txt`, parsed on the host by `forge/gguf.py`):
- Qwen3.5 is a hybrid: only every 4th block (8 of 32) has a KV cache, so f16 KV costs 32 KiB per token (512 MiB at 16K, 272 MiB with q8_0), not the 128 KiB a dense 32-layer model would. The other 24 blocks keep a fixed state of 2.1 MiB each.
- Weights per block are 107 to 116 MiB; `output.weight` is 796 MiB and `token_embd.weight` 515 MiB.
- **The MTP head is not in this file.** The header says `nextn_predict_layers = 1`, but the tensor table has 427 tensors in blocks 0 to 31 and nothing for block 32. `forge plan` therefore reports MTP as declared but unavailable, and phase 2's `draft-mtp` arm should not be expected to work with this GGUF (not tried with llama-server).

llama.cpp b12207 Vulkan, sweeps C and D (the user's `sweep-c.ps1` / `sweep-d.ps1`: `llama-bench -p 512 -n 128 -d 0,8192 -r 3`; snapshots in `tests/fixtures/llama-bench-sweepc-snapshot.*` and `...sweepd-snapshot.*`, the raw files are on the host):

| GPU layers | FA off, f16 KV: gen tok/s at depth 8192 | FA on, f16 KV | FA on, q8_0 KV |
|---|---|---|---|
| 0 | 5.45 | 4.44 | 4.32 |
| 8 | 6.78 | 5.76 | 5.50 |
| 12 | 7.65 | 6.59 | 6.44 |
| 16 | **9.15** (prompt 195 tok/s) | 7.83 (prompt 84) | 7.45 (prompt 86) |

- Flash attention costs 13 to 19% of generation speed and halves prompt speed at depth on this build, with or without a GPU, and q8_0 KV (which needs it) is the slowest setting everywhere. So the plan defaults to FA off and f16 KV on Vulkan and CPU; q8_0 is only chosen when it buys 6 or more layers or f16 does not fit in RAM.
- The load boundary on 3443 MiB free: 16 layers load in all three settings; 20 fail to create the context in all three (24 and up fail to load). Sweep D, FA off, f16: with the default ubatch 512, 17 fails while filling the 8192 depth; with ubatch 256, 18 layers run (9.60 tok/s, prompt 151) and 19 fail. Ollama's "20 layers fit with FA + q8_0" (sweep B) did not reproduce in llama.cpp.
- A smaller ubatch buys layers: 18 layers at ubatch 256 beats 16 at 512 on generation (9.60 vs 9.13) and stays above the 100 tok/s prompt floor. `forge tune` does not search ubatch yet; this is the first thing to add.

Planner results on the host (profile from `forge probe`, GPU idle: 3443 MiB free, 5784 MiB RAM available, `plan --explain`):
- gpu: **15 layers, 16K context, FA off, f16 KV, 6 threads** (estimated 3080 MiB VRAM); with `--kv q8_0` it picks 16, which is the "~16 layers at 16K" of sweeps A and B. cpu: 8K context, feasible. low-power: 3 threads, 8K context, 16 layers.
- Checking the first version against the sweeps found two wrong rules, both fixed: the VRAM scratch constant was an assumed 256 MiB and the estimator chose 22 layers with q8_0 (the sweeps say 20 fail); the RAM rule called the CPU-only run infeasible although sweep C had just run it to completion at the same context (the whole token embedding was counted as resident and only 80% of available RAM was allowed). The constants now sit inside the windows the measurements leave open; both facts are tests.

`forge tune` on the host, run once (b12207, GPU idle: sweep C had written `exit=0` and sweep D had finished; budget 1200 s, depth 8192, `--reps 1`, prompt floor 100 tok/s; raw: `results/forge-tune-host-20261108.log` and `.json`). It measured 12 points in 1195 s (95 to 138 s each) and skipped 3 after out-of-memory:

| GPU layers | FA off, f16: gen / prompt tok/s at depth | FA on, f16 | FA on, q8_0 |
|---|---|---|---|
| 11 | 7.46 / 169 | 6.67 / 81 | 6.25 / 83 |
| 13 | 8.15 / 169 | 6.96 / 82 | 6.60 / 84 |
| 15 | **8.77 / 191** | 7.66 / 83 | 7.23 / 86 |
| 17 | out of memory | out of memory | out of memory |

- The plan's own point won: 15 layers, FA off, f16 KV, 6 threads, 16K context, 8.77 tok/s generation. The command line it printed is in the log. FA on and q8_0 are slower at every layer count and fall under the prompt floor, as in sweep C.
- The tuned result is about 4% short of what is known to be available: the grid (plan +/- 2, +/- 4) steps over 16 layers, which sweeps C and D measured at 9.13 to 9.15 tok/s (3 repetitions) and which sits just below the out-of-memory at 17. After the run, the search got a boundary refinement (after a failure at N, probe N-1 for the winning setting). It is tested against the fake llama-bench only; the host run was not repeated.
- The run also exposed a cache-key bug the fake could not: the build id came out as `b0` because b12207 prints `version: 0.9.0-dev (build 12207, commit 998200491)`. Fixed (the real output is a fixture, `tests/fixtures/llama-bench-version-b12207.txt`); the cache entry saved in `results/` still carries the wrong `b0`.
- One repetition per point means no standard deviation, so 8.77 (15 layers, this run) against 9.15 (16 layers, sweep C, 3 reps) is not a like-for-like comparison.

Tests and checks: 112 tests (standard-library `unittest`, about 30 to 50 s) pass on Linux with Python 3.11 and on the Windows 10 host with Python 3.12 (`results/forge-unittest-windows-host.txt`). They cover the GGUF parser on synthetic files, the probe parsers on recorded outputs, the plan arithmetic against hand computation that does not use the planner's own helpers, autotune against a fake llama-bench (canned JSONL, a real OOM stderr, a timeout, a time budget), cache invalidation, and serve against a fake llama-server (slow health, OOM at load, port in use).

`tests/mutation_check.py` breaks one rule at a time in a temporary copy: 15 mutants (VRAM budget, comparison, margin, KV, scratch, which end of the model is offloaded, serve ignoring a VRAM drop, three cache-key parts, the hardware fingerprint, a cache that never hits, no stop at OOM, no prompt floor, free/total swapped), 15 killed (`results/forge-mutation-check.txt`). The first version let two survive and the tests were wrong, not the code: every layer count I tried was a multiple of 4, the model's repeat period, so offloading the first n blocks instead of the last n weighed the same; and nothing tied the safety margin to the 338 MiB drift in the fixtures. Both are tests now. An independent read of the code then found 12 more defects, all fixed with regression tests: a CPU-tuned result that `serve` ignored, a time-truncated tune cached as final, a fallback ladder cut off before the smaller contexts, an already-answering port counted as a healthy start, a context reduced under old memory pressure served forever, layer counts measured at 8.8K context served at 16K, tied embeddings under-counted, the `-ngl 99` sentinel for 99+-block models, `-dev` not passed to llama-bench, an ARM `/proc/cpuinfo` trailer counted as a CPU, `--host 0.0.0.0` health checks, and a missing `wait()` after `kill()`.

Not verified: a real `llama-server` was never started by `forge serve` (the one-run allowance on the host was for `tune`; flags were only checked against the recorded b12207 `--help`); CUDA, ROCm and Metal paths, macOS, multi-GPU, battery behaviour and the low-power plan on real hardware; MTP (the header has no tensors for it); the planner on any machine or model but this one.
