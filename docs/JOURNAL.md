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

*Update, same day: the plan numbers below (15 layers, a fitted 1100 MiB scratch, "the output layer goes to the GPU only with every block", "`forge tune` does not search ubatch yet") were found wrong when the server was finally started; the entry "fixing the server plan" at the end of this file has the corrected model and what replaced it.*

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

## 2026-11-08 — phase 2: speculative decoding on llama-server

`02-bench/phase2-spec.ps1`. Raw: `results/phase2-spec.jsonl`, `results/phase2.log`. Server: 16K context, FA off, f16 KV, ubatch 512, 6 threads. 4 prompts (three code, one prose), 320 tokens max, temperature 0, seed 1, prompt cache off, 2 repetitions per prompt.

**Layer limit in the server is lower than in llama-bench.** 15 layers fail at load with `ErrorOutOfDeviceMemory` on a 705 MB compute buffer (the server sizes it for the full 16K context; llama-bench does not), 14 load. The `forge` planner (PR #2) picked 15, so its server plan is one layer too optimistic; its fallback dropped to 11.

| spec type | mean gen tok/s | range | drafts accepted | output identical to `none` |
|---|---|---|---|---|
| none (14 layers) | 9.03 | 8.92–9.20 | — | 8/8 (rep 1 = rep 2 for all prompts) |
| ngram-simple | 8.75 | 8.36–9.08 | 28/186 | 8/8 |
| ngram-mod | 10.75 | 8.89–**17.49** | 520/726 | 6/8 |
| ngram-cache | 7.30 | 5.32–11.02 | 990/3007 | 4/8 |
| draft-mtp | — | — | — | server aborts |

Readings:
- `draft-mtp` aborts with `GGML_ASSERT(layer.nextn.eh_proj && "MTP block missing nextn.eh_proj")`. The GGUF header declares one MTP layer but the conversion dropped its tensors (427 tensors, blocks 0–31). MTP needs a GGUF converted with the MTP block kept; not available for this model, and converting from the 19 GB BF16 original does not fit on the host disk today.
- `ngram-mod` learns from earlier requests: on the first pass of each prompt it drafted nothing (8.9 tok/s); on the second pass it drafted from what it had seen and reached 17.5 tok/s on the "rewrite this function" prompt (prompt 1), 9.6–11.7 on the others. An agent re-reads and rewrites the same code across turns, which is the case this helps.
- **It is not output-identical.** Two of the eight `ngram-mod` outputs differ from `none` at temperature 0 (prompts 2 and 3, second pass; prompt 3 even changes length, 225 vs 198 tokens). `none` itself is deterministic across repetitions, so the difference comes from verifying drafts in a batch: the GPU computes a batch of tokens with slightly different rounding than one token at a time, and a near-tie in the logits flips. The protocol called any difference a bug; the honest reading is that speculative decoding on this backend changes outputs, so it is a trade-off whose effect on correctness must be measured (phase 3), not assumed.
- `ngram-cache` is slower than no speculation and differs in half the outputs: rejected.
- `ngram-simple` drafts rarely and costs ~3%: rejected.

Decision for now: default `--spec-type none`; `ngram-mod` is an arm in phase 3 and 4 (quality and agent success), adopted only if it does not lower pass rates.

## 2026-11-08 — fixing the server plan (offload order, compute buffer, one-layer fallback)

**What failed.** `forge serve --once` on the host planned 15 layers, 16K, FA off, f16, ubatch 512. llama-server died at load with `ggml_vulkan: Device memory allocation of size 705695744 failed ... failed to allocate compute pp buffers` (`results/llama-server-fail-ngl15-ctx16384-host-20261108.log`); the old ladder then jumped 15 -> 11, which served at 8.0 tok/s. Phase 2 had already measured the real limit: 15 fail, 14 load, 9.03 tok/s. The planner's one scratch constant (1100 MiB) was fitted on llama-bench, which allocates for a context of ~8.7K; the server allocates for 16K.

**Ground truth.** `llama-fit-params.exe` (in the b12207 folder) prints llama.cpp's own estimate of the model, context and compute buffers per device for given flags, without loading anything (`-fit off -fitp on`, 0.5 s per call, GPU idle: `sweep-c.log` ended in `exit=`, no llama process, no Ollama model loaded). 166 calls over (ngl, context, ubatch, FA, KV): `02-bench/fit-params-grid.ps1` -> `results/fit-params-grid-host-20261108.txt`. Two numbers tie it to the failures: ngl 15, 16K, ubatch 512, FA off gives a 673 MiB compute buffer = the 705695744 bytes that failed; FA on, ubatch 256 gives 258 MiB = the 271058944 bytes that failed in sweep D. So the tool's numbers are what llama.cpp allocates.

**Three things the grid says that the planner had wrong.**
1. *Offload order.* llama.cpp numbers the header's `block_count` slots (33 here: 32 blocks plus the MTP slot whose tensors are not in the file) plus the output layer and offloads the last N: `-ngl 1` puts only the output matrix (796 MiB) on the GPU, `-ngl 2` adds the empty MTP slot, `-ngl N` is N-2 blocks plus the output matrix. The old planner charged N blocks and no output matrix; the 1100 MiB "scratch" had absorbed about 600 MiB of that. Weights and KV+state in the new estimator agree with llama.cpp's own numbers to 2 MiB on all 135 grid points with ubatch <= 512.
2. *The compute buffer is a function of context and ubatch.* Without flash attention it is max(logits phase, attention phase): logits ubatch x (n_vocab x 4 B + 64 KiB), attention n_kv x ubatch x n_head x 4 B of scores plus scratch. With flash attention only the logits phase is left.

| compute buffer, MiB | ctx 2K-8.7K | 12K | 16K | 24K | 32K |
|---|---|---|---|---|---|
| FA off, ubatch 512 | 517 | 545 | **673** | 1745 | 2305 |
| FA off, ubatch 256 | 258 | 293 | 377 | 593 | 777 |
| FA on, ubatch 512 / 256 | 517 / 258 | same | same | same | same |

   Past 16K the buffer holds two copies of the scores (KQ and its softmax) at ubatch 512; why the step comes between 512 and 768 MiB of scores is llama.cpp's graph allocator, not explained here. At ubatch 1024 the buffer jumps again (1.2-2.4 GiB) and is not modelled (the planner refuses ubatch > 512). The fitted model is within -36/+36 MiB of every point with up to 20 layers on the GPU, +100 MiB at `-ngl 99`.
3. *One reserve fits all six load/fail facts.* On 3443 MiB free, with llama.cpp's own model+context+compute numbers:

| run | context | ubatch | loads (MiB needed) | fails (MiB needed) |
|---|---|---|---|---|
| llama-bench | 8704 | 512 | 16 (3054) | 17 (3173) |
| llama-bench | 8704 | 256 | 18 (3031) | 19 (3172) |
| llama-server | 16384 | 512 | 14 (3041) | 15 (3212) |

   Loads need reserve <= free - needed, fails need reserve > free - needed: the reserve lies in (271, 389] MiB. The planner uses 5% of free + 152 MiB = 324 MiB at 3443 free, the middle of the window with its own estimate (271, 378]. One model and one constant cover bench and server; there is no separate bench reservation. The reserve is a fit to one card at one free-VRAM value: it stands for driver and per-process allocations that fit-params does not count, and drift is not covered by it any more (the old claim that the plan survives the 338 MiB seen between the idle and busy readings does not hold: the idle plan needs 3065 MiB plus at least 271 reserve, the busy card had 3105 free). What keeps the server loading under drift is the re-probe at every start and the one-layer fallback.

**What changed.**
- `plan`: placement as above, `compute_mib(model, cfg)` charged for the server's context and ubatch, `--ubatch`, `--spec`. Host plan now: **14 layers**, 16K, FA off, f16, ubatch 512 (`results/forge-plan-host-20261108-fixed.txt`); with ubatch 256 it gives 16 (sweep D: 18 vs 16 at 8.7K); at 32K ubatch 512 leaves no layer on the GPU and 256 leaves 12.
- `serve`: a tuned result is an upper bound, not a promise (llama-bench runs at depth + 640 tokens): it is re-estimated at the server's context and ubatch against the free VRAM and RAM now; if it does not fit, the plan is rebuilt keeping the tuned attention, KV type and ubatch. The fallback ladder goes one layer at a time for six steps, then steps of 4, then a smaller context on the CPU (20 attempts at most). The speculation type is a serve-time choice and is never tuned.
- `tune`: the search is attention/KV (3) x ubatch (512, 256), each at the most layers the server estimate allows, going lower only if that fails; once a setting has run, its lower counts are skipped (generation speed grows with layers on this machine: 5.45, 6.78, 7.65, 9.15 tok/s at 0, 8, 12, 16 layers; ubatch 256: 9.04, 9.19, 9.60 at 16, 17, 18). That is 6 measurements instead of 12-15, so the 20-minute budget covers the whole space. The cache key carries a search-space version, so the entry of the buggy first tune can never be read; the file `results/forge-tune-host-20261108.json` keeps its data and now carries a `_note` that says which version of the code produced it and why its 15 layers cannot be served.
- Speculative decoding: `--spec-type none` is passed explicitly (heuristic H-SPEC-NONE with the phase 2 numbers: ngram-mod up to 17.5 tok/s on code it had seen, mean 10.75 vs 9.03, 2 of 8 outputs differ at temperature 0; ngram-simple -3%, ngram-cache slower and differing in half; draft-mtp aborts, no MTP tensors). `--spec ngram-mod` is the only opt-in.

**On the host after the fix** (`results/serve-host-20261108-after-fix.log`, b12207, GPU free, Ollama unloaded; the server is stopped after each step):

| step | what it checks | planned / served | attempts | result |
|---|---|---|---|---|
| A `serve --once` | plan path | 14 layers, 16K, ubatch 512, `--spec-type none` | 1 (loaded) | healthy |
| B `serve --once` + the real result of the first tune (15 layers, llama-bench) replayed into a scratch cache | server re-check | `tuned configuration skipped: needs ~3236 MiB VRAM at ctx 16384, ubatch 512 (weights 2265, KV+state 275, compute 696), budget now 3119 MiB` -> 14 layers | 1 (loaded) | healthy |
| C long-running serve | one `/v1/chat/completions`, temperature 0, 146 tokens | 14 layers | 1 | **9.23 tok/s** generation (earlier runs of the same script: 9.28, 9.26) |
| D long-running serve + a synthetic entry, 16 layers at ubatch 256 | the planner's prediction, not used to fit anything | 16 layers, ubatch 256 | 1 (loaded) | **9.84 tok/s** (earlier runs: 9.84, 9.70) |

   D is an out-of-sample check: the model said 16 layers at ubatch 256 and 16K fit (3051 of 3119 MiB) before it was tried. The replays are marked as such in the cache entries (`replay` field) and in the log; B uses the real measured result, D has no measurement behind it. Before the fix the same machine served 8.0 tok/s at 11 layers (`results/serve-smoke-host-20261108-before-fix.log`).

**Not verified.** `forge tune` was not run again on the host (the allowance was one run); the new search order, the ubatch axis and the server bound are tested against the fake llama-bench only, with the real sweep D numbers in the fake's prompt-speed ratio. The reserve window is 118 MiB wide and the data on this card cannot narrow it; how it scales with free VRAM or card size is untested. The compute-buffer model is one model, one backend: CUDA, Metal, other architectures and ubatch 128/1024 are untested. The first launch of the host script was detached and died with the SSH session before any model load; the second ran without `FORGE_HOME` and left two server logs in `%LOCALAPPDATA%\forge` (a directory that did not exist; the two files and the directory were removed, nothing else was touched there); the later runs, with `FORGE_HOME` under `forge\cache`, behave the same; the last one, made after the review fixes below on the code that is committed, is the one in `results/`.

**Independent review, and what it found.** A fresh agent read the diff cold (no conclusions given) and found seven edge-case defects, none on the target host's path; six were fixed with regression tests and mutants: (1) the re-check judged a config pinned to one GPU by the roomiest GPU's free VRAM; (2) the weights of an MTP block were never charged when a file contains them (the slot offloaded right after the output matrix); (3) the fallback ladder was cut at 20 attempts for models of 53+ blocks, so its smaller-context rungs were unreachable (the coarse step now scales with depth); (4) `-ngl 2` and `-ngl 1` are the same placement on a model with an MTP slot and were tried twice; (5) the cache key depended on the order of `--ubatches`; (6) `plan --ubatch 1024` printed a traceback. The seventh is a fact the review read out of the grid, not a fix: **`-ngl 0` on a Vulkan build still reserves a 1312 MiB compute buffer on the GPU** (ubatch 512, 16K); `-dev none` moves it to the host (604 MiB) (`results/fit-params-ngl0-devnone-host-20261108.txt`, a dry run of fit-params). The planner charges a CPU plan 0 MiB of VRAM and `forge serve` does not pass `-dev none`, so the CPU rungs of the ladder still need ~1.3 GiB of free VRAM. Recorded in H-CPU-ONLY as a measured gap; not changed, because no CPU-only server was started to see that `-dev none` loads.

## 2026-11-08 — REA (Reverse Engineer Anything) on the host

Requested by the user as an essential "apocalypse" skill: understand software and files with no source and no internet.

Audit before installing (repo `morluto/rea`, MIT, npm `rea-agents` 6.8.0; full method in the Hermes skill `vetting-third-party-agent-tooling`):
- No `preinstall`/`postinstall`/`install` scripts in `package.json`; 42 runtime dependencies, Frida optional (skipped).
- URLs in runtime code are test fixtures (`example.test`, `127.0.0.1`); real npm/GitHub calls live only in release, update and CI scripts. The "telemetry" matches are local Hopper/Ghidra lifecycle logs, nothing sent out.
- `rea setup` edits every agent's configuration (it reports `config_drift` for Hermes, OpenCode, Gemini CLI, Copilot CLI): not run.
- Noted, not blocking: ~443k stars on a repo created 2026-04-14. The account (2020) and the code are coherent with the stated purpose.

Install (`tools/rea-install.ps1`): REA needs Node 22.23+/24.22+, the host has 20.9. A portable Node v24.22.2 (official zip, SHA-256 `ca1bff5d…99e2` matched) in `C:\Users\arthu\llm\rea\`, then `npm install --ignore-scripts --omit=optional rea-agents@6.8.0` into a local folder: 149 packages, 60 s, 136 MB, lockfile integrity equal to the registry's. Nothing global.

Tests on real local files, offline (`tools/rea-tests.ps1`, raw JSON in `results/rea/`):

| test | command | result |
|---|---|---|
| model file | `rea inspect-model-artifact <the DeepSeek heretic GGUF>` | 2 s. `qwen35`, `block_count = 33`, `nextn_predict_layers = 1`, tensors in blocks 0–31 only, **no `blk.32` / `nextn` tensors**. Independently confirms the MTP finding from `forge/gguf.py` and phase 2. It also lists its own limits (header only, weights not read). |
| native binary | `rea inspect-pe-imports llama-server.exe` | 1 s. Imports KERNEL32, the MSVC runtime and `llama-server-impl.dll`: the server is a thin launcher, the real code is in the DLL. |
| readiness | `rea doctor --json` | static analyzers ready; no native decompiler installed (Hopper, Ghidra, IDA, JADX, Kuna all missing), so `decompile` is not available yet. |

Next: Ghidra (offline, open source, needs a JDK) so REA can decompile native code; then register REA as an MCP server for the local agent (OpenCode + the 9B model) by hand, not through `rea setup`, and measure in phase 4 whether the small model can use REA's tools to answer a real question about a binary.
