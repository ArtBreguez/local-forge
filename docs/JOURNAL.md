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

## 2026-11-08 — first real `forge tune` with the fixed planner

`tools/tune2-host.ps1` (`forge tune --budget 2400 --reps 2`, then `forge serve` with the result and three temperature-0 completions). Raw: `results/forge-tune2-host-20261108.log`.

| ngl | FA | KV | ubatch | prompt tok/s at depth 8192 | gen tok/s at depth 8192 |
|---|---|---|---|---|---|
| 14 | off | f16 | 512 | 187.1 | 8.53 |
| **16** | **off** | **f16** | **256** | **145.8** | **9.13** |
| 16 | on | q8_0 | 512 | 85.8 | 7.49 |
| 15 | on | f16 | 512 | 80.3 | 7.57 |
| 18 | on | q8_0 | 256 | 76.3 | 7.87 |
| 17 | on | f16 | 256 | 75.9 | 8.06 |

- 6 measurements in 956 s (the old search needed 12 in 1195 s). The winner is the point the corrected planner predicted out of sample earlier: 16 layers, ubatch 256. Every flash-attention arm is under the 100 tok/s prompt floor again.
- `forge serve` accepted the tuned result ("fits the server's memory estimate"), loaded on the first attempt, and served at **9.85 / 9.82 / 9.88 tok/s** (200 tokens each). Before the fix the server path gave 8.0 tok/s at 11 layers after a failed load; Ollama's own default was 7.6.
- Trade-off of ubatch 256: prompt processing at depth drops from 187 to 146 tok/s (still above the floor). For an agent turn that re-reads ~1,500 new tokens and writes ~400, the two settings are within a few seconds of each other; generation-heavy work favours 256.

## 2026-11-08 — REA decompiles offline (Ghidra 12.1.4 + JDK 25)

`tools/ghidra-install.ps1` installs Ghidra and a Temurin JDK into `C:\Users\arthu\llm\rea\`, both checked against their published SHA-256, nothing global; REA finds them through `GHIDRA_INSTALL_DIR` and `JAVA_HOME`.

First attempt failed honestly: Ghidra 12.2 + JDK 21 installed fine and `rea doctor` called them healthy, but `rea decompile` refused with `unsupported_version` ("accepted releases are 12.1.x (verified build 12.1.4)", "a 64-bit full JDK 25 or newer"). `doctor` checks presence, the provider checks versions. Replaced with Ghidra 12.1.4 and JDK 25.0.5.

Test on a binary with known source (`tools/rea-target/checksum.c`, built on the host with TDM-GCC `-O1`; the host's mingw64 gcc fails to load its LTO plugin): `rea decompile --procedure forge_checksum checksum.exe --provider ghidra`, 34 s cold. Raw: `results/rea/decompile-forge_checksum.json`.

| source | Ghidra via REA |
|---|---|
| `acc = 0x1505u` | `uVar2 = 0x1505` |
| `acc = ((acc << 5) + acc) ^ buf[i]` | `uVar2 = uVar2 * 0x21 ^ (uint)*param_1` (x*32+x = x*33) |
| `if (buf[i] == 0x2A) acc += 7` | `if (*param_1 == 0x2a) uVar2 = uVar2 + 7` |
| `for (i = 0; i < len; i++)` | `if (param_2 < 1) ... do { ... param_1++ } while (param_1 != end)` |
| `return (int)(acc & 0x7FFFFFFF)` | `return uVar2 & 0x7fffffff` |

Every operation, constant and branch of the original is recovered; only names are lost (the binary has symbols for the function name, not for locals). Issue #11 is done; next is wiring REA's MCP server into the local agent (#12).

## 2026-11-09 — phase 3: do the speed settings change the answers? (issue #6)

**Instrument.** `03-agent/quality/`: 40 function-level problems (20 Python, 20 JavaScript) written for this repo, hidden tests run with `python -m unittest` / `node --test`, a reference and a broken solution for every problem (`tests/test_quality.py`, 53 tests, also run on the host with Python 3.12 and Node 20), a runner, a report and a host driver. Greedy decoding, one sample, identical prompts for every arm. A reviewer that saw only the prompts and tests found seven spots where a correct answer could fail unfairly or a stated requirement was not enforced (a null-prototype result, `.length` on arrays, tab and double-space inputs, leading zeros, a missing `+` sign case, two "O(log n)" problems that a linear scan passed); all fixed before any arm ran. Raw results: `results/phase3/` (`report.md` is `qreport.py` output); the one-line summary is that **the model passes 9 of 40 problems and no tested setting changes that detectably, except that thinking makes it worse**.

**Things that changed the design (each found by running it on the real host):**
- `C:\Users\arthu\package.json` has `"type": "module"`. The temp dir is below it, so every `.js` file loaded as an ES module and every JavaScript problem failed with `require is not defined`; it surfaced only when the harness tests ran on the host. The runner now writes a `package.json` with `"type": "commonjs"` next to the solution (regression test: a parent directory with `"type": "module"`).
- **Thinking is a template switch.** The model's chat template (read from the GGUF) ends the generation prompt with `<think>\n` unless `chat_template_kwargs.enable_thinking` is `false`, so with no kwarg it thinks; there is no `/think` tag. The runner sends the kwarg on every request (`false` for A, C, D, E; `true` for B).
- **A plain token cap does not work for thinking.** Smoke run, `max_tokens` 2,560: the first two problems both ran to the cap (~270 s each) with the model still reasoning and no answer (`results/phase3/smoke/B.jsonl`); at temperature 0 the reasoning loops ("I'll just use ... That's not right." repeated). Arm B therefore uses llama-server's own thinking budget: `--reasoning-budget 1024`, set as the environment variable `LLAMA_ARG_THINK_BUDGET` (`forge serve` passes the environment on), with a message injected when it runs out so that the runner can count how often.
- **Python indentation.** In the first runs this model often wrote a whole Python function with one-space or no indentation (`IndentationError`), and its reasoning drafts have the same habit. Probe on `py_01`-`py_05`/`py_08` (A config, `results/phase3/probe/`), rule fixed beforehand = fewer syntax errors: v1 (first wording) 4 of 8 Python answers with a syntax error, v2 (+ "Indent with 4 spaces per level") 3 of 5 against 2 of 5 for v1, v3 (+ a short example answer in the system message) 1 of 8. v3 is used for all arms. Pass count on the probe did not change (1 of 8): the syntax errors became logic errors.
- `forge serve` gained `--kv`, `--fa` and `--ubatch` (a forced setting bypasses the tuned entry; `tests/test_serve.py`), so that C and D can be started through the planner at all.

**Arms** (each a fresh `forge serve`; `results/phase3/<arm>.serve.json` has the printed config and the exact command, shared flags `-m <model> --host 127.0.0.1 --port 8081 -t 6 -tb 6 -np 1 --fit off`):

| arm | change against A | planner / tuned | llama-server flags that differ |
|---|---|---|---|
| A | forge's tuned config, thinking off | tuned, 16 layers | `-ngl 16 -c 16384 -fa off -ctk f16 -ctv f16 -ub 256 --spec-type none` |
| B | thinking on, 1,024-token thinking budget | tuned, 16 layers | as A, env `LLAMA_ARG_THINK_BUDGET=1024` |
| C | `forge serve --kv q8_0 --fa on --ubatch 256` | planned, **18 layers** | `-ngl 18 -fa on -ctk q8_0 -ctv q8_0 -ub 256` |
| D | `forge serve --ctx 32768 --ubatch 256` | planned, **11 layers** | `-ngl 11 -c 32768 -ub 256` |
| E | `forge serve --spec ngram-mod` | tuned, 16 layers | as A with `--spec-type ngram-mod` |
| A2 | A repeated unchanged (noise floor) | tuned, 16 layers | as A |

**Results** (`python 03-agent/quality/qreport.py results/phase3`; arm vs A on the problems both have; server errors left out of every statistic):

| arm | pass (Wilson 95%) | only A / only arm, McNemar p | share of hidden tests passed; A higher / arm higher, sign p | same text as A | median tok/s | median tokens | token cap hit / thinking budget used up | decision |
|---|---|---|---|---|---|---|---|---|
| A | 9/40 = 22% (12-38%) | | 62.2% | | 8.6 (9.7, see below) | 248 | 3 / - | baseline |
| B thinking | 7/37 = 19% (9-34%) | 4 / 3, p = 1.00 | 35.9%; 21 / 7, **p = 0.013** | 0/37 | 9.7 | 1,357 | 6 / 27 of 37 | **reject** |
| C q8_0 + FA | 10/40 = 25% (14-40%) | 1 / 2, p = 1.00 | 63.7%; 5 / 9, p = 0.42 | 13/40 | 9.9 | 246 | 2 / - | undecided |
| D 32K | 10/40 = 25% (14-40%) | 0 / 1, p = 1.00 | 66.7%; 1 / 7, p = 0.07 | 21/40 | 8.2 | 244 | 1 / - | undecided |
| E ngram-mod | 9/40 = 22% (12-38%) | 0 / 0, p = 1.00 | 62.2%; 0 / 0 | 37/40 | 9.7 | 248 | 3 / - | undecided |
| A2 repeat of A | 25 of 40 answered | 0 / 0 | | **25/25** | 9.7 | 266 | | |

Decisions follow the rule in `docs/PROTOCOL.md` (written while the grid ran, before any arm but A was looked at):
- **B reject.** Significantly worse on the graded test (p = 0.013) and it costs 5.5x the tokens (median 1,357 vs 248; 85 min of requests against 24-36 for the other arms). The loss is a Python formatting failure: 18 of its 20 Python answers are `IndentationError`s (A: 2), so thinking makes the model copy its flat-indented draft. On JavaScript, where indentation cannot fail, B is 5/17 against 3/20 (66% against 60% of tests): no detectable gain from thinking either. Three B answers are missing (server error, below).
- **C undecided.** Same score within noise, generation 9.9 vs 9.7 tok/s (+2%, under the 10% benefit threshold), and the tuning measured it slower at depth 8,192 (7.87 vs 9.13 tok/s). Nothing here argues for switching.
- **D undecided.** The score is the same or slightly better (graded +4.5 points, p = 0.07; 7 problems better, 1 worse), but the benefit of 32K, a longer context, is not exercised by 300-token prompts, and it costs 5 layers and 16% speed (8.2 vs 9.7 tok/s). The decision belongs to phase 4: use it only for tasks that need more than 16K.
- **E undecided.** Not one problem changed outcome and 37 of 40 answers are word-for-word the same (the 3 that differ score the same), so `ngram-mod` did not cost correctness here; but it did not buy speed either (9.7 vs 9.7 tok/s). Phase 2 found its gain on code *rewrites* (up to 17.5 tok/s); these problems write new code. Keep it opt-in and test it where the agent edits existing code.

**What 40 problems can say.** A passes only 9, so the paired tests can detect a loss only if most of A's passes flip (6 only-A flips and no others gives p = 0.03); "within noise" for C, D and E means "no large effect", not "equal". The graded view is more sensitive and is why B stands out. The benchmark does not test long contexts (all prompts are about 300 tokens), so D's 32K is only checked for side effects of the different layer split. Greedy decoding is repeatable here: A2, after a server restart, gave exactly A's text on all 25 problems it answered (and 5 of 5 in the prompt probe). Different settings do change the text (C 13 of 40 identical to A, D 21 of 40, E 37 of 40) without a detectable change in score.

**What went wrong during the run (so the numbers are read with it in mind):**
- Old one-shot tasks (`llm-agent`, `llm-bench-b`, `llm-bench-c`, `llm-ghidra`) kept an unfired 23:59 trigger from earlier `schtasks /Create /SC ONCE /ST 23:59` + manual `/Run`, and started themselves at 23:59:00, while A2 was running (it had started at 23:44). From 00:12:41 A2's llama-server stopped answering (`urlopen error [WinError 10061]`, connection refused) for the last 15 problems. Whether it crashed under those jobs or was killed I do not know. So A2 has 25 answers, all identical to A's, and its speed (9.7 tok/s, stable over the 25) is the baseline I use. Added to the protocol: list the `llm-*` triggers before a long host run, and delete or disable a one-shot task's trigger after running it by hand.
- Arm B had 3 server errors, `js_02`-`js_04` at 20:56, one to three minutes after its server started: `vk::Device::allocateMemory: ErrorOutOfDeviceMemory` (HTTP 500, retried twice each). The server then answered the other 37 normally; the cause is unknown. I did not rerun them because the GPU was by then in use by the jobs above, so B is compared on 37 problems; the runner keeps the error rows in `B.jsonl` and `qreport.py` leaves them out.
- Arm A's median 8.6 tok/s is low: its first 20 requests (7.8 median) overlap the time I polled the host with a PowerShell call every minute, its last 20 (9.1) do not. I switched to `scp`-only polling after that and use A2's 9.7 as the speed baseline. This is a correlation, not a proven cause. Likewise three requests in the smoke and probe runs fell to 2.7-4.4 tok/s exactly while unit tests ran in the Linux VM, which shares the CPU with the host.
- On a host shared with other jobs I killed processes I did not own, twice. The grid driver stopped each arm by killing its process tree and then, as a safety net, every `llama-server.exe` by name; at the end of A2 that also killed Ollama's runner for the other jobs (it restarted a second later), and a cleanup script I ran afterwards killed the restarted one before I had checked who owned it. Ollama starts a new runner on its next request, so I expect little damage, but their sweeps may show a failed request around 00:19. The driver (`qgrid.py`) now kills only its own tree, and the count it logs after each arm is "any owner".
- Everything the grid started was stopped after every arm (`llama-server processes left: 0` in `results/phase3/phase3-full.log` for A-E; A2's line reads 1: Ollama's runner for the other jobs). I loaded no Ollama model.

Next: phase 4 starts from arm A (the tuned config, thinking off, prompt style with an example) and takes `ngram-mod` and 32K as per-task options to test on real edits, not as defaults.

## 2026-11-09 — phase 4: is a local agent usable? (issues #7, #8)

**Question.** The definition in `docs/PROTOCOL.md`: at least 60% of tasks succeed (every test passes, the visible test files byte-identical, inside the time limit) and the median time per task is at most 15 minutes. Two arms on forge's config A (16 layers, ubatch 256, FA off, f16, 16K, `spec none`, thinking off, temperature 0; `serve-A.json` in `results/phase4/`): **O** OpenCode v2.2.5 alone, **T** the thin orchestrator (`03-agent/orchestrator/`). 12 tasks (`03-agent/tasks/README.md`), 15 minutes per run enforced by the runner on the whole process tree (Windows job object; every run logged `leftover_processes: 0`), the hidden tests re-run and the test files re-hashed by the runner, never by the agent.

**Result** (`results/phase4/report.md`, generated by `03-agent/harness/report.py`; raw rows in `O.jsonl`, `T.jsonl`). The unit is the first run of each task; times and token counts are over every run (O 23 runs, T 16):

| arm | success, first run per task (Wilson 95%) | pooled over all runs | median min, all runs / successes | timeouts | share of tests passed | median steps | usable? |
|---|---|---|---|---|---|---|---|
| O OpenCode alone | **7/12 = 58% (32-81%)** | 14/23 = 61% (41-78%) | 7.6 / 2.9 | 7 | 75% | 11 | **no**, by one task |
| T orchestrator | **4/12 = 33% (14-61%)** | 4/16 = 25% (10-49%) | 4.5 / 2.9 | 0 | 78% | 3 | **no** |

O against T, exact McNemar on the same 12 tasks: only O passes `js-lru`, `js-queue`, `py-discount`, `py-refactor-split`; only T passes `js-fix-csv`; 3 pass in both (`js-slugify`, `py-fix-paginate`, `py-intervals`), 4 in neither (`js-duration`, `js-router-wildcard`, `py-todo-tags`, `py-wrap`); p = 0.375, so **the difference is not significant at 12 tasks, but its sign is against the orchestrator**. Second runs: O has one on 11 of the 12 tasks (`py-refactor-split` did not fit in the budget): 10 of 11 gave the same pass/fail as run 1; the exception is `js-router-wildcard`, which timed out in run 1 (5/10) and passed in run 2 (10/10, 7.6 min). T has a second run on 4 tasks (`js-lru`, `js-queue`, `js-slugify`, `py-discount`): 3 of 4 the same, `js-slugify` passed and then failed (5/8).

**Verdict, stated plainly: by the definition in `docs/PROTOCOL.md`, applied as I fixed it before the grid (the first run of each task), a local agent on this machine is not usable: OpenCode alone reaches 58% (7 of 12 tasks), one task short of 60%, with a median of 7.6 minutes per run (that part passes); the thin orchestrator reaches 33%. The O result is on the line, not clearly under it: the Wilson interval (32-81%) contains 60% and counting every run (14 of 23) gives 61%. So: "not shown to be usable", and "borderline" is the honest word for OpenCode alone; "not usable" for the orchestrator. The thin orchestrator does not move the number up, it moves it down, while making runs faster (median 4.5 vs 7.6 min) and cheaper (median 3 steps vs 11).**

**What would have to change for O to reach the bar.** Four of its five failures are the 15-minute limit (`js-duration` 6/11 tests, `js-router-wildcard` 5/10, `js-fix-csv` 2/9, `py-wrap` 0/9 in run 1); the fifth, `py-todo-tags` (10/13 tests), ended on its own after 41 steps. Its successes are fast (median 2.9 min), its failures are long: at 9 tok/s a run gets about 11-13 model steps in 15 minutes. One more task passing is enough for 60%, and the cheapest candidates are the timeouts, so speed is the lever the data points at (a faster model or `ngram-mod` on edits, or a longer limit); I did not measure how many of the four would finish in 30 minutes, so that is a hypothesis, not a result. The harness side (T) cannot fix it with the design tried here.

**Why T lost** (from the traces in `T.jsonl`; `orchestrator.version` 2):
- It stops early with most of the time unused: the 8 failures ended after a median of 5 minutes (of 15), four of them on "3 rounds without more tests passing". Its tests-first loop only has the visible tests as signal, and in 3 of the 8 failures (`js-lru`, `js-router-wildcard`, `py-refactor-split`) the visible tests were green and the hidden ones found a gap; the review round ("compare the files with the task text") rarely helped: of 7 review rounds 4 produced no edit and 3 produced one (in one of them the edit broke tests and was reverted).
- Ten replies were cut at the token cap (1,100-2,200 tokens, 2-4 minutes each at 9 tok/s) without a usable `FILE` block, almost all of them in retry rounds at temperature 0.3-0.6, where the model writes prose.
- Whole-file rewrites by a 9B model at 9 tok/s are slow and error-prone compared with OpenCode's small edits plus its own test runs; O wins exactly the tasks that need several edit-and-test cycles (`js-queue` 28 steps, `py-discount` 16, `py-refactor-split` 8).
- T is not repeatable once it retries (`js-slugify`: ok in run 1, 5/8 in run 2, presumably because the retry seed changes with the run number); its first attempt of each step is greedy.

**The Python indentation repair** (`pyindent.py`): it never fired in the T runs (`repairs: 0` in all 16 runs): at temperature 0 with thinking off and the file in context the model keeps the indentation it sees, as in phase 3's arm A (2 of 20 Python answers). So **T-norepair would have been identical to T by construction and was not run** (a run of it would only repeat T). Offline, on the answers phase 3 recorded (`eval_repair.py`, `results/phase4/repair-offline.txt`): of 22 distinct Python answers that did not compile, 20 compile after the repair and 3 then pass their hidden tests (none did before); on the 10 Python reference files with nested blocks, flattened to one space per line, all 10 compile again, 2 come back byte for byte, and 2 of the 6 tasks pass all their tests. It rebuilds the nesting from block openers, and flat text cannot say where a block ends, so it is right about as often as the model's logic is. `T-ngram` (optional) was not run for lack of host time.

**What went wrong (read the numbers with it):**
- **The first T pass was invalid and was thrown away.** Windows pipes return CRLF; the failure-block parser expected LF, so on Python tasks the model was told "0 tests fail" while one did, and on `js-queue` it wrote Python into JavaScript files because the prompt example was Python. The unit tests ran on Linux and passed. I stopped the grid (STOP file, the running agent finished), fixed both (`orchestrate.py`: CRLF, a never-"passing" fallback block, a language-matched example), re-ran every T task from scratch and kept the discarded pass as `T-v1-crlf-bug.jsonl` (2 of 7 tasks done before the stop). All T numbers above are the corrected version; O is unaffected. Lesson: run the harness unit tests on the target OS, with the target's line endings, before a grid.
- OpenCode is not repeatable at temperature 0 (the proxy forces temperature, seed, thinking off): `py-wrap` was solved in 132 s in the smoke run (7-minute limit) and timed out at 0/9 tests in the grid. Across the 11 tasks with a second O run the outcome agreed on 10 (see above), and the one exception moved the pooled rate from 58% to 61%: the O number carries a noise of about one task.
- One O run (`js-fix-csv`) had a request rejected for context overflow (16K) and one API error; it timed out anyway.
- The tasks are small by design (one to five files), 12 of them: one task is 8 points, and the prompts were not read by an independent reviewer. The model passes 22% of function-level problems (phase 3), here 58% with a tool loop: the tasks are easier than the phase 3 set or the loop matters; this run cannot separate the two.
- Host: `opencode.cmd` from the task brief does not exist (an `npm i -g opencode-ai` failed with ENOSPC in the early days and left a half-installed 1.19.5); the working install is `C:\Users\arthu\.opencode\bin\opencode.exe` v2.2.5, and the version of run 1 is not recorded anywhere. OpenCode v2 normally talks to a background service that a controller cannot kill with the tree; `--standalone` keeps the server inside the process tree. Host time: about 5 h 20 min from the smoke run to the last run. I stopped the second T pass after 4 of 12 tasks and ran second O runs instead, because the verdict hinges on O. The host's GPU log showed no other benchmark at the start and no other `llm-*` task existed; `llm-phase4` is deleted.

**Decision for phase 5.** Not more orchestrator prompt engineering: the tool loop beat whole-file rewrites on 4 tasks out of 5 that differ. Try, in this order: (1) **speed first**: the same O arm with `ngram-mod` (phase 2: up to 17.5 tok/s on rewrites, phase 3: no cost in correctness) and with a 30-minute limit, 2 runs per task, to learn how many of the four timeouts are time and how many are capability; (2) **verification inside the tool loop, not around it**: give OpenCode the orchestrator's one useful idea, a tool that runs the task's tests and returns only the first failing test's trimmed output, and a stop rule from the harness (never the agent's summary) - this is issue #7's "OpenCode + a thin orchestrator" arm, which this phase did not build; (3) a second, faster model that fits the 4 GB card (the protocol lists it as out of scope until added there) if (1) shows the failures are time.

## 2026-11-09 — phase 5: is the agent's gap time or ability? (issue #22)

**Question.** Phase 4 left OpenCode alone (arm **O**) at 7/12 first-run (58%, one task under the 60% line) with 4 of its 5 failures at the 15-minute limit. If the gap is time, speed fixes it; if it is ability, it does not. Three arms on forge's config A (16 layers, ubatch 256, FA off, f16, 16K, thinking off, temperature 0, seed 1), the same 12 tasks and prompts, the phase 4 runner (hidden tests re-run and test files re-hashed by the runner, limit enforced on the process tree, every run `leftover_processes: 0`), OpenCode v2.2.5 `--standalone` through the metering proxy: **O30** the same with a 30-minute limit; **O30-ngram** O30 on a server with `--spec ngram-mod`; **O30-tool** O30 plus one extra tool. The decision rules were written in issue #22 before the first run. Raw rows: `results/phase5/` (`O30.jsonl`, `O30-ngram.jsonl`, `O30-tool.jsonl`, serve configs, host log, `report.md` from `03-agent/harness/report.py results/phase5 --phase4 results/phase4`); phase 4's O is the baseline.

**Result** (first run of each task is the unit; every arm has 12; no second runs fitted, see below):

| arm | success at its limit (Wilson 95%) | within 15 min (the protocol's bound) | passes after minute 15 | median min, all runs / successes | timeouts | tests passed | median gen tok/s | usable at its limit? |
|---|---|---|---|---|---|---|---|---|
| O (phase 4, 15 min) | 7/12 = 58% (32-81%) | 7/12 | 0 | 7.6 / 2.9 | 4 | 75% | n/a | no |
| O30 | 6/12 = 50% (25-75%) | 5/12 = 42% | 1 (`js-queue`, minute 22.9) | 6.4 / 4.1 | 2 | 78% | 8.8 | no |
| O30-ngram | 6/12 = 50% (25-75%) | 6/12 = 50% | 0 | 8.8 / 3.3 | 2 | 82% | 9.1 | no |
| O30-tool | 9/12 = 75% (47-91%) | 8/12 = 67% | 1 (`py-discount`, minute 17.6) | 6.2 / 4.8 | 2 | 90% | 9.0 | yes, by the letter (read below) |

Exact McNemar, first runs: O30 vs O 1-2, p = 1.00 (only O30 `js-router-wildcard`; only O `js-lru`, `js-slugify`); O30-ngram vs O 1-2, p = 1.00; O30-ngram vs O30 1-1, p = 1.00; O30-tool vs O 2-0, p = 0.50; O30-tool vs O30 3-0, p = 0.25. Nothing is significant at 12 tasks (6-0 is needed).

**The direct question: did the phase 4 timeouts finish in 30 minutes, and when?** (O30 first run; the other arms in `report.md`)

| task (phase 4 O) | O30 | O30-ngram | O30-tool |
|---|---|---|---|
| `js-duration` (timeout twice) | no: ended itself at minute 29.6 with 3/11, `compaction.failed` | no: 8/11 at 18.4, `compaction.failed` | no: timeout 5/11 |
| `js-fix-csv` (timeout twice) | no: timeout 2/9 | no: timeout 5/9 | no: timeout 5/9 |
| `js-router-wildcard` (timeout, then ok at 7.6) | yes, at minute **8.3** | yes, 10.9 | yes, 6.4 |
| `py-wrap` (timeout twice) | no: stopped itself at minute 2.7 with 7/9 | no: timeout 3/9 | yes, 6.1 |
| `py-todo-tags` (phase 4: ended itself, 10/13) | no: timeout 8/13 | no: 11/13 at 10.5 | no: 11/13 at 16.9 |

Not one of the four finished *because* it had more time. The two that passed (`js-router-wildcard`, `py-wrap`) did it inside 15 minutes; `js-router-wildcard` already passed in phase 4 run 2 at 7.6 minutes, so it is a task whose outcome varies between runs, not one that needed time. Of 36 phase 5 runs only 2 passed after minute 15 (`js-queue` 22.9 in O30, which phase 4 solved at 12.7; `py-discount` 17.6 in O30-tool).

**Per task over every run of every arm** (passes / runs, phase 4's two O runs included): `js-duration` 0/5, `js-fix-csv` 0/5, `py-todo-tags` 0/5, `py-wrap` 1/5, `js-lru` 3/5, `js-slugify` 4/5, `py-discount` 4/5, `js-router-wildcard` 4/5, and `js-queue` 5/5, `py-fix-paginate` 5/5, `py-intervals` 5/5, `py-refactor-split` 4/4. Three tasks fail in every run of every arm whatever the time, one nearly so, five pass nearly always, and three (`js-lru`, `js-slugify`, `py-discount`) flip between runs. The same configuration, O and O30 (they differ only in the limit), scored 7 and 6 of 12 and disagreed on 3 tasks: the noise of this instrument is about two tasks, in both directions.

**How the failed runs ended** (first runs): O30 6 failures = 2 timeouts (`py-todo-tags`, `js-fix-csv`), 2 stopped on their own with a test failing (`py-wrap` at 2.7 min, `js-slugify` at 2.3) and 2 ended by OpenCode itself with `compaction.failed: Compaction produced no summary` (`js-lru` at 16.8 min, `js-duration` at 29.6); O30-ngram 6 = 2 timeouts, 2 stopped on their own, 2 compaction failures (`py-discount` 19.5, `js-duration` 18.4); O30-tool 3 = 2 timeouts, 1 stopped on its own. So half of the failures in the 30-minute arms were not the limit at all: the agent ended the session with a test still failing, or its 16K context filled, OpenCode tried to compact the session, the 9B model returned no summary and the run ended. The second one is a *long-session* failure that a longer limit cannot fix and that phase 4 could not see (at 15 minutes only one run had a context overflow).

**ngram-mod, speed and success.** Median generation speed per run 9.1 tok/s against 8.8 (+3%), median model steps per wall minute 2.08 against 1.97 (+6%); weighting the per-run rates by completion tokens it is 11.0 against 8.6 tok/s (+27%), because its gain sits in long rewrite-heavy runs: the two ngram timeouts did 51 and 52 steps in 30 minutes against 26 and 17 for O30's. Draft acceptance 49% (23,702 of 48,318 drafted tokens, from the server's `timings` in the proxy's records). Success is unchanged, 6/12 against 6/12, with different tasks (+`js-slugify`, -`py-discount`, p = 1.00): twice the steps in the same time bought no extra pass, which is the cleanest single piece of evidence that the failures are not a shortage of steps. By the rule of the issue (median +10%) it is **not adopted**: +3% is under the threshold, and it changed 2 of 12 outcomes in both directions, as phase 2 warned.

**The test tool.** `03-agent/harness/testtool_mcp.py`, a stdlib MCP server (newline-delimited JSON-RPC over stdio) that reuses the orchestrator's `run_tests`, `parse_python`, `parse_node` and `_trim_traceback` and returns `ALL TESTS PASS (n tests).` or `k of n tests pass, f fail.` plus only the first failing test's trimmed output. Exact configuration, written per run into the isolated OpenCode config (`runner.test_tool_mcp`):

```json
"mcp": {"servers": {"forge": {"type": "local", "codemode": false,
  "command": ["C:\\Python312\\python.exe", "<repo>\\03-agent\\harness\\testtool_mcp.py", "--repo", "<run dir>\\repo",
              "--test-cmd", "[\"C:\\\\Python312\\\\python.exe\", \"-m\", \"unittest\"]", "--lang", "py", "--log", "<run dir>\\mcp-calls.jsonl"]}}}
```

The model sees it as the native tool `forge_run_tests`; the task prompt is identical to every other arm. Three things about OpenCode v2.2.5 had to be found by running it, none is in the v1 documentation: the schema is `mcp.servers.<name>` (the v1 form `mcp.<name>` is silently ignored and `opencode mcp list` reports "No MCP servers configured" even for a correct config, so that command is no check); by default v2 hides MCP tools behind a JavaScript `execute` tool ("Code Mode", `tools.forge.run_tests()`), where a 9B model would have to write JavaScript, and `codemode: false` is what puts the tool in the model's own tool list; and OpenCode needs about 1.5 s to connect the server, so the first model request of a run does not list the tool (the second does). I tested it end to end before the grid with a scripted fake model that called the tool through a real OpenCode run (the model saw the first-failure text) and with two real-model smoke runs (`results/phase5/smoke/`).

The result is the interesting and the delicate one: **9/12 = 75%** first-run, 8/12 within 15 minutes, median 6.2 min per run, which is above the protocol's 60% and median bars. But the model called the tool in only 5 of 12 runs (25 calls: `py-todo-tags` 2, `js-queue` 6, `py-discount` 12, `js-duration` 4, `py-intervals` 1), and in those 5 runs the outcome equals O30's in every case (3 passes, 2 failures); the three tasks it gained over O30 (`js-lru`, `js-slugify`, `py-wrap`) are three of the tasks that flip between runs and the tool was **not called in any of them**. So the 75% is not attributable to the tool; the most likely reading is a favourable draw of the same noise (adding a tool changes the request, so the greedy trajectory changes, as it did between O and O30). In `py-todo-tags` the tool said all visible tests pass and the hidden ones still failed 2 of 13: a green visible suite is not the finish line, which is the same wall the orchestrator hit in phase 4.

**Verdict against `docs/PROTOCOL.md`, stated plainly.**
- *Time or ability? Ability.* The pre-registered rule for "time" needed O30 at 8 of 12 or more *and* at least 2 passes after minute 15; O30 has 6 and 1. Doubling the limit moved the first-run rate from 58% to 50% (inside the noise), none of the four phase 4 timeouts was rescued by time, five of the failed runs ended on their own long before the limit, and giving the agent twice the steps per minute (ngram) changed nothing. Four tasks (`js-duration`, `js-fix-csv`, `py-todo-tags`, `py-wrap`) are not solved by this model with this loop in any run of any arm except one `py-wrap` run, at any time allowed (up to 30 minutes), and a second failure mode (compaction at 16K) appears only in long sessions. Speed fixes are not the lever.
- *Usable?* By the definition as fixed (rate >= 60% and median <= 15 min), read off the first runs: **O30 no (50%), O30-ngram no (50%), O30-tool yes (75%, median 6.2 min; 67% when re-scored at the 15-minute bound)**. I do not claim the tool made it usable: the Wilson interval of 9/12 is 47-91%, it contains values under the bar; the gain over O30 is p = 0.25 (3-0); the tool was not called in the three tasks it gained; and phase 4's O sat at 58% and phase 5's O30 at 50% with no change but the limit. What the data supports is "this configuration reached the line on one draw", and a replicate is needed before it means anything.
- The 30-minute limit does not change the median criterion and was not allowed to: the medians above are as measured, 6-9 minutes, far under 15; the criterion that decides is the success rate, as in phase 4. The report scores each arm both at its own limit and at 15 minutes.

**What went wrong, so the numbers are read with it:**
- **No second runs.** Budget: 7 h 15 min of grid time on the host (plus a 6-minute smoke run) for one run per task per arm (O30 2 h 37 min, O30-ngram 2 h 21 min, O30-tool 2 h 16 min; the 30-minute timeouts dominate), so the stability line of phase 4 does not exist for phase 5; the noise estimate comes from phase 4's pairs (10 of 11 agree) and from O against O30 here (9 of 12 agree). The grid's 390-minute launch cutoff stopped O30-tool after 8 tasks; I finished its last 4 (the fast ones) with a second grid launch, 25 minutes over the stated budget, because an arm on 8 of 12 tasks would not have been comparable.
- **The mid-run patch.** After the first four O30 runs showed two failures that were not the limit (a compaction error, a run that stopped itself), I added `agent_events` to the runner (the `agent_errors` and `compaction_lines` fields; the grid starts a fresh runner per run, so it applied from the next run). The first five O30 records (`py-todo-tags`, `js-lru`, `py-wrap`, `py-fix-paginate`, `js-fix-csv`) lack those fields; the report reads their ending from the tail of the agent log, which holds the final error event (checked for `js-lru`).
- **Why O30 differs from O is not known.** The same task under O and O30 gets a different run directory (`...__O__1` vs `...__O30__1`), and OpenCode's system prompt contains the working directory, so the prompts differ by a few tokens; greedy decoding on a 9B model can diverge on that. I did not test it (it would take a second run with identical directory names), so it is a candidate cause, not a finding. The consequence stands either way: single first runs of OpenCode carry about two tasks of noise.
- **Draft counts** are in the stream's `timings` only for requests where the server drafted; the two smoke runs showed none, the grid's ngram arm 48,318.
- **A concurrent session** moved the shared checkout to another branch (`modal/qbench`) during the grid. Nothing was lost: the host ran from a `git archive` of this branch, and I finished in a separate git worktree. The Hermes gateway python processes on the host were running before and during, as in phase 4; no other `llm-*` task existed, `llm-phase5` and `llm-phase5b` are deleted, Ollama had no model loaded, no OpenCode or llama process was left, the host's `work\` is empty.
- Windows unit tests (80, Python 3.12, Node 20) passed before the grid; the later report and `agent_events` edits were tested on Linux only (the runner path itself ran 36 times on the host).

**Decision for phase 6.** Not more time, not more orchestration. (1) **Measure before believing:** replicate O and O30-tool with at least 3 runs per task, identical run directory names across arms, and compare pooled per-run rates; a difference of two tasks on 12 tasks is not a result, and the O30-tool number is the one that needs this. (2) **Read the four hard tasks** (`js-duration`, `js-fix-csv`, `py-todo-tags`, `py-wrap`; 0/5 or 1/5 in every arm) against their hidden tests: the prompts were never read by an independent reviewer (phase 4 said so), and if the cause is an ambiguous prompt the suite is what to fix; if not, the model is the limit and the lever is a stronger one that fits the machine (the protocol lists other candidates as out of scope until added there; the phase 3 benchmark of six models on Modal on the `modal/qbench` branch, from another working session, is the natural input for that decision and is not part of this PR). (3) **The 16K context** ends long OpenCode sessions (`compaction.failed` in 4 of 12 failed first runs of the two arms that ran long); the 32K arm of phase 3 costs 5 layers and 16% speed, so test it only on the tasks that reach compaction, not as a default. (4) `ngram-mod` stays opt-in: it does not change the success rate and doubles the steps only on long runs.
