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
