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
