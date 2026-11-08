# Experiment protocol

Question: **what is the fastest configuration of this machine that still gives a usable local coding agent, and which local model is that agent best built on?**

"Usable" is defined up front, so the result cannot be argued into place afterwards:
an agent run is a success when every test of the task passes, the test files are byte-identical to the originals, and it finishes inside the time limit. A configuration is usable when its agent success rate on the task suite is at least 60% and its median time per task is at most 15 minutes.

## Controls (apply to every phase)

- One model loaded at a time; Ollama models are unloaded (`keep_alive: 0`) before any llama.cpp run, and nothing else uses the GPU.
- Greedy decoding (temperature 0, fixed seed) for every speed and quality measurement, so output differences come from the configuration, not from sampling.
- Repetitions: speed tests 3 per point (llama-bench `-r 3`, mean and standard deviation reported); agent runs 3 per task per arm.
- Run order is randomised within a phase where crashes cannot cascade; the raw output of every run goes to `results/` before any summary is written.
- The machine state is recorded with each run: free RAM, the Linux VM's share (it holds 7.3 GB of the 16 GB), runtime versions.
- Every number quoted in `docs/JOURNAL.md` or the article points to a file in `results/`.

## Phase 1: throughput (no quality judgement yet)

Runtime A, Ollama 0.40.14 (sweeps A and B, done). Runtime B, llama.cpp b12207 Vulkan build on the same GGUF file.

Factors, llama-bench (`-p 512 -n 128`, `-t 6`):
- GPU layers: 0, 8, 12, 16, 20, 24, 28, all
- attention/KV: flash attention off with f16 KV; on with f16 KV; on with q8_0 KV
- context depth: 0 and 8,192 tokens already in the cache (an agent's turn reads with a full context)

Metrics: prompt processing tok/s (pp512), generation tok/s (tg128), each at both depths.

## Phase 2: speculative decoding (llama-server)

On the best phase-1 point: `--spec-type none | draft-mtp | ngram-mod | draft-mtp,ngram-mod`, draft length 1 to 4. The model ships one MTP layer (`nextn_predict_layers = 1`). Measured on a fixed set of coding prompts with real completions: generation tok/s, draft acceptance rate, and whether the output is token-identical to `none` (it must be at temperature 0; any difference is a bug, not a trade-off).

## Phase 3: quality (does the configuration or the model change correctness?)

A fixed set of small coding problems with hidden tests (function-level, Python and JavaScript), pass@1 at temperature 0. Arms: configurations of the one candidate model below, thinking off / on x KV cache f16 / q8_0 x context 16K / 32K, with the speed-optimal GPU layers for each from `forge tune`. It checks that neither KV quantisation nor the longer context costs correctness, and what thinking buys.

## Phase 4: the agent, end to end

Task suite in `03-agent/tasks/` of graded difficulty (single function, bug fix with a failing test, multi-file change, a refactor that keeps tests green). Each run starts from a clean copy, runs OpenCode with the arm's model and server, enforces the time limit from inside the same process tree, re-runs the tests itself and hashes the test files.

Arms: model × server configuration × thinking on/off × harness (OpenCode alone; OpenCode with a thin orchestrator that feeds back only the failing test output). 3 runs per task per arm.

## Candidate model

One model: `DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic` i1-IQ4_XS (the user's pick; does not fit fully in 4 GB VRAM). It is the only model kept on the host. The other candidates this protocol once listed (a Qwen2.5 coder 7B, a ~4B model that fits in VRAM) are out of scope unless added later; adding one means a new entry here before any run, plus the hardware-adaptive layer (`forge/`) already handles any GGUF.

What phase 3 varies is therefore the configuration, not the model: thinking on/off, KV f16/q8_0, 16K/32K context. The GPU layer count for each arm is not a free choice but whatever `forge tune` measures as best for it (`docs/JOURNAL.md` records the plan and the tuned values per arm). The tuned values are layers *and* ubatch: on this card the 32K arm needs ubatch 256 (the compute buffer alone is 2.3 GiB at 512, 0.8 GiB at 256, `results/fit-params-grid-host-20261108.txt`), so ubatch is recorded per arm too.
