# Experiment protocol

Question: **what is the fastest configuration of this machine that still gives a usable local coding agent, and which local model is that agent best built on?**

"Usable" is defined up front, so the result cannot be argued into place afterwards:
an agent run is a success when every test of the task passes, the test files are byte-identical to the originals, and it finishes inside the time limit. A configuration is usable when its agent success rate on the task suite is at least 60% and its median time per task is at most 15 minutes.

## Controls (apply to every phase)

- One model loaded at a time; Ollama models are unloaded (`keep_alive: 0`) before any llama.cpp run, and nothing else uses the GPU.
- Before a long host run, list every `llm-*` scheduled task with its trigger. A task created with `/SC ONCE /ST 23:59` and started by hand keeps that trigger and starts itself again at 23:59 (phase 3: four old jobs did, in the middle of the last arm). Delete or disable the trigger after a manual run. Never stop a process by image name on this host, only the tree you started.
- Greedy decoding (temperature 0, fixed seed) for every speed and quality measurement, so output differences come from the configuration, not from sampling. Greedy decoding is repeatable on this stack (phase 3: the same answers after a server restart, 25 of 25), so a different answer under a different setting is an effect of the setting.
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

On the best phase-1 point: `--spec-type none | draft-mtp | ngram-mod | draft-mtp,ngram-mod`, draft length 1 to 4. The model ships one MTP layer (`nextn_predict_layers = 1`). Measured on a fixed set of coding prompts with real completions: generation tok/s, draft acceptance rate, and whether the output is token-identical to `none`. This protocol first required identity at temperature 0 and called any difference a bug. Phase 2 measured otherwise (on this backend `ngram-mod` changes 2 of 8 outputs), so identity is a result to report, and what matters is whether the differences cost correctness: that is arm E of phase 3.

## Phase 3: quality (do the speed settings change correctness?)

Instrument: 40 function-level problems (20 Python, 20 JavaScript) in `03-agent/quality/`, written for this repo, each a prompt and hidden unit tests (`python -m unittest`, `node --test`). Every problem has a reference solution that passes all its tests and a broken one that fails at least one (`tests/test_quality.py`), and the prompts were read against the tests by a reviewer who saw only the files. The prompt is identical for every arm. One sample per problem at temperature 0 and a fixed seed; the unit of comparison is the problem.

Arms: **one setting changed at a time against the default**, each started with `forge serve` so that the planner places the layers for that arm, the exact llama-server command line recorded:

| arm | change against A | `forge serve` flags |
|---|---|---|
| A | none: forge's tuned config (16 layers, ubatch 256, FA off, f16, 16K, spec none), thinking off | |
| B | thinking on, with a 1,024-token thinking budget | env `LLAMA_ARG_THINK_BUDGET=1024` |
| C | KV q8_0 with flash attention on (ubatch 256 as in A) | `--kv q8_0 --fa on --ubatch 256` |
| D | 32K context (ubatch 256 as in A) | `--ctx 32768 --ubatch 256` |
| E | speculative decoding `ngram-mod` | `--spec ngram-mod` |

The earlier plan, a full thinking x KV x context grid, is dropped: eight arms on 40 problems have no power to separate anything, and a one-factor design says which setting did what.

How thinking is switched (documented from the model's own chat template): the template ends the generation prompt with `<think>\n` unless `chat_template_kwargs.enable_thinking` is `false`, so with no kwarg the model thinks. Every request therefore sends the kwarg explicitly, `false` for A, C, D, E and `true` for B. There is no `/think` tag. Unbounded thinking is not usable on this machine: at about 10 tok/s a plain `max_tokens` cap of 2,560 was used up while the model was still thinking in the first two problems tried (greedy decoding makes the reasoning loop), which would have scored the whole arm as "no answer". Arm B uses llama-server's own thinking budget (`--reasoning-budget`, set through `LLAMA_ARG_THINK_BUDGET` because `forge serve` passes the environment on) and counts how often the budget runs out.

Statistics: pass rate with a Wilson 95% interval; each arm against A on the same problems with an exact McNemar test; and a graded view, the share of each problem's hidden tests that passed, with an exact sign test. A difference whose p is not below 0.05 is reported as within noise. With 40 problems that is a statement about resolution: one problem is 2.5 points, an interval is about +-13 points wide, and about six problems must flip one way (and none the other) to reach p < 0.05.

Decision rule per arm (set while the grid was running, before any arm other than A was looked at): **reject** if the arm is significantly worse than A on either test; **adopt** (provisional, at this sample size) if it is not significantly worse on either test *and* the same run shows the benefit it was meant to bring (median generation tok/s at least 10% above A for C and E, a significantly better score for B; for D the benefit, a longer context, is not exercised by these short prompts, so D can only be rejected or left undecided); **undecided** in every other case.

## Phase 4: the agent, end to end

Task suite in `03-agent/tasks/` of graded difficulty (single function, bug fix with a failing test, multi-file change, a refactor that keeps tests green). Each run starts from a clean copy, runs OpenCode with the arm's model and server, enforces the time limit from inside the same process tree, re-runs the tests itself and hashes the test files.

Arms: model × server configuration × thinking on/off × harness (OpenCode alone; OpenCode with a thin orchestrator that feeds back only the failing test output). 3 runs per task per arm.

## Candidate model

One model: `DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic` i1-IQ4_XS (the user's pick; does not fit fully in 4 GB VRAM). It is the only model kept on the host. The other candidates this protocol once listed (a Qwen2.5 coder 7B, a ~4B model that fits in VRAM) are out of scope unless added later; adding one means a new entry here before any run, plus the hardware-adaptive layer (`forge/`) already handles any GGUF.

What phase 3 varies is therefore the configuration, not the model: thinking on/off, KV f16/q8_0, 16K/32K context, speculation off/`ngram-mod`. The GPU layer count for each arm is not a free choice: arm A takes the tuned value, the other arms get whatever the planner places for their settings (`docs/JOURNAL.md` records the layers per arm). The tuned values are layers *and* ubatch: on this card the 32K arm needs ubatch 256 (the compute buffer alone is 2.3 GiB at 512, 0.8 GiB at 256, `results/fit-params-grid-host-20261108.txt`), so ubatch is recorded per arm too.
