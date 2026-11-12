# local-forge

A coding-agent setup that runs entirely on a home PC: pick a model, fit it to the machine, measure it, and drive it with an agent (OpenCode) the way Hermes drives Claude Code. Every number in this repo comes from a script in this repo, run on the machine below.

Status: work in progress. The goal is in `docs/VISION.md`: an agent that keeps working with no internet at all. The journal (`docs/JOURNAL.md`) is the source for the article.

## The machine

| part | spec |
|---|---|
| CPU | AMD Ryzen 5 3600, 6 cores / 12 threads |
| RAM | 16 GB DDR4-3000 (a Linux VM holds 7.3 GB of it) |
| GPU | AMD Radeon RX 6500 XT, 4 GB VRAM, PCIe 4.0 x4, Vulkan backend |
| OS | Windows 10 Home |
| Runtime | Ollama 0.40.14 |

`02-bench/specs.ps1` prints this table for your own machine.

## The model

`DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic`, a Qwen3.5 9B (8.95B parameters, 33 layers, one MTP/next-token layer) distilled on DeepSeek V4 Pro reasoning traces, then abliterated. The original repo ships 19 GB of BF16 safetensors. We run the i1-IQ4_XS GGUF quant (5.2 GB) from `mradermacher/DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic-i1-GGUF`.

## Layout

| folder | what it does |
|---|---|
| `01-model/` | `setup.cmd` pulls the GGUF, builds the `dsv4-9b-heretic` variant from `Modelfile`, installs OpenCode |
| `02-bench/` | `bench.py` sweeps GPU layers, context and threads through the Ollama API and writes JSON lines; `model-info.ps1` reads what Ollama actually did (layers offloaded, threads, flash attention) |
| `03-agent/` | the agent experiments: `tasks/` (the phase 4 suite: 12 tasks with hidden tests), `harness/` (task loader and validators, the runner that enforces the time limit on the whole process tree and judges the result itself, the grid driver, the report), `orchestrator/` (the thin orchestrator and its Python indentation repair), `quality/` (phase 3, 40 function-level problems), `agent-test/` (the task of OpenCode run 1) |
| `tools/` | small status probes used while runs are in flight |
| `distill/` | the distillation pipeline (Python, standard library only in its core): sample an open-weight teacher on a subject's tasks, verify the answers by code, filter and de-duplicate the reasoning, write SFT records, train a student, with REA (`tools/rea-bench`) as its first subject; `distill/README.md` says how to add a subject or a student. `tools/modal/` and `tools/kaggle/` keep the commands |
| `skills/` | vendored agent skills for the offline agent, each with `PROVENANCE.md` (source, commit, licence) |
| `results/` | raw outputs of every run (JSONL, logs, the code the agent wrote) |
| `forge/` | the hardware-adaptive layer (Python 3.11+, standard library only, Windows and Linux): `probe` the machine, `plan` a configuration from a GGUF header without running anything, `tune` it with llama-bench, `serve` llama-server with the result |
| `tests/` | unit tests for all of it, with fake llama-bench/llama-server programs and recorded outputs of the real ones in `tests/fixtures/` (`README.md` there says which are real and which are hand-written), plus `mutation_check.py` |
| `docs/JOURNAL.md` | what was tried, in order, with the numbers and the dead ends |

## Rules the setup follows

- The agent's own summary is not evidence. The runner re-runs the tests and hashes the test file after the agent stops.
- One model loaded at a time: two do not fit in 16 GB next to the VM.
- Every config claim is a measurement in `results/`, not a recommendation copied from elsewhere.

## Run it

Needs Python 3.11+ and a llama.cpp build (`llama-bench`, `llama-server`) in a folder; nothing to install. On the Windows host:

```
set LL=C:\Users\arthu\llm\llama-b12207
set MODEL=C:\Users\arthu\.ollama\models\blobs\sha256-55f51ef810a801849ab84f2ae029d1e6142bf83f563983f550d3ce4898750312

python -m forge probe --llama-dir %LL%                              # the machine as JSON (GPU list comes from llama-bench --list-devices)
python -m forge plan  --llama-dir %LL% --model %MODEL% --explain    # gpu / cpu / low-power plans, each rule with its evidence
python -m forge tune  --llama-dir %LL% --model %MODEL% --budget 1200   # measure around the plan, cache the answer, print the command line
python -m forge serve --llama-dir %LL% --model %MODEL% --port 8080  # start llama-server, wait for /health, print the endpoint
```

The endpoint is `http://127.0.0.1:8080/v1` (OpenAI-compatible). `serve` probes again on every start: on battery it takes the low-power plan, with no GPU or no layer fitting it takes the CPU plan, and a tuned configuration is used only after it passes the *server's* memory estimate (llama-server allocates its compute buffer for the full context, llama-bench for a smaller one, so a layer count that loaded in llama-bench can still fail here: 15 layers did, 14 load); if it does not fit, `serve` re-plans on the free memory keeping the tuned attention, KV type and ubatch. If llama-server still cannot load, it retries with one GPU layer fewer at a time (then steps of 4, then a smaller context). `tune` measures the attention/KV settings (flash attention off f16, on f16, on q8_0) and ubatch 512 and 256 (`--ubatches`), only at layer counts the server estimate allows, and stores its result under `%LOCALAPPDATA%\forge` (or `$FORGE_HOME`, or `--cache-dir`), keyed by hardware, model file, llama.cpp build and search version, so the next start is instant; `--force` re-measures.

Speculative decoding is off (`--spec-type none` is passed explicitly). `--spec ngram-mod` (serve, plan) turns on the one variant that helped in phase 2: up to 17.5 tok/s on code it had already seen, but not output-identical to the non-speculative run at temperature 0. `--ubatch N` (plan) sets the physical batch; `plan --explain` shows the compute buffer the GPU plan reserves. `python -m forge heuristics` lists every planning rule and the measurement behind it (or "untested"): the llama.cpp offload order, the compute-buffer model (checked against `llama-fit-params`, `02-bench/fit-params-grid.ps1`), the reserve, the one-layer fallback.

Do not run `tune` while another benchmark holds the GPU: it loads the model. Tests: `python -m unittest discover -s tests -t .`; `python tests/mutation_check.py` breaks one rule at a time in a temporary copy and requires the tests to fail.

### The agent suite and the orchestrator (phase 4)

```
python 03-agent/harness/taskbench.py validate            # reference passes, broken and the start fail: all 12 tasks
python tests/test_agent.py                               # (or: python -m unittest tests.test_agent) runner, orchestrator, report, repair

# the thin orchestrator on any OpenAI-compatible endpoint (here the forge server of the section above), on any repository:
python 03-agent/orchestrator/orchestrate.py --endpoint http://127.0.0.1:8080/v1 --repo path\to\repo ^
       --task-file task.md --test-cmd "[\"python\", \"-m\", \"unittest\"]" --protect tests/test_x.py --result result.json

# one agent on one task from a clean copy, judged by the runner (arms: O = OpenCode alone, T = orchestrator, T-norepair, T-ngram)
python 03-agent/harness/runner.py --arm T --tasks py-wrap --endpoint http://127.0.0.1:8080/v1 --out-dir results/phase4 --time-limit 900

# the whole grid on the host (starts and stops forge serve itself), then the report
python 03-agent/harness/grid.py --stages "O+T@1;T-norepair@1" --out-dir OUT --work-dir WORK --repo . --opencode PATH\to\opencode.exe
python 03-agent/harness/report.py results/phase4 > results/phase4/report.md
```

The orchestrator is the loop, the model only writes files. Per round it sends the task, the repository files, the visible tests and the *first* failing test's trimmed output; the reply is the full new text of at most two files (`FILE: path` + fenced block). Before a file is written, Python goes through a deterministic indentation repair (`pyindent.py`, `--no-repair` switches it off) and a syntax check; the orchestrator runs the tests itself after every edit and never reads the model's summary. When the visible tests pass it asks once whether anything in the task text is still missing (`DONE` ends it), and at the end it restores the best state it saw. Budgets: `--max-calls`, `--retries` (rounds without more tests passing), `--deadline-s`. `tests/test_agent.py` runs all of it against an oracle chat server, no model needed. `03-agent/tasks/README.md` lists the tasks.
