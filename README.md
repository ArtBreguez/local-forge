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
| `03-agent/` | a fixed coding task (`agent-test/`: implement `parseDuration` so 5 tests pass), the OpenCode config, and the runner that times the agent and counts passing tests itself |
| `tools/` | small status probes used while runs are in flight |
| `results/` | raw outputs of every run (JSONL, logs, the code the agent wrote) |
| `docs/JOURNAL.md` | what was tried, in order, with the numbers and the dead ends |

## Rules the setup follows

- The agent's own summary is not evidence. The runner re-runs the tests and hashes the test file after the agent stops.
- One model loaded at a time: two do not fit in 16 GB next to the VM.
- Every config claim is a measurement in `results/`, not a recommendation copied from elsewhere.
