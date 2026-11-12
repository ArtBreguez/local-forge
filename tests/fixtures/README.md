# Fixtures: what is real and what is not

Recorded on this project's machines (Ryzen 5 3600 / RX 6500 XT Windows host, Linux VM), nothing edited except where noted:

| file | origin |
|---|---|
| `list-devices-rx6500xt-idle.txt` | `llama-bench --list-devices` b12207, the `Available devices:` block as logged by sweep C before it started (3443 MiB free) |
| `list-devices-rx6500xt-busy.txt` | full output of the same command run while a benchmark held the GPU (3105 MiB free) |
| `list-devices-cpu-only.txt` | same command with `GGML_VK_VISIBLE_DEVICES=99` so the Vulkan backend finds no device: `Available devices:` / `(none)` |
| `llama-bench-sweepc-snapshot.jsonl` / `.log` | llama-bench `-o jsonl` rows and the per-call log of sweep C (`02-bench/sweep-c.ps1`), copied after all 24 calls (8 layer counts x 3 attention/KV settings) finished but before the `exit=` line appeared; includes the load/context failures at ngl >= 20 |
| `llama-bench-oom-stderr.txt` | stderr of a real failed call (ngl=99), converted from UTF-16 to UTF-8, PowerShell error preamble removed |
| `llama-bench-help-b12207.txt`, `llama-server-help-b12207.txt` | `--help` of the b12207 binaries |
| `windows-cim-processor.json`, `windows-cim-videocontroller.json` | PowerShell `Get-CimInstance ... \| ConvertTo-Json -Compress` on the Windows host |
| `thinking-pilot/replies.jsonl`, `tasks.jsonl` | 24 replies and the 16 decompiled tasks (seed 7) they answer, copied unedited from the output of the Kaggle thinking pilot (2026-11-11: CodeForge 27B heretic Q4, 57 tasks x 2, 4096 tokens; `~/work/kaggle-rea/out2/replies.jsonl`): the 12 accepted replies the repetition filter dropped for quoting the question, 6 replies that are real loops, the 6 antidebug replies whose `Answer:` line had no technique (`case` says which); `tests/test_rea_thinking_repetition.py` (the 12 and the 6), `tests/test_rea_thinking_prompt.py` (the antidebug 6) |
| `linux-vm-meminfo.txt`, `linux-vm-cpuinfo.txt` | `/proc/meminfo`, `/proc/cpuinfo` of the Linux VM |
| `gguf-dump-dsv4-9b.txt`, `modelinfo-dsv4-9b.json` | the real GGUF header parsed on the host by `forge/gguf.py` (metadata dump / `forge plan --dump-model`; the JSON was regenerated on the host with the fixed code and has the vocabulary size, `n_vocab` 248320) |

**Hand-written, not recorded** (no such hardware here): `nvidia-smi-sample.csv` follows the documented
`--query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits` format; the CUDA/ROCm/Metal
device lines and the `pmset`/`power_supply` samples inside `tests/test_probe.py` are written by hand from the
formats llama.cpp, Linux and macOS document.
