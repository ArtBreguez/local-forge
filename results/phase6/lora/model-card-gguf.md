---
license: apache-2.0
base_model: xtools-at/DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic
datasets:
  - HiddenTerps/local-forge-agent-sft
tags:
  - gguf
  - reverse-engineering
  - uncensored
  - local-llm
  - llama-cpp
language:
  - en
---

# LocalForge-9B-REA-heretic-GGUF

Q4_K_M GGUF of the [LocalForge-9B-REA-heretic](https://huggingface.co/HiddenTerps/LocalForge-9B-REA-heretic) LoRA
(5.63 GB). Converted from the merged safetensors with llama.cpp, F16 → Q4_K_M.

Run with llama.cpp:

```bash
./llama-server -m LocalForge-9B-REA-heretic-Q4_K_M.gguf -ngl 999 -c 32768 -fa off
```

This is the file size the local-forge 9B already ran at on the host (4 GB GPU), so speed and fit are unchanged.

See the main repo for the benchmark numbers, the refusals check, the limits, and the training recipe.
