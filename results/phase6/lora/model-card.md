---
license: apache-2.0
base_model: xtools-at/DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic
datasets:
  - HiddenTerps/local-forge-agent-sft
tags:
  - reverse-engineering
  - uncensored
  - lora
  - llama-cpp
  - local-llm
language:
  - en
---

# LocalForge-9B-REA-heretic

A LoRA fine-tune of the uncensored `DeepSeek-V4-Pro-Qwen3.5-9B-heretic` base on the
[local-forge](https://github.com/ArtBreguez/local-forge) reverse-engineering dataset, for the **recipe arm** of the
local-forge REA benchmark: given Ghidra pseudocode of one function and a question, return the concrete value or name
asked for. It is not a general reverse-engineering model (see Limits).

## What changed

LoRA (rank 32, alpha 64, dropout 0.05) on `q_proj k_proj v_proj o_proj gate_proj up_proj down_proj in_proj_qkvz in_proj_ba out_proj`,
2 epochs, lr 1.5e-4 cosine, bf16, batch 4 × grad-accum 4. 3,646 examples, 456 steps, loss 0.70 → 0.24.
Trained on an A100-40GB in ~44 min.

| REA benchmark (same Q4_K_M path for both) | base | **LoRA** |
|---|---|---|
| 60 held-out tasks (seed 1001, never trained on) | 34/60 (57%) | **50/60 (83%)** |
| 10 fixed questions | 9/10 | **10/10** |
| agent arm (tool use) | 4/10 | **5/10** |

McNemar exact p = 0.0001. Gains concentrate in LCG (0/6 → 6/6), magic constant (2/6 → 6/6), rotate (2/6 → 6/6).
Still weak: hash family (1/6) and threshold (0/6 → 2/6).

The refusals check (10 requests a censored model would refuse, incl. keygen, trial-bypass, watermark removal,
shellcode, meth synthesis, bomb, phishing, explicit content, ethnic joke): **10/10 answered, 0 refused** — the base was
already uncensored and the LoRA did not bring any refusal back.

## Files

- Merged safetensors (this repo), `lora-adapter/` (the PEFT adapter for re-training).
- Quantized Q4_K_M GGUF in [LocalForge-9B-REA-heretic-GGUF](https://huggingface.co/HiddenTerps/LocalForge-9B-REA-heretic-GGUF).

## Training data

`HiddenTerps/local-forge-agent-sft` (private) — `rea_sft` split, 3,646 examples. Generated from compiled-stripped
binaries, decompiled with Ghidra, with an automatic answer key. Teacher answers enter only when they match the key.
The 60 held-out tasks (seed 1001) and the 10 fixed questions are never in the training set.

## Limits (honest)

- 10 generated patterns, one function per binary, x86-64 Linux/gcc only.
- The agent arm improved only 4→5/10; the LoRA helps the recipe arm, not autonomous multi-step tool use.
- Trained with `enable_thinking=False` chat template; best used with thinking disabled.

## Reproduction

```bash
modal run tools/modal/modal_lora.py --epochs 2 --rank 32   # from the local-forge repo, branch with tools/modal
```
