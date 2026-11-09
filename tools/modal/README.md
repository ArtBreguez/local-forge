# Modal pipeline (rented GPU lab)

Everything local-forge ran on [Modal](https://modal.com): model comparison, the REA benchmark, the REA training data, the LoRA, the refusals check and the Hugging Face upload. The host (4 GB GPU) only measures speed and fit; quality is measured here, every model in the same container, same llama.cpp build (pinned by digest), same prompts, temperature 0.

Numbers from Modal (CUDA, llama.cpp b12211) are not directly comparable with the host (Vulkan, b12207): compare models within one environment.

## Setup (once)

```bash
uv tool install modal
modal token set --token-id <id> --token-secret <secret> --profile <name>   # type it yourself, never paste tokens in chat
modal profile activate <name>
# Hugging Face write token, for the gated/private dataset and the model upload:
read -rs HF; modal secret create hf-write HF_TOKEN="$HF"; unset HF
```

Volumes are created on first use: `lf-models` (downloaded GGUFs, checked against SHA-256), `lf-lora` (training output and our own GGUFs).

## The steps, in order

| # | Script | What it does | Writes | Cost seen (Nov 2026) |
|---|---|---|---|---|
| 1 | `modal_qbench.py` | 40 coding problems with hidden tests (same runner as phase 3, `03-agent/quality`): first try at temp 0 and 10 samples at temp 0.7, per model in `MODELS` | `results/phase6/modal/` | US$ 3–4 per 6 models |
| 2 | `modal_rea.py` | REA benchmark: 10 fixed questions on one stripped binary (`rea-bench/targets.c`), recipe arm (Ghidra pseudocode given) and agent arm (`funcs`, `decomp`, `decomp-all`, `refdata` tools, 12 steps), plus the 60 generated held-out tasks (seed 1001) | `results/phase6/rea/` | ~US$ 1 per model |
| 3 | `modal_rea_data.py --stage decompile` | Generate tasks with `rea-bench/gen_rea_tasks.py` (10 kinds, automatic answer key), compile stripped, decompile with Ghidra, resolve referenced bytes | `results/phase6/rea-data/decompiled-seed*.jsonl` | ~US$ 1 per 1,200 |
| 4 | `modal_rea_data.py --stage sample --model <teacher> --k 4` | The teacher (uncensored, e.g. CodeForge 27B heretic Q4) answers each task k times; only answers that match the key are kept | `results/phase6/rea-data/sft-*.jsonl` | ~US$ 5 per 4,800 answers |
| 5 | (upload) | The SFT file went to the private dataset `HiddenTerps/local-forge-agent-sft` (`rea/sft_train.jsonl`, `rea/tasks_*.jsonl`) | Hugging Face | — |
| 6 | `modal_lora.py [--smoke]` | LoRA (rank 32, all attention/MLP projections) on the base safetensors, merge, convert and quantize **both** the trained and the base model to Q4_K_M by the same path | volume `lf-lora`, `results/phase6/lora/*.json` | ~US$ 2–3 (A100, 44 min) |
| 7 | `modal_rea.py --only heretic-9b-base-q4km,heretic-9b-rea-lora-q4km` | Before/after on the REA benchmark, both GGUFs from step 6 | `results/phase6/rea/` | ~US$ 1 |
| 8 | `modal_censor.py` | Refusals check on base and trained (the model must stay uncensored) | stdout, `results/phase6/censor.log` | < US$ 0.5 |
| 9 | `modal_hf_upload.py` | Upload merged weights + LoRA adapter, and the Q4_K_M GGUF, to private repos | Hugging Face | CPU only |

Run each with `modal run tools/modal/<script>.py [options]`; `--help` lists the options. Long runs are best started detached from the shell (e.g. `systemd-run --user --unit=modal-x ...`) so a dropped session does not kill them.

## Rules the scripts keep

- Only uncensored candidates are chosen; censored models appear only as reference rows.
- Held-out tasks (seed 1001) and the 10 fixed questions never enter training.
- Teacher answers enter the data only when they match the automatic answer key.
- No outputs of OpenAI or Anthropic models in the data.
- GGUFs downloaded by SHA-256; llama.cpp image pinned by digest.

## Results so far

- Coding (40 problems, first try / 10 samples): CodeForge 27B heretic Q4 92% / 100%; 9B heretic 25% / 68%.
- REA, 9B heretic, base vs LoRA (same Q4_K_M path): held-out 34/60 → **50/60** (McNemar exact p = 0.0001), fixed questions 9/10 → 10/10, agent arm 4/10 → 5/10. Gains concentrated in LCG (0→6/6), magic constant (2→6/6) and rotate (2→6/6); hash (1/6) and threshold (0→2/6) still weak.
- Limits: 10 generated patterns, one function per binary, x86-64 Linux/gcc only. Not a general reverse-engineering model.

The training data itself (`results/phase6/rea-data/*.jsonl`) is not in the repo; it lives in the private dataset above.
