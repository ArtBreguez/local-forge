# Teacher candidates: open-weight uncensored models

The REA dataset needs a teacher that answers reverse-engineering tasks (keygen, crack, anti-debug) without refusing. A task enters the set only when the teacher's final answer matches the truth computed by the code. In phase A the stored text was rendered from that truth and no model text was stored; in the reasoning run (`modal_rea_data.py --stage sample --reasoning`, `tools/modal/README.md`) the teacher's reply is stored as well, next to the rendered one, with its final answer verified and its steps not checked. Either way any open-weight model can be a teacher, whatever its origin.

This file catalogues the strongest candidates found on 2026-11-11. None of the new ones is measured on our tasks yet; the scores below are third-party. Run `modal_qbench` (40 code problems) and a small `modal_rea_data --stage sample` before trusting any of them.

## How to find them again

`tools/hf_teachers.py` queries the public Hub API (stdlib only, no token):

```
python3 tools/hf_teachers.py                                  # all uncensored models, by downloads
python3 tools/hf_teachers.py --family qwen3.8 glm-5 --min-downloads 1000
python3 tools/hf_teachers.py --info orcarouter/Qwen3.8-27B-Uncensored-FP8   # params, licence, base
python3 tools/hf_teachers.py --gguf GLM-5.3-Flash                           # quantized repacks
```

The same search by hand: https://huggingface.co/models?search=heretic&sort=downloads (also `abliterated`, `uncensored`, `derestricted`, `heresy`). These are the names of the usual refusal-removal methods: heretic (p-e-w's tool), abliteration (huihui-ai and others), Derestricted (ArliAI).

How to read a result:
- **Downloads and likes** show use, not quality. Read the card for a benchmark or a refusal count.
- **Base model** (`--info`) decides the strength. The refusal removal costs a little; the base is what matters.
- **Skip** quantized repacks (`GGUF`, `AWQ`, `FP8` of a model already listed: use them to deploy, not to choose), merges with marketing names (`Turbo-Brilliance-50x-AGI`), LoRA-only repos and image text encoders.
- `--info` reports packed parameters for MXFP4 repos (gpt-oss-120b MXFP4 shows 2.2B); use the base model's size.

Third-party scores: the [UGI Leaderboard](https://huggingface.co/spaces/DontPlanToEnd/UGI-Leaderboard) (uncensored score and reasoning) and its digest at https://hft.md/uncensored-ai/ (103 models). UGI measures willingness and general intelligence, not code or reverse engineering.

## Measured on our tasks

| Model | Code bench (pass@1 / pass@10) | REA set yield | GPU |
|---|---|---|---|
| CodeForge 27B heretic (Qwen3.8 27B base, 4-bit) | 92% / 100% | phase A: 3,187 tasks answered, yield per family in `tools/modal/README.md` ("Next run"); the reasoning run is the next one | L40S |
| Qwen3.5-35B-A3B heretic | 57% / 75% | 735 of 3,187 tasks (23%), seed 7 | L40S |

## Candidates, by the GPU they need

Sizes are total parameters. "Fits" assumes vLLM, 4-bit or FP8, with room for the KV cache.

### Fit the GPU we already use (one L40S, 48 GB)

| Model | Base | Size | Licence | Why |
|---|---|---|---|---|
| [huihui-ai/Huihui-Qwen3.8-27B-abliterated](https://huggingface.co/huihui-ai/Huihui-Qwen3.8-27B-abliterated) | Qwen3.8 27B | 27.8B | Apache-2.0 | The same base as CodeForge (Qwen3.8 27B), abliterated instead of heretic-tuned: not a newer generation. Worth one run to see whether the yield comes from the base or from CodeForge's tuning. |
| [orcarouter/Qwen3.8-27B-Uncensored](https://huggingface.co/orcarouter/Qwen3.8-27B-Uncensored) | Qwen3.8 27B | 27.8B | Apache-2.0 | The same base as CodeForge again, a third refusal-removal method; FP8 and NVFP4 repacks ready. A deployment option for that base, not a new teacher. |
| [koute/GLM-4.7-Flash-Derestricted](https://huggingface.co/koute/GLM-4.7-Flash-Derestricted) | GLM-4.7 Flash | 31.2B | MIT | Different family; a second opinion on the hard families. |
| [coder3101/gemma-4-31B-it-heretic](https://huggingface.co/coder3101/gemma-4-31B-it-heretic) | Gemma 4 31B | 31.3B | Apache-2.0 | UGI #1 of the digest (uncensored 65.7, reasoning 36.5, 10/10 answers). |
| [llmfan46/gemma-4-26B-A4B-it-ultra-uncensored-heretic](https://huggingface.co/llmfan46/gemma-4-26B-A4B-it-ultra-uncensored-heretic) | Gemma 4 26B A4B | 26B, 4B active | Apache-2.0 | Fast (reasoning 35.7); cheap many-sample runs. |
| [llmfan46/Qwen3.6-35B-A3B-uncensored-heretic](https://huggingface.co/llmfan46/Qwen3.6-35B-A3B-uncensored-heretic) | Qwen3.6 35B A3B | 35B, 3B active | Apache-2.0 | Successor of our weaker teacher. |

### One H100 or A100 (80 GB)

| Model | Base | Size | Licence | Why |
|---|---|---|---|---|
| [ArliAI/gpt-oss-120b-Derestricted](https://huggingface.co/ArliAI/gpt-oss-120b-Derestricted) ([MXFP4 repack](https://huggingface.co/justinjja/gpt-oss-120b-Derestricted-MXFP4)) | gpt-oss 120B | 117B, 5.1B active | Apache-2.0 | Strong at code; MXFP4 (~63 GB) fits one H100. |
| [ArliAI/GLM-4.5-Air-Derestricted](https://huggingface.co/ArliAI/GLM-4.5-Air-Derestricted) | GLM-4.5 Air | 110B, 12B active | MIT | Best reasoning in the digest (36.6). |
| [MuXodious/gpt-oss-120b-tainted-heresy](https://huggingface.co/MuXodious/gpt-oss-120b-tainted-heresy) | gpt-oss 120B | 120B, 5.1B active | Apache-2.0 | Alternative method (reasoning 32.3). |
| [MuXodious/gpt-4o-distil-Llama-3.3-70B-Instruct-PaperWitch-heresy](https://huggingface.co/MuXodious/gpt-4o-distil-Llama-3.3-70B-Instruct-PaperWitch-heresy) | Llama 3.3 70B (GPT-4o distill) | 70B | not stated | Fallback (reasoning 30.6). |

### Several GPUs (catalogued, not planned)

| Model | Base | Size | Licence | Hardware |
|---|---|---|---|---|
| [orcarouter/Qwen3.8-Flash-Next-Uncensored-FP8](https://huggingface.co/orcarouter/Qwen3.8-Flash-Next-Uncensored-FP8) | Qwen3.8 Flash-Next | 180B | Apache-2.0 | 2–3 H100 (FP8); GGUF repacks exist |
| [scrunter/Qwen3-VL-235B-A22B-Thinking-heretic](https://huggingface.co/scrunter/Qwen3-VL-235B-A22B-Thinking-heretic) | Qwen3-VL 235B A22B | 235B, 22B active | not stated | 2 H100 (4-bit) |
| [prem-research/DeepSeek-V4-Flash-0731-abliterated](https://huggingface.co/prem-research/DeepSeek-V4-Flash-0731-abliterated) | DeepSeek V4 Flash | 304B | MIT | 4 H100 (FP8) |
| [orcarouter/GLM-5.3-Flash-Uncensored-FP8](https://huggingface.co/orcarouter/GLM-5.3-Flash-Uncensored-FP8) ([huihui GGUF](https://huggingface.co/huihui-ai/Huihui-GLM-5.3-Flash-abliterated-GGUF)) | GLM-5.3 Flash | 321B | MIT | 4–8 H100 |
| [ArliAI/GLM-4.6-Derestricted-v3](https://huggingface.co/ArliAI/GLM-4.6-Derestricted-v3) | GLM-4.6 | 357B | MIT | 8 H100 (FP8) |
| [audnai/penclaw-GLM-5.3-abliterated](https://huggingface.co/audnai/penclaw-GLM-5.3-abliterated) | GLM-5.3 | 753B | other | 8+ H100 |
| [dealignai/DeepSeek-V4.1-Flash-1017-UNCENSORED-FP8](https://huggingface.co/dealignai/DeepSeek-V4.1-Flash-1017-UNCENSORED-FP8) | DeepSeek V4.1 Flash | 763B | MIT | 8+ H100; GGUF repacks exist |
| [Uniboshi/Kimi-K3-Abliterated-V1](https://huggingface.co/Uniboshi/Kimi-K3-Abliterated-V1) | Kimi K3 | 2.78T | other | multi-node; GGUF repacks exist |

## Left out

- Roleplay and writing tunes (Behemoth, Anubis, Euryale, Cydonia, Assistant_Pepe, Stylizer): high UGI scores, weak at code.
- Merges with no evaluation (DavidAU "Turbo-Brilliance", "Cold-Fusion" and similar).
- Qwen2.5-Coder heretic: an old generation.
- Small edge models (Gemma 4 E2B/E4B, 9B distills): too weak for the hard families.

## Next step

Run the 40-problem code bench on the first three L40S rows and gpt-oss-120b-Derestricted against CodeForge, then a small sample run of the best one on the families where yield is low (cipher, multi, fsm). Add the measured rows to the table above.
