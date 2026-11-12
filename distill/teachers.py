"""The teacher models: one list for the Modal sampler, the Modal benchmarks and the Kaggle kernel. Stdlib only.

A teacher is an open-weight GGUF served by llama.cpp (the rule of the pipeline: no output of an OpenAI or Anthropic model in a dataset). `gpu` is the Modal GPU class that holds it. The list was
`MODELS` of `tools/modal/modal_qbench.py`, which re-exports it under that name; entries are tuples (name, repo, file, gpu) as before.
"""
from __future__ import annotations

from typing import NamedTuple


class Teacher(NamedTuple):
    name: str
    repo: str       # Hugging Face repo of the GGUF
    file: str
    gpu: str        # Modal GPU class: "L4" or "L40S"


# licences: all Apache-2.0 except the heretic merge (see its card)
TEACHERS = [
    Teacher("heretic-9b", "mradermacher/DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic-i1-GGUF",
            "DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic.i1-IQ4_XS.gguf", "L4"),
    Teacher("qwen35-9b", "unsloth/Qwen3.5-9B-GGUF", "Qwen3.5-9B-IQ4_XS.gguf", "L4"),
    Teacher("qwen25-coder-7b", "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF", "qwen2.5-coder-7b-instruct-q4_k_m.gguf", "L4"),
    Teacher("qwen3-coder-30b-a3b", "unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF", "Qwen3-Coder-30B-A3B-Instruct-UD-IQ3_XXS.gguf", "L4"),
    Teacher("qwen35-35b-a3b", "unsloth/Qwen3.5-35B-A3B-GGUF", "Qwen3.5-35B-A3B-UD-IQ3_XXS.gguf", "L4"),
    Teacher("qwen35-27b", "unsloth/Qwen3.5-27B-GGUF", "Qwen3.5-27B-Q4_K_M.gguf", "L40S"),
    # heretic (abliterated, uncensored) versions of the bigger models, same quant sizes as above where possible
    Teacher("qwen35-35b-a3b-heretic", "mradermacher/Qwen3.5-35B-A3B-heretic-i1-GGUF", "Qwen3.5-35B-A3B-heretic.i1-IQ3_XXS.gguf", "L4"),
    Teacher("qwen35-35b-a3b-heretic-v2", "mradermacher/Qwen3.5-35B-A3B-heretic-v2-i1-GGUF", "Qwen3.5-35B-A3B-heretic-v2.i1-IQ3_XXS.gguf", "L4"),
    Teacher("codeforge-27b-heretic-iq3", "mradermacher/Qwen3.8-27B-CodeForge-3xMerge-UNCENSORED-heretic-i1-GGUF",
            "Qwen3.8-27B-CodeForge-3xMerge-UNCENSORED-heretic.i1-IQ3_XXS.gguf", "L40S"),
    Teacher("codeforge-27b-heretic-q4", "mradermacher/Qwen3.8-27B-CodeForge-3xMerge-UNCENSORED-heretic-i1-GGUF",
            "Qwen3.8-27B-CodeForge-3xMerge-UNCENSORED-heretic.i1-Q4_K_M.gguf", "L40S"),
]


def find(name: str) -> Teacher:
    """The teacher called `name`; SystemExit naming the known ones otherwise."""
    for t in TEACHERS:
        if t.name == name:
            return t
    raise SystemExit(f"unknown model {name}; known: {[t.name for t in TEACHERS]}")
