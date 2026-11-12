"""The teacher models: one list for the Modal sampler, the Modal benchmarks and the Kaggle kernel. Stdlib only.

A teacher is an open-weight GGUF served by llama.cpp. THE RULE (the owner's), by what the run stores:
  - default mode: any open-weight teacher may FILTER, whatever its origin. The teacher's answer only decides whether a task is kept; the record is rendered from the truth, and none of the teacher's text is stored.
  - reasoning and thinking modes: the teacher's reply is stored verbatim (`target_source: "teacher-reasoning"`), so only a teacher that was NOT trained by or distilled from an OpenAI or Anthropic model may
    be used. gpt-oss-* (OpenAI's own models) and the GPT-4o distill (docs/TEACHER_CANDIDATES.md) are filter-only.
`Teacher.reasoning_ok` is that rule as data, and it defaults to False: a teacher is added without it until someone has looked at where it comes from. `require_reasoning` is what the Modal sample stage
(`--reasoning` / `--thinking`) and the Kaggle kit call, and what refuses a teacher without it. `gpu` is the Modal GPU class that holds the teacher. The list was `MODELS` of `tools/modal/modal_qbench.py`,
which re-exports it under that name; entries are tuples (name, repo, file, gpu, reasoning_ok): code that unpacks four names takes `*_`.
"""
from __future__ import annotations

from typing import NamedTuple


class Teacher(NamedTuple):
    name: str
    repo: str       # Hugging Face repo of the GGUF
    file: str
    gpu: str        # Modal GPU class: "L4" or "L40S"
    reasoning_ok: bool = False      # may its text be STORED (reasoning / thinking modes)? Only if it was not trained by or distilled from an OpenAI or Anthropic model. False = filter only.


# licences: all Apache-2.0 except the heretic merge (see its card)
# reasoning_ok: the Qwen and unsloth entries are Alibaba's models and their quantizations; heretic-9b is a distill of DeepSeek-V4-Pro onto Qwen3.5-9B; heretic-35b-* abliterate Qwen3.5-35B-A3B.
# PROVENANCE OPEN, codeforge-*: it is True because the catalogue has always carried it and the owner's rule names only gpt-oss-* and the GPT-4o distill, but the card of its base,
# xtools-at/Qwen3.8-27B-CodeForge-3xMerge, lists three merged fine-tunes, and the cards of two of them speak of OpenAI/Anthropic-oriented data (nerkyor's Coder390 EfficientThink: "Opus 5.5 and GPT-6
# Astra ... contributed gold answers and teacher trajectories"; TeichAI's Fable-Distill: "Claude/Fable-oriented coding traces"; read as of 2026-11-12 through a summarizing fetch, not by hand).
# If the owner reads that as "trained by or distilled from" an OpenAI/Anthropic model, set it to False: every reasoning and thinking run with CodeForge is then refused, and so is the shipped config.
TEACHERS = [
    Teacher("heretic-9b", "mradermacher/DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic-i1-GGUF",
            "DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic.i1-IQ4_XS.gguf", "L4", True),
    Teacher("qwen35-9b", "unsloth/Qwen3.5-9B-GGUF", "Qwen3.5-9B-IQ4_XS.gguf", "L4", True),
    Teacher("qwen25-coder-7b", "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF", "qwen2.5-coder-7b-instruct-q4_k_m.gguf", "L4", True),
    Teacher("qwen3-coder-30b-a3b", "unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF", "Qwen3-Coder-30B-A3B-Instruct-UD-IQ3_XXS.gguf", "L4", True),
    Teacher("qwen35-35b-a3b", "unsloth/Qwen3.5-35B-A3B-GGUF", "Qwen3.5-35B-A3B-UD-IQ3_XXS.gguf", "L4", True),
    Teacher("qwen35-27b", "unsloth/Qwen3.5-27B-GGUF", "Qwen3.5-27B-Q4_K_M.gguf", "L40S", True),
    # heretic (abliterated, uncensored) versions of the bigger models, same quant sizes as above where possible
    Teacher("qwen35-35b-a3b-heretic", "mradermacher/Qwen3.5-35B-A3B-heretic-i1-GGUF", "Qwen3.5-35B-A3B-heretic.i1-IQ3_XXS.gguf", "L4", True),
    Teacher("qwen35-35b-a3b-heretic-v2", "mradermacher/Qwen3.5-35B-A3B-heretic-v2-i1-GGUF", "Qwen3.5-35B-A3B-heretic-v2.i1-IQ3_XXS.gguf", "L4", True),
    Teacher("codeforge-27b-heretic-iq3", "mradermacher/Qwen3.8-27B-CodeForge-3xMerge-UNCENSORED-heretic-i1-GGUF",
            "Qwen3.8-27B-CodeForge-3xMerge-UNCENSORED-heretic.i1-IQ3_XXS.gguf", "L40S", True),
    Teacher("codeforge-27b-heretic-q4", "mradermacher/Qwen3.8-27B-CodeForge-3xMerge-UNCENSORED-heretic-i1-GGUF",
            "Qwen3.8-27B-CodeForge-3xMerge-UNCENSORED-heretic.i1-Q4_K_M.gguf", "L40S", True),
]

# Names, repos and files that say the model is OpenAI's or Anthropic's, or a distill of one: whatever its flag says, a catalogue entry with one of these in it is filter-only (a test pins it).
FILTER_ONLY_MARKERS = ("gpt-oss", "gpt-4", "gpt4", "gpt-5", "gpt-6", "chatgpt", "claude", "opus", "sonnet", "haiku", "fable")


def find(name: str) -> Teacher:
    """The teacher called `name`; SystemExit naming the known ones otherwise."""
    for t in TEACHERS:
        if t.name == name:
            return t
    raise SystemExit(f"unknown model {name}; known: {[t.name for t in TEACHERS]}")


def marked_filter_only(t: Teacher) -> bool:
    """True when the name, repo or file of the entry names an OpenAI or Anthropic model (FILTER_ONLY_MARKERS): the flag cannot be True for it."""
    text = f"{t.name} {t.repo} {t.file}".lower()
    return any(m in text for m in FILTER_ONLY_MARKERS)


def matching(name: str = "", repo: str = "", file: str = "") -> list:
    """The catalogue entries that are this teacher: the one called `name`, and every one with this `repo` AND `file` (the same GGUF under another name is the same teacher)."""
    return [t for t in TEACHERS if (name and t.name == name) or (repo and file and (t.repo, t.file) == (repo, file))]


def require_reasoning(name: str = "", repo: str = "", file: str = "", strict: bool = False, why: str = "--reasoning / --thinking") -> None:
    """SystemExit unless the teacher may have its text stored (`why` names what stores it). A catalogue entry that matches by name or by repo and file and lacks `reasoning_ok` refuses. A model that is in
    no entry is refused only with `strict` (the Kaggle kit, where `--model-repo` can name any GGUF: add it to distill/teachers.py, with its provenance, first); the Modal stage looks the name up before
    it gets here, and its generic body, which other subjects call with their own names, refuses what the catalogue says is filter-only."""
    found = matching(name, repo, file)
    refused = [t for t in found if not t.reasoning_ok or marked_filter_only(t)]
    if refused:
        raise SystemExit(f"{why} stores the teacher's reply verbatim, and {', '.join(t.name for t in refused)} may only FILTER (default mode: its text is not stored): it is an OpenAI or Anthropic model or a distill of one "
                         f"(`reasoning_ok` is False in distill/teachers.py; docs/TEACHER_CANDIDATES.md has the rule). Teachers that may: {[t.name for t in TEACHERS if t.reasoning_ok and not marked_filter_only(t)]}")
    if strict and not found:
        raise SystemExit(f"{why} stores the teacher's reply verbatim, and {name or repo + '/' + file} is in no entry of distill/teachers.py: add it there, and set `reasoning_ok` only when it was not trained by or "
                         f"distilled from an OpenAI or Anthropic model (docs/TEACHER_CANDIDATES.md)")
