"""The teacher models: one list for the Modal sampler, the Modal benchmarks and the Kaggle kernel. Stdlib only.

A teacher is an open-weight GGUF served by llama.cpp. THE RULE (the owner's), by what the run stores:
  - default mode: any open-weight teacher may FILTER, whatever its origin. The teacher's answer only decides whether a task is kept; the record is rendered from the truth, and none of the teacher's text is stored.
  - reasoning and thinking modes: the teacher's reply is stored verbatim (`target_source: "teacher-reasoning"`), and the dataset rule is that no DIRECT OpenAI or Anthropic output is stored: gpt-oss-*
    (OpenAI's own models) and the GPT-4o distill (docs/TEACHER_CANDIDATES.md) are filter-only. An open-weight model whose lineage includes fine-tunes on such data may be accepted case by case by the
    owner, with the decision recorded (CodeForge, below, and docs/TEACHER_CANDIDATES.md). A dataset card that publishes teacher-reasoning records must state the teacher's lineage.
`Teacher.reasoning_ok` is that rule as data, and it defaults to False: a teacher is added without it until someone has looked at where it comes from and the owner has decided. `require_reasoning` is what the Modal sample stage
(`--reasoning` / `--thinking`) and the Kaggle kit call, and what refuses a teacher without it. `gpu` is the Modal GPU class that holds the teacher. The list was `MODELS` of `tools/modal/modal_qbench.py`,
which re-exports it under that name; entries are tuples (name, repo, file, gpu, reasoning_ok): code that unpacks four names takes `*_`.
"""
from __future__ import annotations

import re
from typing import NamedTuple


class Teacher(NamedTuple):
    name: str
    repo: str       # Hugging Face repo of the GGUF
    file: str
    gpu: str        # Modal GPU class: "L4" or "L40S"
    reasoning_ok: bool = False      # may its text be STORED (reasoning / thinking modes)? Not if it is an OpenAI or Anthropic model or a distill of one; a lineage that includes fine-tunes on such data is the owner's call. False = filter only.


# licences: all Apache-2.0 except the heretic merge (see its card)
# reasoning_ok: the Qwen and unsloth entries are Alibaba's models and their quantizations; heretic-9b is a distill of DeepSeek-V4-Pro onto Qwen3.5-9B; heretic-35b-* abliterate Qwen3.5-35B-A3B.
# ACCEPTED BY THE OWNER on 2026-11-12, codeforge-*: `reasoning_ok=True` is the owner's decision, and this is its record. The card of its base, xtools-at/Qwen3.8-27B-CodeForge-3xMerge, merges three fine-tunes:
# nerkyor's Coder390 EfficientThink (teacher trajectories of Opus 5.5 and GPT-6 Astra), TeichAI's Fable Distill (claude-fable-5-claude-code and Fable-5-Chat) and Jackrong's Qwopus Coder Flash (Qwen, DeepSeek and
# NVIDIA teachers). Its lineage includes fine-tunes on Anthropic and OpenAI model output; it is not one of their models and none of their output is stored directly, which is what the dataset rule forbids.
# A dataset that publishes its replies states this lineage (docs/TEACHER_CANDIDATES.md).
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
# The markers are matched against the text with its separators dropped (`squashed`: lower case, no `-`, `_`, `.` or space), so each one is written without them: "gptoss" is gpt-oss, gpt_oss, GPT.OSS and
# gptoss; "gpt5" is gpt-5, gpt5 and gpt-5.1 (round 13, m1: the hyphen spellings alone let `gptoss`, `gpt_oss`, `gpt5` and `gpt6` through).
FILTER_ONLY_MARKERS = ("gptoss", "gpt4", "gpt5", "gpt6", "chatgpt", "claude", "opus", "sonnet", "haiku", "fable")


def squashed(text: str) -> str:
    """`text` in lower case without `-`, `_`, `.` and spaces: what the markers are looked for in, so that the spelling of a name (gpt-oss, gpt_oss, GPT.OSS, gptoss) does not decide."""
    return re.sub(r"[-_. ]", "", text.lower())


def said_filter_only(*texts: str) -> list:
    """The FILTER_ONLY_MARKERS that any of `texts` (the name, the repo, the file) says. Each text is read by itself: a marker is not made of the end of one and the start of the next."""
    flat = [squashed(t) for t in texts]
    return [m for m in FILTER_ONLY_MARKERS if any(m in t for t in flat)]


def find(name: str) -> Teacher:
    """The teacher called `name`; SystemExit naming the known ones otherwise."""
    for t in TEACHERS:
        if t.name == name:
            return t
    raise SystemExit(f"unknown model {name}; known: {[t.name for t in TEACHERS]}")


def marked_filter_only(t: Teacher) -> bool:
    """True when the name, repo or file of the entry names an OpenAI or Anthropic model (FILTER_ONLY_MARKERS): the flag cannot be True for it."""
    return bool(said_filter_only(t.name, t.repo, t.file))


def matching(name: str = "", repo: str = "", file: str = "") -> list:
    """The catalogue entries that are this teacher: the one called `name`, and every one with this `repo` AND `file` (the same GGUF under another name is the same teacher)."""
    return [t for t in TEACHERS if (name and t.name == name) or (repo and file and (t.repo, t.file) == (repo, file))]


def require_reasoning(name: str = "", repo: str = "", file: str = "", strict: bool = False, why: str = "--reasoning / --thinking") -> None:
    """SystemExit unless the teacher may have its text stored (`why` names what stores it). It judges what was GIVEN as well as the catalogue entries it finds, because the name, the repo and the file can
    disagree and a caller can pass any of them (round 12, M2: `--teacher qwen35-9b --model-repo unsloth/gpt-oss-120b-GGUF --model-file ...` used to pass on the name alone):
      - `name` is an entry and `repo` and `file` are both given: they must be that entry's own GGUF, else "name and GGUF disagree" (a catalogue name that may store text does not carry another GGUF);
      - an entry that matches by name or by repo and file and lacks `reasoning_ok` refuses, and so does one whose text names an OpenAI or Anthropic model (FILTER_ONLY_MARKERS);
      - the text given (name, repo, file) is read for those markers too, whatever the spelling (`squashed`: gpt-oss, gpt_oss, GPT.OSS, gptoss), so a GGUF that is in no entry but is called gpt-oss-... is refused, with a name or
        without one;
      - a model that is in no entry is refused only with `strict` (the Kaggle kit, where `--model-repo` can name any GGUF: add it to distill/teachers.py, with its provenance, first); the Modal stage looks the
        name up before it gets here, and its generic body, which other subjects call with their own names, refuses what the catalogue says is filter-only."""
    named = [t for t in TEACHERS if name and t.name == name]
    if named and repo and file and (named[0].repo, named[0].file) != (repo, file):
        raise SystemExit(f"{why} stores the teacher's reply verbatim, and the name and the GGUF disagree: {name} is {named[0].repo}/{named[0].file}, not {repo}/{file}. Give the name alone, or add that GGUF to "
                         f"distill/teachers.py under its own name, with `reasoning_ok` set by the rule (docs/TEACHER_CANDIDATES.md)")
    found = matching(name, repo, file)
    refused = [t for t in found if not t.reasoning_ok or marked_filter_only(t)]
    said = said_filter_only(name, repo, file)
    if refused or said:
        who = ", ".join(t.name for t in refused) or name or f"{repo}/{file}"
        raise SystemExit(f"{why} stores the teacher's reply verbatim, and {who} may only FILTER (default mode: its text is not stored): it is an OpenAI or Anthropic model or a distill of one "
                         f"(`reasoning_ok` is False in distill/teachers.py, or its name, repo or file says {', '.join(said) or 'so'}; docs/TEACHER_CANDIDATES.md has the rule). "
                         f"Teachers that may: {[t.name for t in TEACHERS if t.reasoning_ok and not marked_filter_only(t)]}")
    if strict and not found:
        raise SystemExit(f"{why} stores the teacher's reply verbatim, and {name or repo + '/' + file} is in no entry of distill/teachers.py: add it there, and set `reasoning_ok` only when it was not trained by or "
                         f"distilled from an OpenAI or Anthropic model, or after the owner has accepted its lineage case by case (docs/TEACHER_CANDIDATES.md)")
