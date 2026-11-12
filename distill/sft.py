"""How one SFT record becomes a training example, for any trainer. Stdlib only: the tokenizer is passed in, so the Modal trainer (`backends/modal_train.py`) and the Kaggle kernel
(`backends/kaggle/lora_eval.py`, which ships this file in its dataset) run the SAME code and cannot drift apart. Moved from `backends/modal_train.py` unchanged (that module imports every name below,
so `modal_train.parse_assistant`, `modal_train.encode_record`, ... are still there).

What the code below decides, per RECORD: a plain record (no think tag), a thinking record (`<think>reasoning</think>` then an answer) or a malformed one (dropped); the prompt is the chat
template's generation prompt (no loss), the target is the rest plus EOS (loss), a pair over the window is dropped and never cut. `modal_train.py`'s docstring has the full account of the thinking
format and of `--mix`.
"""
from __future__ import annotations

from .config import MAX_LEN

# The LoRA target modules of the student (Qwen3.5 family: attention, MLP and the linear-attention projections). One list for every trainer.
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "in_proj_qkvz", "in_proj_ba", "out_proj"]


def encode_ids(p_ids: list, f_ids: list, max_len: int = MAX_LEN):
    """The training example for one SFT pair, or None when it must not be trained on. `p_ids`: the prompt (system + user + generation prompt), `f_ids`: the prompt,
    the answer and the EOS token. Only the answer is trained (labels -100 on the prompt). A pair longer than `max_len` is DROPPED, never cut: a cut pair has no
    EOS, so the model would be taught an answer that does not end, and a prompt that alone fills the window leaves nothing to learn from (round 8, m12: the
    encoder used to slice `f_ids[:2048]` without a word, and the length of a task's prompt is not bounded by anything in its generator)."""
    if len(f_ids) > max_len or len(f_ids) <= len(p_ids):
        return None
    return {"input_ids": list(f_ids), "attention_mask": [1] * len(f_ids), "labels": [-100] * len(p_ids) + list(f_ids[len(p_ids):])}


# ---- thinking records (round 13) ---------------------------------------------------------------------------------------------------------------------

THINK_OPEN, THINK_CLOSE = "<think>", "</think>"
THINK_GENERATION_PROMPT = "<think>\n"               # what the base template appends to the prompt for `enable_thinking=True` (or not given): the model writes its reasoning next
EMPTY_THINK = "<think>\n\n</think>\n\n"             # ... for `enable_thinking=False`: no reasoning, the answer follows
FATES = ("think", "plain", "skipped_plain", "malformed")      # what becomes of a record: trained as a thinking record, as a plain one, skipped (plain record of a thinking config, no --mix), dropped (bad tags)


def parse_assistant(text: str) -> tuple:
    """(kind, parts) of the assistant text of a record, read the way the base chat template reads an assistant message: `content.split('</think>')[0]...` for the reasoning and
    `content.split('</think>')[-1].lstrip('\\n')` for the answer, after a trim of the whole text.
      ("plain", None)                 no think tag at all (a truth-template record, a visible-reasoning one, phase A's)
      ("think", (reasoning, answer))  exactly one `<think>` at the very start, one `</think>`, a reasoning and an answer that are not empty (what `reasoning.compose_trace` writes)
      ("malformed", why)              any other text that has a think tag in it: a block that is not closed, two blocks, a tag in the middle of the answer, no reasoning, no answer after it.
    `reasoning` is the template's trimmed `reasoning_content`; `answer` is its `content`. A malformed record is dropped, never trained as plain: that would teach a literal `<think>`."""
    if THINK_OPEN not in text and THINK_CLOSE not in text:
        return "plain", None
    body = text.strip()
    if body.count(THINK_OPEN) != 1 or body.count(THINK_CLOSE) != 1 or not body.startswith(THINK_OPEN):
        return "malformed", "needs exactly one <think> at the start and one </think>"
    head, _, tail = body.partition(THINK_CLOSE)
    reasoning, answer = head[len(THINK_OPEN):].strip(), tail.lstrip("\n")
    if not reasoning:
        return "malformed", "the think block is empty"
    if not answer.strip():
        return "malformed", "nothing follows the think block"
    return "think", (reasoning, answer)


def is_thinking_config(texts) -> bool:
    """True when at least one record of the dataset is a thinking record: the dataset is a thinking config (a `--thinking` sample run's file). Decided on the whole dataset, before a smoke run takes its slice."""
    return any(parse_assistant(t)[0] == "think" for t in texts)


def record_fate(text: str, thinking_config: bool = False, mix: bool = False) -> str:
    """One of FATES. A thinking record is trained as one; a malformed one is dropped; a plain one is trained as it always was, except in a thinking config without `mix`, where it is skipped
    (a thinking config trains the thinking records only unless the run asks for the mixture)."""
    kind, _ = parse_assistant(text)
    if kind != "plain":
        return "think" if kind == "think" else "malformed"
    return "skipped_plain" if thinking_config and not mix else "plain"


def render_record(tok, msgs: list, fate: str, check: bool = False) -> tuple:
    """(prompt, full) strings of one SFT record. `msgs` = system, user, assistant. The prompt is the base template's generation prompt (no loss); full = the prompt, the target and the EOS token.
      think  prompt with `enable_thinking=True` (ends in `<think>\\n`), target `{reasoning}\\n</think>\\n\\n{answer}`: prompt + target is the template's own rendering of the whole conversation.
      plain  prompt with `enable_thinking=False`, target the text as it is: phase A's path, byte for byte (in a thinking config this is the empty think block the template renders for an answer with no reasoning).
    `check`: raise ValueError when the prompt does not end the way the base template ends it (another tokenizer, another template): the labels would be placed wrongly."""
    if fate == "think":
        reasoning, answer = parse_assistant(msgs[-1]["content"])[1]
        prompt = tok.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True, enable_thinking=True)
        if check and not (prompt.endswith(THINK_GENERATION_PROMPT) and not prompt.endswith(EMPTY_THINK)):
            raise ValueError(f"a thinking record needs a prompt that ends in {THINK_GENERATION_PROMPT!r}; this tokenizer's template ends it in {prompt[-30:]!r}")
        return prompt, prompt + reasoning + "\n" + THINK_CLOSE + "\n\n" + answer + tok.eos_token
    prompt = tok.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    if check and not prompt.endswith(EMPTY_THINK):
        raise ValueError(f"a plain record of a thinking config needs a prompt that ends in {EMPTY_THINK!r}; this tokenizer's template ends it in {prompt[-30:]!r}")
    return prompt, prompt + msgs[-1]["content"] + tok.eos_token


def encode_record(tok, msgs: list, max_len: int = MAX_LEN, thinking_config: bool = False, mix: bool = False) -> tuple:
    """(training example or None, fate) of one SFT record: `render_record`, the token ids of the prompt and of the whole text, `encode_ids` (only the target is trained; a pair over `max_len` is
    dropped, never cut). None for a skipped or malformed record. In a thinking config the prompt's tokens must be the start of the whole text's (ValueError otherwise): they are, because the prompt ends
    in a newline that no token continues, and the check says so loudly if a tokenizer disagrees."""
    fate = record_fate(msgs[-1]["content"], thinking_config, mix)
    if fate in ("skipped_plain", "malformed"):
        return None, fate
    prompt, full = render_record(tok, msgs, fate, check=thinking_config)
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    f_ids = tok(full, add_special_tokens=False)["input_ids"]
    if thinking_config and f_ids[:len(p_ids)] != p_ids:
        raise ValueError(f"the tokens of the prompt are not the start of the tokens of the whole record ({fate}): the loss mask would be misplaced")
    return encode_ids(p_ids, f_ids, max_len), fate


def fate_counts(pairs) -> dict:
    """{"think", "plain", "skipped_plain", "malformed", "too_long"}: how many records were trained as each (`pairs` = (fate, kept) per record), the number dropped for length, and the thinking /
    plain records among those."""
    out = {"think": 0, "plain": 0, "skipped_plain": 0, "malformed": 0, "too_long": 0, "too_long_think": 0}
    for fate, kept in pairs:
        if fate in ("think", "plain") and not kept:
            out["too_long"] += 1
            out["too_long_think"] += fate == "think"
        else:
            out[fate] += 1
    return out
