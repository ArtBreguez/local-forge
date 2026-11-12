r"""LoRA fine-tune of a student model on a verified SFT set (any subject), then merge, convert to GGUF and quantize. Moved from `tools/modal/modal_lora.py` (now a wrapper of this file); the student is a
parameter (`--student`, default: the local-forge 9B uncensored that REA's LoRA is trained on) and nothing else about the run changed.

    modal run tools/modal/modal_lora.py --smoke            # 60 examples, 1 epoch, proves the pipeline end to end
    modal run tools/modal/modal_lora.py                    # full run
    modal run tools/modal/modal_lora.py --max-len 3072     # a longer training window (a reasoning run's records): a pair over it is dropped, never cut
    modal run tools/modal/modal_lora.py --config rea_reasoning --max-len 3072    # another dataset config (a reasoning file): its own output path /out/rea_reasoning, never /out/full
    modal run tools/modal/modal_lora.py --config rea_reasoning --tag reasoning-v1 --max-len 3072    # ... or a name of your own: /out/reasoning-v1
    modal run tools/modal/modal_lora.py --config rea_thinking --max-len 8192            # a thinking file (`<think>...</think>` in the assistant text): the thinking records only
    modal run tools/modal/modal_lora.py --config rea_thinking --max-len 8192 --mix      # ... plus its truth-template records, rendered with an empty think block: /out/rea_thinking-mix
    modal run tools/modal/modal_lora.py --student Qwen/Qwen3-8B --config arith_reasoning --tag arith-8b    # another student on another subject's file (its tokenizer's chat template must render `<think>` like Qwen3's)
    modal run tools/modal/modal_lora.py --run-config distill/configs/rea-thinking.json --config rea_thinking   # student, max_len and mix from a run config instead of their flags
Outputs land in the `lf-lora` volume: adapter/, merged/, gguf/{base,lora}-Q4_K_M.gguf, and are copied to results/phase6/lora/. The directory is `/out/<tag>/` and the files
are `<tag>-train.json`, `<tag>-gguf.json` with the tag of `run_tag`: "full" (or "smoke") for the default config `rea_sft` with no `--tag`, exactly as before, and for any other
`--config` or `--tag` a name of its own that is never "full" or "smoke": phase A's output is never overwritten by a run on another file.
Thinking records (round 13). The base model's chat template (Qwen3.5 family, saved from HiddenTerps/LocalForge-9B-REA-heretic) renders an assistant turn that follows the last user message as
`<|im_start|>assistant\n<think>\n{reasoning}\n</think>\n\n{answer}<|im_end|>`, and the generation prompt as `<think>\n` (`enable_thinking=True`, or not given) or `<think>\n\n</think>\n\n`
(`enable_thinking=False`). A record whose assistant text is `<think>\nreasoning\n</think>\n\nanswer` (what a `--thinking` sample run stores) is therefore trained as: prompt = the template's generation
prompt with `enable_thinking=True` (ends in `<think>\n`, no loss), target = `{reasoning}\n</think>\n\n{answer}` + EOS (loss on the reasoning and the answer). Prompt + target is the token sequence of
the template's own rendering of the whole conversation (tests/test_rea_thinking.py renders the real template). The decision is per RECORD, from its text (`parse_assistant`): no think tag = a plain record,
one complete block followed by an answer = a thinking record, anything else with a tag in it = malformed and dropped (counted). A plain record keeps today's path exactly (prompt with
`enable_thinking=False`, text + EOS). A DATASET is a thinking config when at least one record is a thinking record: it then trains the thinking records ONLY (its plain records, the truth-template ones,
are skipped and counted); `--mix` keeps them, each rendered with the empty think block of `enable_thinking=False`, which is also how the template renders an assistant turn with no reasoning. That teaches
the model to answer directly, with no reasoning, when its prompt says so; it is a choice, so it is a flag, and a mixed run has its own output (`<config>-mix`). A config with no thinking record
(`rea_sft`, a visible-reasoning file) is unchanged whatever `--mix` says.
Base and LoRA GGUFs are produced by the SAME convert+quantize path (Q4_K_M, no imatrix), so the before/after on the
held-out set compares like with like; the host's i1-IQ4_XS is a different quant and is not the comparison point.
"""
from __future__ import annotations

import json
import pathlib
import re
import subprocess
import time

import modal

from .. import config as run_config_module
from ..config import DEFAULT_STUDENT, MAX_LEN

_HERE = pathlib.Path(__file__).resolve()
ROOT = _HERE.parents[2]      # distill/backends/modal_train.py: the repository root (in the Modal container: /root, where distill is mounted)
BASE = DEFAULT_STUDENT       # the default student: the model REA's LoRA is trained on (`--student` trains another)
DATASET = "HiddenTerps/local-forge-agent-sft"
LLAMA_FULL_DIGEST = "sha256:553c5561e59409ce02701ea05d10a25ed08124bc14d1273a9fe6802fdb20d531"   # ghcr.io/ggml-org/llama.cpp:full-cuda12

train_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.14.1", index_url="https://download.pytorch.org/whl/cu126")
    .pip_install("transformers==5.22.2", "peft==0.21.2", "datasets>=4.4", "accelerate>=1.12", "huggingface_hub>=1.31,<3", "sentencepiece", "protobuf")
    .env({"TOKENIZERS_PARALLELISM": "false"})
    .add_local_python_source("distill")      # this module is `distill.backends.modal_train`: the container imports it by that name
)
gguf_image = (
    modal.Image.from_registry(f"ghcr.io/ggml-org/llama.cpp@{LLAMA_FULL_DIGEST}", add_python="3.11")
    .entrypoint([])
    .pip_install("torch==2.14.1", index_url="https://download.pytorch.org/whl/cpu")
    .pip_install("huggingface_hub>=1.31,<3", "numpy", "sentencepiece", "safetensors", "transformers==5.22.2", "gguf", "protobuf")
    .add_local_python_source("distill")
)
vol = modal.Volume.from_name("lf-lora", create_if_missing=True)
hf_secret = modal.Secret.from_name("hf-write")
app = modal.App("lf-lora")

DEFAULT_CONFIG = "rea_sft"       # the dataset config phase A trained on; its outputs are /out/full and /out/smoke
RESERVED_TAGS = ("full", "smoke")        # the output names of the default run: no other config or `--tag` may write there


def run_tag(smoke: bool = False, config: str = DEFAULT_CONFIG, tag: str = "", mix: bool = False) -> str:
    """The name of a run's output: its directory `/out/<tag>/` on the volume and its files `<tag>-train.json`, `<tag>-gguf.json`. The default (config `rea_sft`, no tag) is "full", or "smoke" with
    `--smoke`: what it always was. A run on another config is named after it (`rea_reasoning`), and `--tag` names it by hand; either way with `-smoke` after it for a smoke run. "full" and "smoke" are
    reserved for the default run (SystemExit): training on a reasoning file must never overwrite phase A's `/out/full`. `mix` (a thinking file plus its truth-template records) names a derived
    output `<config>-mix`, so a mixed run and a thinking-only run on the same file never share a directory; the default config has no such run (SystemExit: it would write `/out/full`)."""
    if tag:
        name = tag
    elif config == DEFAULT_CONFIG:
        if mix:
            raise SystemExit(f"--mix mixes the truth-template records of a THINKING file into its thinking records; the default config {DEFAULT_CONFIG!r} has none and would write /out/full: pass --config and --tag")
        return "smoke" if smoke else "full"
    else:
        name = re.sub(r"[^A-Za-z0-9._-]+", "-", config).strip("-.") + ("-mix" if mix else "")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name or ""):
        raise SystemExit(f"output tag {name!r} (from {'--tag' if tag else '--config'}) is not a plain name: letters, digits, `.`, `_`, `-`; pass --tag")
    if name in RESERVED_TAGS:
        raise SystemExit(f"output tag {name!r} is reserved for the default run on the config {DEFAULT_CONFIG!r} (/out/full, /out/smoke): pass --tag with another name")
    return f"{name}-smoke" if smoke else name


# MAX_LEN (tokens of prompt + answer + EOS that one training example may have; `--max-len` trains on a longer window: a reasoning run's records are several times longer) is `distill.config.MAX_LEN`.


def batch_plan(max_len: int) -> tuple:
    """(per-device batch size, gradient accumulation steps) for a training window. 4 x 4 at the default 2048 tokens, as before; a longer window up to 8192 keeps the tokens per micro-batch
    at 4 x 2048 (the memory of the A100-40GB run that is known to fit) and the batch of 16 examples per optimizer step: 2 x 8 at 3072 and 4096, 1 x 16 at 8192 (one sequence of 8192 tokens per micro-batch,
    the same 8192 tokens as 4 x 2048, so the token-proportional memory (logits, activations with gradient checkpointing) is the same by construction; the attention part is not, and the first run is a
    `--smoke --max-len 8192`). ABOVE 8192 that no longer holds: a micro-batch cannot be smaller than one sequence, so the plan stays 1 x 16 and the tokens per micro-batch GROW with the window: at 12288, the
    window of the shipped thinking config, one micro-batch is 12288 tokens, 1.5 times the 4 x 2048 that is known to fit, so even the token-proportional memory is not the same, and nothing says it fits
    (tools/modal/README.md says so; start with `--smoke`, and `--max-len 8192` is the fallback). NOT measured on a GPU for a window above 2048."""
    per_device = min(4, max(1, 4 * 2048 // max_len))
    return per_device, max(1, 16 // per_device)


def encode_ids(p_ids: list, f_ids: list, max_len: int = MAX_LEN):
    """The training example for one SFT pair, or None when it must not be trained on. `p_ids`: the prompt (system + user + generation prompt), `f_ids`: the prompt,
    the answer and the EOS token. Only the answer is trained (labels -100 on the prompt). A pair longer than `max_len` is DROPPED, never cut: a cut pair has no
    EOS, so the model would be taught an answer that does not end, and a prompt that alone fills the window leaves nothing to learn from (round 8, m12: the
    encoder used to slice `f_ids[:2048]` without a word, and the Ghidra pseudocode of a task is not bounded by anything in the generator)."""
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


@app.function(image=train_image, gpu="A100-40GB", volumes={"/out": vol}, secrets=[hf_secret], timeout=4 * 3600)
def train(smoke: bool = False, epochs: float = 2.0, lr: float = 1.5e-4, rank: int = 32, max_len: int = MAX_LEN, config: str = DEFAULT_CONFIG, out_tag: str = "", mix: bool = False, student: str = BASE) -> dict:
    import torch
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForSeq2Seq
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(student)
    tag = out_tag or run_tag(smoke, config)          # /out/<tag>/: the default config is "full" / "smoke" as before, any other config its own name
    if mix and not out_tag:
        tag = run_tag(smoke, config, "", True)       # a mixed run is never written where the thinking-only run of the same file is
    ds = load_dataset(DATASET, config, split="train")
    thinking_config = is_thinking_config(ex[-1]["content"] for ex in ds["messages"])         # of the whole file, before a smoke run takes its slice
    if smoke:
        ds = ds.select(range(60))
    ds = ds.shuffle(seed=7)

    def encode(ex):
        row, fate = encode_record(tok, ex["messages"], max_len, thinking_config, mix)
        return {**(row or {"input_ids": [], "attention_mask": [], "labels": []}), "keep": row is not None, "fate": fate}

    n_in = len(ds)
    ds = ds.map(encode, remove_columns=ds.column_names)
    counts = fate_counts(zip(ds["fate"], ds["keep"]))
    ds = ds.filter(lambda ex: ex["keep"]).remove_columns(["keep", "fate"])
    dropped = counts["too_long"]
    print(f"{n_in} SFT pairs, {dropped} dropped because prompt + answer is longer than {max_len} tokens (none is cut), {len(ds)} trained on")
    if thinking_config or counts["malformed"]:
        print(f"thinking config: {counts['think']} thinking records trained (prompt ends in <think>, loss on reasoning + answer), {counts['plain']} plain records trained with an empty think block "
              f"({'--mix' if mix else 'no --mix'}), {counts['skipped_plain']} plain records skipped, {counts['malformed']} malformed dropped, {counts['too_long_think']} thinking records over {max_len} tokens")
    if not len(ds):
        if not (counts["skipped_plain"] or counts["malformed"]):
            raise SystemExit(f"every one of the {n_in} SFT pairs is longer than {max_len} tokens: nothing to train on")
        raise SystemExit(f"nothing to train on: {n_in} SFT pairs, {dropped} longer than {max_len} tokens, {counts['skipped_plain']} skipped plain, {counts['malformed']} malformed")
    model = AutoModelForCausalLM.from_pretrained(student, torch_dtype=torch.bfloat16, device_map={"": 0})
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    cfg = LoraConfig(r=rank, lora_alpha=2 * rank, lora_dropout=0.05, task_type="CAUSAL_LM",
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "in_proj_qkvz", "in_proj_ba", "out_proj"])
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()
    per_device, accum = batch_plan(max_len)
    args = TrainingArguments(output_dir="/tmp/run", per_device_train_batch_size=per_device, gradient_accumulation_steps=accum, num_train_epochs=0.2 if smoke else epochs,
                             learning_rate=lr, lr_scheduler_type="cosine", warmup_steps=0.05, logging_steps=10, save_strategy="no",
                             bf16=True, report_to=[], dataloader_num_workers=2, seed=7)
    trainer = Trainer(model=model, args=args, train_dataset=ds, data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100))
    out = trainer.train()
    model.save_pretrained(f"/out/{tag}/adapter"); tok.save_pretrained(f"/out/{tag}/adapter")
    merged = model.merge_and_unload()
    merged.save_pretrained(f"/out/{tag}/merged", safe_serialization=True); tok.save_pretrained(f"/out/{tag}/merged")
    vol.commit()
    log = [x for x in trainer.state.log_history if "loss" in x]
    return {"tag": tag, "config": config, "student": student, "n": len(ds), "dropped_too_long": dropped, "thinking_config": thinking_config, "mix": mix, "records": counts, "max_len": max_len, "train_loss_first": log[0]["loss"] if log else None, "train_loss_last": log[-1]["loss"] if log else None,
            "steps": out.global_step, "wall_s": round(time.time() - t0), "trainable": str(cfg.target_modules)}


@app.function(image=gguf_image, cpu=8, memory=65536, volumes={"/out": vol}, secrets=[hf_secret], timeout=3 * 3600)
def to_gguf(tag: str, which: str, student: str = BASE) -> dict:
    """which = 'lora' (merged weights from /out/<tag>/merged) or 'base' (download the student). Both -> F16 GGUF -> Q4_K_M."""
    import shutil
    from huggingface_hub import snapshot_download
    t0 = time.time()
    if which == "lora":
        src = f"/out/{tag}/merged"
    else:
        # The merged LoRA model is saved by transformers, which does not load the MTP head, so it has no
        # model-mtp tensors. Drop them from the base too (index included), so both GGUFs carry the same tensors.
        snap = snapshot_download(student, allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.txt"], ignore_patterns=["model-mtp.safetensors"])
        src = "/tmp/base"
        shutil.rmtree(src, ignore_errors=True)
        shutil.copytree(snap, src, symlinks=False)
        idx = pathlib.Path(src, "model.safetensors.index.json")
        j = json.loads(idx.read_text())
        j["weight_map"] = {k: v for k, v in j["weight_map"].items() if v != "model-mtp.safetensors"}
        idx.write_text(json.dumps(j))
    outdir = pathlib.Path(f"/out/{tag}/gguf"); outdir.mkdir(parents=True, exist_ok=True)
    f16 = outdir / f"{which}-F16.gguf"; q4 = outdir / f"{which}-Q4_K_M.gguf"
    conv = subprocess.run(["python", "/app/convert_hf_to_gguf.py", src, "--outtype", "f16", "--outfile", str(f16)], capture_output=True, text=True)
    if conv.returncode != 0:
        return {"error": "convert failed", "tail": conv.stderr[-3000:]}
    qz = subprocess.run(["/app/llama-quantize", str(f16), str(q4), "Q4_K_M"], capture_output=True, text=True)
    if qz.returncode != 0:
        return {"error": "quantize failed", "tail": qz.stderr[-3000:]}
    f16.unlink()
    vol.commit()
    return {"which": which, "gguf": str(q4), "size_gb": round(q4.stat().st_size / 1e9, 2), "wall_s": round(time.time() - t0)}


@app.local_entrypoint()
def main(smoke: bool = False, epochs: float = 2.0, lr: float = 1.5e-4, rank: int = 32, skip_train: bool = False, max_len: int = MAX_LEN, config: str = DEFAULT_CONFIG, tag: str = "", mix: bool = False,
         student: str = BASE, run_config: str = ""):
    """config: the dataset config to train on (default `rea_sft`, phase A's file). tag: the output name; see `run_tag` (a non-default config or a tag never writes to /out/full). max_len: the training window
    (8192 for thinking records). mix: a thinking config (records with `<think>`) also trains its truth-template records, rendered with an empty think block, so the model also answers directly when its prompt
    asks for no reasoning; without it a thinking config trains the thinking records only. The output is `<config>-mix`. No effect on a file with no thinking record.
    student: the Hugging Face model to fine-tune (default: the local-forge 9B uncensored); its tokenizer's chat template must render a `<think>` block like the Qwen3 family's for a thinking file.
    run_config: a JSON file of `distill.config.RunConfig` that sets student, max_len and mix instead of their flags."""
    if run_config:
        f = run_config_module.merge_train_flags(run_config, dict(student=student, max_len=max_len, mix=mix))
        student, max_len, mix = f["student"], f["max_len"], f["mix"]
    tag = run_tag(smoke, config, tag, mix)           # checked before anything is rented
    dest = ROOT / "results" / "phase6" / "lora"; dest.mkdir(parents=True, exist_ok=True)
    if not skip_train:
        if student == BASE:      # a default run sends the call it always sent
            r = train.remote(smoke, epochs, lr, rank, max_len, config, tag, mix)
        else:
            r = train.remote(smoke, epochs, lr, rank, max_len, config, tag, mix, student)
        print("train:", r); (dest / f"{tag}-train.json").write_text(json.dumps(r, indent=1))
    g = list(to_gguf.map([tag, tag], ["lora", "base"]) if student == BASE else to_gguf.map([tag, tag], ["lora", "base"], [student, student]))
    print("gguf:", g); (dest / f"{tag}-gguf.json").write_text(json.dumps(g, indent=1))
