"""LoRA fine-tune of the local-forge 9B (uncensored) on the verified REA SFT set, then merge, convert to GGUF and quantize.

    modal run tools/modal_lora.py --smoke            # 60 examples, 1 epoch, proves the pipeline end to end
    modal run tools/modal_lora.py                    # full run
    modal run tools/modal_lora.py --max-len 3072     # a longer training window (a reasoning run's records): a pair over it is dropped, never cut
Outputs land in the `lf-lora` volume: adapter/, merged/, gguf/{base,lora}-Q4_K_M.gguf, and are copied to results/phase6/lora/.
Base and LoRA GGUFs are produced by the SAME convert+quantize path (Q4_K_M, no imatrix), so the before/after on the
held-out set compares like with like; the host's i1-IQ4_XS is a different quant and is not the comparison point.
"""
from __future__ import annotations

import json
import pathlib
import subprocess
import time

import modal

_HERE = pathlib.Path(__file__).resolve()
ROOT = _HERE.parents[2] if len(_HERE.parents) > 2 else _HERE.parent  # inside the Modal container the file sits at /root
BASE = "xtools-at/DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic"
DATASET = "HiddenTerps/local-forge-agent-sft"
LLAMA_FULL_DIGEST = "sha256:553c5561e59409ce02701ea05d10a25ed08124bc14d1273a9fe6802fdb20d531"   # ghcr.io/ggml-org/llama.cpp:full-cuda12

train_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.14.1", index_url="https://download.pytorch.org/whl/cu126")
    .pip_install("transformers==5.22.2", "peft==0.21.2", "datasets>=4.4", "accelerate>=1.12", "huggingface_hub>=1.31,<3", "sentencepiece", "protobuf")
    .env({"TOKENIZERS_PARALLELISM": "false"})
)
gguf_image = (
    modal.Image.from_registry(f"ghcr.io/ggml-org/llama.cpp@{LLAMA_FULL_DIGEST}", add_python="3.11")
    .entrypoint([])
    .pip_install("torch==2.14.1", index_url="https://download.pytorch.org/whl/cpu")
    .pip_install("huggingface_hub>=1.31,<3", "numpy", "sentencepiece", "safetensors", "transformers==5.22.2", "gguf", "protobuf")

)
vol = modal.Volume.from_name("lf-lora", create_if_missing=True)
hf_secret = modal.Secret.from_name("hf-write")
app = modal.App("lf-lora")

MAX_LEN = 2048          # tokens of prompt + answer + EOS that one training example may have (the default; `--max-len` trains on a longer window: a reasoning run's records are several times longer)


def batch_plan(max_len: int) -> tuple:
    """(per-device batch size, gradient accumulation steps) for a training window. 4 x 4 at the default 2048 tokens, as before; a longer window keeps the tokens per micro-batch
    at 4 x 2048 (the memory of the A100-40GB run that is known to fit) and the batch of 16 examples per optimizer step: 2 x 8 at 3072 and 4096. NOT measured on a GPU for a window above 2048."""
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


@app.function(image=train_image, gpu="A100-40GB", volumes={"/out": vol}, secrets=[hf_secret], timeout=4 * 3600)
def train(smoke: bool = False, epochs: float = 2.0, lr: float = 1.5e-4, rank: int = 32, max_len: int = MAX_LEN) -> dict:
    import torch
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForSeq2Seq
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(BASE)
    ds = load_dataset(DATASET, "rea_sft", split="train")
    if smoke:
        ds = ds.select(range(60))
    ds = ds.shuffle(seed=7)

    def encode(ex):
        msgs = ex["messages"]
        prompt = tok.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        full = prompt + msgs[-1]["content"] + tok.eos_token
        p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        f_ids = tok(full, add_special_tokens=False)["input_ids"]
        row = encode_ids(p_ids, f_ids, max_len)
        return {**(row or {"input_ids": [], "attention_mask": [], "labels": []}), "keep": row is not None}

    n_in = len(ds)
    ds = ds.map(encode, remove_columns=ds.column_names).filter(lambda ex: ex["keep"]).remove_columns(["keep"])
    dropped = n_in - len(ds)
    print(f"{n_in} SFT pairs, {dropped} dropped because prompt + answer is longer than {max_len} tokens (none is cut), {len(ds)} trained on")
    if not len(ds):
        raise SystemExit(f"every one of the {n_in} SFT pairs is longer than {max_len} tokens: nothing to train on")
    model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16, device_map={"": 0})
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
    tag = "smoke" if smoke else "full"
    model.save_pretrained(f"/out/{tag}/adapter"); tok.save_pretrained(f"/out/{tag}/adapter")
    merged = model.merge_and_unload()
    merged.save_pretrained(f"/out/{tag}/merged", safe_serialization=True); tok.save_pretrained(f"/out/{tag}/merged")
    vol.commit()
    log = [x for x in trainer.state.log_history if "loss" in x]
    return {"tag": tag, "n": len(ds), "dropped_too_long": dropped, "max_len": max_len, "train_loss_first": log[0]["loss"] if log else None, "train_loss_last": log[-1]["loss"] if log else None,
            "steps": out.global_step, "wall_s": round(time.time() - t0), "trainable": str(cfg.target_modules)}


@app.function(image=gguf_image, cpu=8, memory=65536, volumes={"/out": vol}, secrets=[hf_secret], timeout=3 * 3600)
def to_gguf(tag: str, which: str) -> dict:
    """which = 'lora' (merged weights from /out/<tag>/merged) or 'base' (download BASE). Both -> F16 GGUF -> Q4_K_M."""
    import shutil
    from huggingface_hub import snapshot_download
    t0 = time.time()
    if which == "lora":
        src = f"/out/{tag}/merged"
    else:
        # The merged LoRA model is saved by transformers, which does not load the MTP head, so it has no
        # model-mtp tensors. Drop them from the base too (index included), so both GGUFs carry the same tensors.
        snap = snapshot_download(BASE, allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.txt"], ignore_patterns=["model-mtp.safetensors"])
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
def main(smoke: bool = False, epochs: float = 2.0, lr: float = 1.5e-4, rank: int = 32, skip_train: bool = False, max_len: int = MAX_LEN):
    dest = ROOT / "results" / "phase6" / "lora"; dest.mkdir(parents=True, exist_ok=True)
    tag = "smoke" if smoke else "full"
    if not skip_train:
        r = train.remote(smoke, epochs, lr, rank, max_len)
        print("train:", r); (dest / f"{tag}-train.json").write_text(json.dumps(r, indent=1))
    g = list(to_gguf.map([tag, tag], ["lora", "base"]))
    print("gguf:", g); (dest / f"{tag}-gguf.json").write_text(json.dumps(g, indent=1))
