"""Kaggle GPU kernel: QLoRA-train the student on the SFT records of the dataset, then answer the held-out tasks twice, with the base student and with base + adapter, so that
`python3 -m distill.backends.kaggle.score_lora` (on your machine, where the truth is) can say whether the adapter helps. `build_lora.py` writes the dataset and the kernel; the README of this folder has the
commands.

It is the file of the dataset's `lib`, NOT pushed as the kernel's `run.py` (which is a launcher that imports it: a script kernel ships one file, and the workers below need to start this one by path). Next to it in
the dataset are `run_teacher.py` (reused: the checkpoint format, resume and deadline rules of the teacher kernel) and `distill.<module>.py` (the stdlib-only core: `config`, `traces`, `domain`, `sft`, rebuilt by
`unpack_lib` as the package `distill`, because the Kaggle CLI uploads the files of a folder and skips its sub-folders). The record -> training example code is `distill/sft.py`, the code of the Modal trainer.

What it does, in order (`main`, no GPU library imported in this process):
  1. reads `lora-config.json`, installs the pins (the versions `modal_train.py` trains with), downloads the student once (a public repo, bf16, ~18 GB, to /tmp: never to /kaggle/working);
  2. TRAIN (`--worker train`, `torchrun` over both T4: one process per GPU, the 4-bit student in each, gradients averaged by hand): fp16 (a T4 has no bf16), NF4, gradient checkpointing, batch 1 per
     process, the loss computed in chunks over the target positions only (the student's vocabulary is 248,320 tokens: the logits of one 4096-token sequence are 4 GB in fp32, more than a T4 has next to the
     model), LoRA on the same modules as `modal_train.py`. Skipped when a finished adapter is already in /kaggle/working or in an attached output. It stops at `train_hours` or when the evaluation would no longer
     fit before the session deadline, saves the adapter and a `train.json`, and checkpoints every `save_every` steps (resume: the newest complete `ckpt/step-N` of this kernel or of an attached output);
  3. EVAL (`--worker eval`, the same two processes): the base student (`model.disable_adapter()`) and base + adapter answer the same held-out tasks with the thinking template and Qwen3's thinking sampling, `k`
     replies each, the tasks split between the processes; every batch is appended at once to `replies-base.r<rank>.jsonl` / `replies-lora.r<rank>.jsonl` in the line format of `run_teacher.py`, base and
     adapter for the same tasks next to each other, so a run stopped by the deadline holds paired answers. Resume asks only what is missing, failed or stale (`run_teacher.pending`);
  4. merges the shards into `replies-base.jsonl` and `replies-lora.jsonl`, optionally (`export_gguf`) merges the adapter into fp16 weights and writes a Q4_K_M GGUF, writes `run.json`.
`--smoke` does all of it on a few of the longest records and one batch of `eval_batch` (8) tasks per process, 256 new tokens (files are named `smoke-...`, so that it can never be taken for a result): it measures the
step time, the generation speed AT THE BATCH THE FULL RUN USES and the peak memory that the builder's estimate only assumes (`measured` in run.json, `build_lora.py --measured`).
Nothing is scored here: the truth is not in the dataset. HF `generate` seeds a batch, not a reply (seed = seed_base + the batch's first item): the same replies are not reproducible batch by batch.
"""
from __future__ import annotations

import functools
import json
import math
import os
import pathlib
import random
import shutil
import subprocess
import sys
import time

T0 = time.time()
HERE = pathlib.Path(__file__).resolve().parent
OUT = pathlib.Path("/kaggle/working")
INPUT = pathlib.Path("/kaggle/input")
LIB = pathlib.Path(os.environ.get("LORA_LIB", "/tmp/lf-lib"))          # where the flat `distill.<module>.py` files are unpacked as a package (the variable is for the tests)
STUDENT_DIR = pathlib.Path("/tmp/student")
ARMS = ("base", "lora")
LIB_MODULES = ("__init__", "config", "traces", "domain", "sft")          # the stdlib-only core this kernel runs; `build_lora.py` ships them as `distill.<module>.py`


def unpack_lib(src: pathlib.Path = HERE, dest: pathlib.Path = LIB) -> bool:
    """The flat files `distill.<module>.py` next to this file become the package `dest/distill/<module>.py`, and `dest` goes first on sys.path. False (nothing done) when there are none: in the repository
    `distill` is the package itself."""
    flat = sorted(pathlib.Path(src).glob("distill.*.py"))
    if not flat:
        return False
    pkg = pathlib.Path(dest) / "distill"
    pkg.mkdir(parents=True, exist_ok=True)
    for f in flat:
        target = pkg / f.name[len("distill."):]
        if not target.exists() or target.read_bytes() != f.read_bytes():
            shutil.copyfile(f, target)
    if str(dest) not in sys.path:
        sys.path.insert(0, str(dest))
    return True


unpack_lib()
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import run_teacher  # noqa: E402  (reused: log, sh, can_start, pending, load_previous, read_lines, prompt_sha, stored_text)
from distill import config, sft  # noqa: E402

DEFAULTS = {
    "student": config.DEFAULT_STUDENT,       # the base model of modal_train.py
    "max_len": 4096,                         # training window in tokens (a longer record is dropped, never cut); the Modal thinking config is 8192, which a T4 cannot hold
    "rank": 16,                              # modal_train.py: 32; halved for the T4's memory (the optimizer state of the adapter is fp32); alpha is 2 x rank as there
    "dropout": 0.05,
    "epochs": 2.0,
    "lr": 1.5e-4,                            # the Modal run's learning rate, cosine, 5% warmup
    "global_batch": 16,                      # sequences per optimizer step, the Modal run's: 8 per process on two GPUs, 1 at a time
    "mix": False,                            # a thinking file's truth-template records are NOT trained (modal_train's default); True: trained with an empty think block
    "seed": 7,
    "gpus": 2,
    "gpu_name": "T4",                        # the kernel stops before the download when a GPU is not this one ("" = any): `enable_gpu` does not choose between a P100 and the two T4
    "max_steps": 0,                          # cap on the optimizer steps (the builder sets it when `train_hours` binds), 0 = epochs decide
    "save_every": 10,                        # steps between checkpoints
    "loss_chunk": 512,                       # target positions per slice of the loss
    "max_grad_norm": 1.0,
    "k": 1,                                  # replies per task and arm
    "max_new_tokens": config.DEFAULT_THINKING_MAX_TOKENS,
    "eval_batch": 8,                         # replies generated together on one GPU
    "sampling": dict(config.THINKING_SAMPLING),
    "seed_base": 1234,
    "train_hours": 3.0,                      # training stops this long after it starts, whatever the steps say
    "deadline_h": 7.5,                       # hours after the start at which no new batch or step is started (the weekly quota, not the 12 h session, is what limits this)
    "eval_reserve_s": 0,                     # what training leaves for the evaluation (the builder's upper bound)
    "export_gguf": False,
    "export_reserve_s": 0,
    "fla": False,                            # also install flash-linear-attention (fast kernels for the student's linear-attention layers; not tried on a T4)
    "smoke": False,
    "smoke_steps": 4,
    "smoke_new_tokens": 256,                 # the smoke run's reply budget: one batch of `eval_batch` replies per process and arm, so that `gen_step_s` is measured at the batch the full run uses
    "pins": ["transformers==5.22.2", "peft==0.21.2", "accelerate>=1.12", "bitsandbytes>=0.46", "huggingface_hub>=1.31,<3", "hf_transfer", "sentencepiece", "protobuf"],      # modal_train.py's, plus bitsandbytes
}
EOS_TEXT = ("<|im_end|>", "<|endoftext|>")


def log(*a):
    run_teacher.log(*a)


# ---- configuration ----------------------------------------------------------------------------------------------------------------------------

def resolve_config(user: dict) -> dict:
    """DEFAULTS, then the dataset's lora-config.json. The sampling is a thinking run's and never greedy (`config.sampling_for`); sizes are positive."""
    cfg = {**DEFAULTS, **user}
    cfg["sampling"] = config.sampling_for(True, **{k: v for k, v in dict(cfg["sampling"]).items() if k in config.SAMPLING_KEYS})
    for key in ("max_len", "rank", "global_batch", "gpus", "k", "max_new_tokens", "eval_batch", "save_every", "loss_chunk"):
        if not isinstance(cfg[key], int) or isinstance(cfg[key], bool) or cfg[key] < 1:
            raise ValueError(f"{key} {cfg[key]!r}: an integer of at least 1")
    if cfg["epochs"] <= 0 or cfg["lr"] <= 0 or cfg["train_hours"] <= 0 or cfg["deadline_h"] <= 0:
        raise ValueError("epochs, lr, train_hours and deadline_h must be positive")
    return cfg


# ---- what training does with its steps (pure) -------------------------------------------------------------------------------------------------

def lr_at(step: int, total: int, base: float, warmup: float = 0.05) -> float:
    """The learning rate of optimizer step `step` (0-based) of `total`: linear warmup over the first `warmup` share, then cosine to 0 (the Modal run's `cosine` with `warmup_steps=0.05`)."""
    warm = max(1, math.ceil(total * warmup))
    if step < warm:
        return base * (step + 1) / warm
    progress = (step - warm) / max(1, total - warm)
    return base * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))


def total_steps(n: int, epochs: float, global_batch: int, cap: int = 0) -> int:
    """Optimizer steps of the run: `epochs` passes over `n` examples at `global_batch` per step, at most `cap` (0: no cap). At least 1."""
    full = max(1, math.ceil(epochs * n / global_batch))
    return min(full, cap) if cap else full


@functools.lru_cache(maxsize=8)
def epoch_perm(n: int, epoch: int, seed: int) -> tuple:
    p = list(range(n))
    random.Random(seed * 1000 + epoch).shuffle(p)
    return tuple(p)


def step_indices(n: int, global_batch: int, step: int, seed: int) -> list:
    """The examples of optimizer step `step`: a stream of shuffled epochs (one permutation per epoch, seeded), `global_batch` of it per step. The same list on every process and on a resume, so a resumed run
    continues where it stopped and no data-loader state has to be saved."""
    return [epoch_perm(n, i // n, seed)[i % n] for i in range(step * global_batch, (step + 1) * global_batch)]


def rank_slice(indices: list, rank: int, world: int) -> list:
    """The share of a step that process `rank` of `world` computes."""
    return indices[rank::world]


def chunk_bounds(n: int, chunk: int) -> list:
    """[(start, stop)] slices of `n` positions of at most `chunk`: the loss is computed one slice at a time."""
    return [(i, min(i + chunk, n)) for i in range(0, n, chunk)]


def train_deadline(now: float, session_deadline: float, train_hours: float, eval_reserve_s: float) -> float:
    """When training must stop: `train_hours` from `now`, and never so late that the evaluation (`eval_reserve_s`, its upper bound) no longer fits before the session deadline."""
    return min(now + train_hours * 3600, session_deadline - eval_reserve_s)


def smoke_examples(examples: list, steps: int, global_batch: int) -> list:
    """The longest `steps * global_batch` examples (all of them when fewer): a smoke run trains on the records that need the most memory."""
    return sorted(examples, key=lambda e: -len(e["input_ids"]))[:steps * global_batch]


def encode_records(tok, rows: list, max_len: int, mix: bool) -> tuple:
    """([training example], fate counts) of the SFT records, by `sft.encode_record` (the Modal trainer's code): a thinking file trains its thinking records, and its truth-template ones only with `mix`; a record
    over `max_len` is dropped, never cut. The examples are in the file's order; the shuffle is `step_indices`'."""
    thinking_config = sft.is_thinking_config(r["messages"][-1]["content"] for r in rows)
    examples, fates = [], []
    for r in rows:
        ex, fate = sft.encode_record(tok, r["messages"], max_len, thinking_config, mix)
        fates.append((fate, ex is not None))
        if ex is not None:
            examples.append(ex)
    return examples, sft.fate_counts(fates)


# ---- the checkpoints and the finished adapter (pure: files only) ------------------------------------------------------------------------------

def find_adapter(roots) -> "pathlib.Path | None":
    """The finished adapter of an earlier session: a folder `adapter/` with `adapter_model.safetensors` and a `train.json` that says `done` or `partial`, under one of `roots` (this kernel's output first, then
    attached outputs). Training is skipped when there is one. A smoke run's is `adapter-smoke` and never matches."""
    for root in roots:
        root = pathlib.Path(root)
        for marker in sorted(root.rglob("train.json")) if root.exists() else []:
            try:
                status = json.loads(marker.read_text()).get("status")
            except (OSError, json.JSONDecodeError):
                continue
            if marker.parent.name == "adapter" and (marker.parent / "adapter_model.safetensors").exists() and status in ("done", "partial"):
                return marker.parent
    return None


def latest_checkpoint(roots) -> "pathlib.Path | None":
    """The newest COMPLETE checkpoint `ckpt/step-<n>/` (it has the `DONE` file, written last) under one of `roots`, None when there is none. A checkpoint a killed session left half written has no DONE."""
    best, best_step = None, -1
    for root in roots:
        root = pathlib.Path(root)
        for marker in root.rglob("DONE") if root.exists() else []:
            d = marker.parent
            if d.parent.name == "ckpt" and d.name.startswith("step-") and d.name[5:].isdigit() and int(d.name[5:]) > best_step:
                best, best_step = d, int(d.name[5:])
    return best


# ---- what evaluation does (pure) --------------------------------------------------------------------------------------------------------------

def shard_path(out: pathlib.Path, arm: str, rank: int, prefix: str = "") -> pathlib.Path:
    return pathlib.Path(out) / f"{prefix}replies-{arm}.r{rank}.jsonl"


def previous_files(arm: str, out: pathlib.Path = OUT, input_root: pathlib.Path = INPUT, prefix: str = "") -> list:
    """The replies of `arm` that earlier sessions wrote (an attached output's `replies-<arm>.jsonl` and its shards), then this kernel's own `replies-<arm>.jsonl` and shards: what a resume starts from."""
    names = f"{prefix}replies-{arm}*.jsonl"
    attached = sorted(pathlib.Path(input_root).rglob(names)) if pathlib.Path(input_root).exists() else []
    return [*attached, *sorted(pathlib.Path(out).glob(names))]


def eval_cfg(cfg: dict) -> dict:
    """What `run_teacher.pending` reads: `k` replies per task, the budget `max_new_tokens`, no per-family budget, no trust of unfingerprinted replies (this kernel always writes the fingerprint)."""
    return {"k": cfg["k"], "max_tokens": cfg["max_new_tokens"], "max_tokens_by_family": {}, "trust_unfingerprinted": False}


def work_items(tasks: list, previous: dict, cfg: dict) -> list:
    """[(task, sample index)] the evaluation still has to ask: for each task, every sample index that is missing, failed or stale in EITHER arm (`previous` = {arm: {id: line}}), so that the two arms stay
    on the same samples. Sorted by prompt length (the batches pad less), longest first, ties by id."""
    ec, items = eval_cfg(cfg), []
    for t in tasks:
        todo = sorted({i for arm in ARMS for i in run_teacher.pending(ec, t, previous.get(arm, {}).get(t["id"]))})
        items += [(t, i) for i in todo]
    return sorted(items, key=lambda it: (-sum(len(m["content"]) for m in it[0]["messages"]), it[0]["id"], it[1]))


def assign(items: list, rank: int, world: int) -> list:
    """The share of `work_items` that process `rank` of `world` asks: whole TASKS, round robin in the order the items come, so that every sample of a task is in one process's shard (the merge keeps the
    last line of an id: a task split over two shards would lose a process's samples)."""
    order = {}
    for t, _ in items:
        order.setdefault(t["id"], len(order))
    return [it for it in items if order[it[0]["id"]] % world == rank]


def split_generation(text: str) -> tuple:
    """(reasoning, content) of the text a model wrote after a generation prompt that ends in `<think>\\n`: what is in front of `</think>` is the reasoning, what follows is the reply. No `</think>`: all of
    it is reasoning (the model never closed the block: a cut-off reply) and the reply is empty. A leading `<think>` the model repeated is dropped."""
    for e in EOS_TEXT:
        text = text.replace(e, "")
    text = text.lstrip()
    if text.startswith("<think>"):
        text = text[len("<think>"):]
    reasoning, sep, content = text.partition("</think>")
    return (reasoning.strip(), content.strip()) if sep else (text.strip(), "")


def finish_and_tokens(ids: list, eos_ids) -> tuple:
    """(finish_reason, completion tokens, ids before the end of turn) of one generated row: "stop" at the first end-of-turn token (not counted in the text, counted in the tokens), else "length" with every
    token the row has."""
    for i, t in enumerate(ids):
        if t in eos_ids:
            return "stop", i + 1, ids[:i]
    return "length", len(ids), list(ids)


def reply_line(task: dict, sha: str, outs: list) -> dict:
    """A line of the replies file: the format of `run_teacher.py`."""
    return {"id": task["id"], "kind": task["kind"], "prompt_sha": sha, "outs": outs}


def merge_shards(out: pathlib.Path, arm: str, prefix: str = "", input_root: pathlib.Path = INPUT) -> int:
    """`<prefix>replies-<arm>.jsonl` from the replies of earlier sessions and this one's shards (the last line of an id wins: `run_teacher.load_previous`), written whole through a `.part` file.
    Returns the number of tasks."""
    out = pathlib.Path(out)
    merged = run_teacher.load_previous(previous_files(arm, out, input_root, prefix))
    final = out / f"{prefix}replies-{arm}.jsonl"
    part = final.with_suffix(".part")
    part.write_text("".join(json.dumps(rec) + "\n" for rec in merged.values()))
    os.replace(part, final)
    for shard in out.glob(f"{prefix}replies-{arm}.r*.jsonl"):
        shard.unlink()
    return len(merged)


# ---- the GPU workers (torch is imported here and nowhere above) --------------------------------------------------------------------------------

def dist_setup(torch) -> tuple:
    rank, world, local = (int(os.environ.get(k, d)) for k, d in (("RANK", 0), ("WORLD_SIZE", 1), ("LOCAL_RANK", 0)))
    torch.cuda.set_device(local)
    if world > 1:
        torch.distributed.init_process_group("nccl")
    return rank, world, local


def load_student(torch, cfg: dict, local: int, adapter=None, train: bool = False):
    """The student in 4 bits (NF4, fp16 compute: a T4 has no bf16) on GPU `local`, with LoRA: a fresh one (training), or the saved adapter at `adapter` (a resume, or the evaluation)."""
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(str(STUDENT_DIR), quantization_config=bnb, torch_dtype=torch.float16, device_map={"": local})
    model.config.use_cache = False
    if train:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    if adapter is not None:
        model = PeftModel.from_pretrained(model, str(adapter), is_trainable=train)
    elif train:
        model = get_peft_model(model, LoraConfig(r=cfg["rank"], lora_alpha=2 * cfg["rank"], lora_dropout=cfg["dropout"], task_type="CAUSAL_LM", target_modules=sft.LORA_TARGETS))
    torch.cuda.empty_cache()
    return model


def chunked_loss(torch, hidden, labels, lm_head, chunk: int):
    """Mean cross-entropy of the next token over the target positions of one sequence (`hidden` [1, T, H] after the final norm, `labels` [1, T], -100 on the prompt), the logits made `chunk` positions at a
    time inside `torch.utils.checkpoint`, so that no slice of them is kept for the backward pass: the logits of the whole sequence (T x 248,320 in fp32) never exist. Returns (loss, target tokens)."""
    h, y = hidden[0, :-1], labels[0, 1:]
    keep = y != -100
    h, y = h[keep], y[keep]
    n = int(y.numel())
    part = lambda hh, yy: torch.nn.functional.cross_entropy(lm_head(hh).float(), yy, reduction="sum")
    total = sum(torch.utils.checkpoint.checkpoint(part, h[a:b], y[a:b], use_reentrant=False) for a, b in chunk_bounds(n, chunk))
    return total / max(n, 1), n


def all_reduce_mean(torch, params, world: int) -> None:
    """Average the gradients of the adapter over the processes (one flat all-reduce; the adapter is the only thing that has gradients). Done BEFORE the scaler unscales them, so that an overflow on one
    process is an overflow on all of them and every process skips the same step."""
    if world == 1:
        return
    flat = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1) for p in params])
    torch.distributed.all_reduce(flat)
    flat /= world
    off = 0
    for p in params:
        n = p.numel()
        p.grad = flat[off:off + n].view_as(p).clone()
        off += n


def save_checkpoint(torch, model, opt, scaler, step: int, ckpt: pathlib.Path) -> None:
    """`ckpt/step-<step>/` (adapter, optimizer, scaler) written to a `.tmp` folder and renamed, `DONE` last; the two newest are kept. Process 0 only."""
    tmp, final = ckpt / f"step-{step}.tmp", ckpt / f"step-{step}"
    shutil.rmtree(tmp, ignore_errors=True)
    model.save_pretrained(str(tmp / "adapter"))
    torch.save({"opt": opt.state_dict(), "scaler": scaler.state_dict(), "step": step}, tmp / "state.pt")
    (tmp / "DONE").write_text(str(step))
    shutil.rmtree(final, ignore_errors=True)
    os.replace(tmp, final)
    for old in sorted((d for d in ckpt.glob("step-*") if d.name[5:].isdigit()), key=lambda d: int(d.name[5:]))[:-2]:
        shutil.rmtree(old, ignore_errors=True)


def train_worker(cfg: dict, data_dir: pathlib.Path) -> None:
    import torch
    import torch.utils.checkpoint  # noqa: F401  (chunked_loss)
    from transformers import AutoTokenizer
    rank, world, local = dist_setup(torch)
    rows = [json.loads(line) for line in (data_dir / "train.jsonl").read_text().splitlines() if line.strip()]
    tok = AutoTokenizer.from_pretrained(str(STUDENT_DIR))
    examples, counts = encode_records(tok, rows, cfg["max_len"], cfg["mix"])
    gb = cfg["global_batch"]
    if cfg["smoke"]:
        gb = world * 2
        examples = smoke_examples(examples, cfg["smoke_steps"], gb)
    if not examples:
        raise SystemExit(f"nothing to train on: {counts}")
    steps = cfg["smoke_steps"] if cfg["smoke"] else total_steps(len(examples), cfg["epochs"], gb, cfg["max_steps"])
    deadline = float(os.environ["LORA_TRAIN_DEADLINE"])
    ckpt = OUT / "ckpt"
    resume = None if cfg["smoke"] else latest_checkpoint([OUT, INPUT])
    model = load_student(torch, cfg, local, adapter=resume / "adapter" if resume else None, train=True)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda")
    start = 0
    if resume:
        state = torch.load(resume / "state.pt", map_location="cpu")
        opt.load_state_dict(state["opt"])
        scaler.load_state_dict(state["scaler"])
        start = state["step"]
    if rank == 0:
        log(f"train: {len(examples)} examples {counts}, {steps} steps of {gb} sequences on {world} GPU(s), from step {start}" + (f" (checkpoint {resume})" if resume else ""))
    inner = model.get_base_model()          # the LoRA layers live inside it; its `.model` returns the hidden states, its `lm_head` is frozen
    model.train()
    t_run, tokens, step_secs, bad, losses, last = time.time(), 0, [], 0, [], start
    for step in range(start, steps):
        stop = torch.tensor([int(time.time() >= deadline)], device=f"cuda:{local}")
        if world > 1:
            torch.distributed.broadcast(stop, 0)          # one decision for every process: a process that stopped alone would leave the others waiting in the all-reduce
        if int(stop):
            break
        t_step, step_loss, step_tokens = time.time(), 0.0, 0
        mine = rank_slice(step_indices(len(examples), gb, step, cfg["seed"]), rank, world)
        for idx in mine:
            ex = examples[idx]
            ids = torch.tensor([ex["input_ids"]], device=f"cuda:{local}")
            labels = torch.tensor([ex["labels"]], device=f"cuda:{local}")
            with torch.autocast("cuda", dtype=torch.float16):
                hidden = inner.model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).last_hidden_state
                loss, _ = chunked_loss(torch, hidden, labels, inner.lm_head, cfg["loss_chunk"])
            value = float(loss.detach())
            bad = bad + 1 if not math.isfinite(value) else 0
            if bad >= 5:
                raise SystemExit(f"the loss is {value} five micro-batches in a row (fp16 overflow): the T4 cannot train this student in fp16 at this setting; nothing is saved")
            scaler.scale(loss / len(mine)).backward()
            step_loss += value / len(mine)
            step_tokens += len(ex["input_ids"])
        all_reduce_mean(torch, params, world)
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, cfg["max_grad_norm"])
        for g in opt.param_groups:
            g["lr"] = lr_at(step, steps, cfg["lr"])
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        tokens += step_tokens
        step_secs.append(time.time() - t_step)
        losses.append(step_loss)
        last = step + 1
        if rank == 0:
            log(f"step {last}/{steps} loss {step_loss:.4f} lr {lr_at(step, steps, cfg['lr']):.2e} {step_tokens / step_secs[-1]:.0f} tok/s (this process) peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
            if not cfg["smoke"] and last % cfg["save_every"] == 0 and last < steps:
                save_checkpoint(torch, model, opt, scaler, last, ckpt)
        if world > 1:
            torch.distributed.barrier()
    done = last >= steps
    if rank == 0:
        adapter = OUT / ("adapter-smoke" if cfg["smoke"] else "adapter")
        model.save_pretrained(str(adapter))
        info = {"status": "done" if done else "partial", "steps_done": last, "steps_planned": steps, "resumed_from_step": start, "records": counts, "examples": len(examples), "global_batch": gb, "gpus": world,
                "loss_first": losses[0] if losses else None, "loss_last": losses[-1] if losses else None, "wall_s": round(time.time() - t_run), "deadline_hit": not done,
                "tok_per_s_per_gpu": round(tokens / max(sum(step_secs), 1e-9), 1), "step_s": round(sum(step_secs) / max(len(step_secs), 1), 1), "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
        (adapter / "train.json").write_text(json.dumps(info, indent=1))
        if not cfg["smoke"]:
            shutil.rmtree(ckpt, ignore_errors=True)
        log("train finished", info)


def generate_batch(torch, model, tok, items: list, arm: str, cfg: dict, eos_ids: set, local: int) -> tuple:
    """([reply], decode steps) for each (task, sample index) of `items` (one batch), by the thinking template and the config's sampling: [{"text", "finish", "tokens", "prompt_tokens", "secs", "prompt_sha",
    "max_tokens"}], the format of `run_teacher.ask_one`. `secs` is the whole batch's time. `arm` "base" runs with the adapter off, on the very weights the adapter sits on."""
    import contextlib
    prompts = [tok.apply_chat_template(t["messages"], tokenize=False, add_generation_prompt=True, enable_thinking=True) for t, _ in items]
    enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(f"cuda:{local}")
    torch.manual_seed(cfg["seed_base"] + items[0][1])
    t0 = time.time()
    with torch.no_grad(), (model.disable_adapter() if arm == "base" else contextlib.nullcontext()):
        gen = model.generate(**enc, max_new_tokens=cfg["max_new_tokens"], do_sample=True, pad_token_id=tok.pad_token_id, use_cache=True, **cfg["sampling"])
    secs = round(time.time() - t0, 1)
    new, outs = gen[:, enc["input_ids"].shape[1]:], []
    for row, (t, _), mask in zip(new.tolist(), items, enc["attention_mask"]):
        finish, tokens, ids = finish_and_tokens(row, eos_ids)
        reasoning, content = split_generation(tok.decode(ids, skip_special_tokens=False))
        outs.append({"text": run_teacher.stored_text(content, reasoning, True), "finish": finish, "tokens": tokens, "prompt_tokens": int(mask.sum()), "secs": secs,
                     "prompt_sha": run_teacher.prompt_sha(t["messages"]), "max_tokens": cfg["max_new_tokens"]})
    return outs, int(new.shape[1])


def eval_worker(cfg: dict, data_dir: pathlib.Path) -> None:
    import torch
    from transformers import AutoTokenizer
    rank, world, local = dist_setup(torch)
    prefix = "smoke-" if cfg["smoke"] else ""
    adapter = OUT / "adapter-smoke" if cfg["smoke"] else find_adapter([OUT, INPUT])
    if adapter is None:
        raise SystemExit("no finished adapter to evaluate (train.json with status done or partial)")
    tasks = [json.loads(line) for line in (data_dir / "eval_tasks.jsonl").read_text().splitlines() if line.strip()]
    if cfg["smoke"]:
        cfg = {**cfg, "k": 1, "max_new_tokens": cfg["smoke_new_tokens"]}
        tasks = sorted(tasks, key=lambda t: -sum(len(m["content"]) for m in t["messages"]))[:world * cfg["eval_batch"]]          # one full batch per process: `gen_step_s` is measured at the batch the run uses
    previous = {arm: run_teacher.load_previous(previous_files(arm, OUT, INPUT, prefix)) for arm in ARMS}
    items = assign(work_items(tasks, previous, cfg), rank, world)
    tok = AutoTokenizer.from_pretrained(str(STUDENT_DIR), padding_side="left")
    eos_ids = {i for i in (tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")) if isinstance(i, int)}
    model = load_student(torch, cfg, local, adapter=adapter).eval()
    deadline, mean_secs, done, gen_tokens, gen_secs, decode_steps = float(os.environ["LORA_DEADLINE"]), 0.0, 0, 0, 0.0, 0
    lines = {arm: {tid: dict(rec, outs=list(rec["outs"])) for tid, rec in previous[arm].items()} for arm in ARMS}
    log(f"eval (process {rank}): {len(items)} replies per arm to ask of {len(tasks)} tasks, batches of {cfg['eval_batch']}, up to {cfg['max_new_tokens']} new tokens")
    for b in range(0, len(items), cfg["eval_batch"]):
        batch = items[b:b + cfg["eval_batch"]]
        if not run_teacher.can_start(time.time(), deadline, 2 * mean_secs):
            log(f"eval (process {rank}): deadline, {len(items) - b} replies left for a later session")
            break
        t_batch = time.time()
        for arm in ARMS:
            outs, steps = generate_batch(torch, model, tok, batch, arm, cfg, eos_ids, local)
            with open(shard_path(OUT, arm, rank, prefix), "a") as f:
                for (t, i), out in zip(batch, outs):
                    line = lines[arm].setdefault(t["id"], reply_line(t, run_teacher.prompt_sha(t["messages"]), []))
                    line["outs"] += [None] * (i + 1 - len(line["outs"]))
                    line["outs"][i] = out
                    f.write(json.dumps(line) + "\n")
            gen_tokens += sum(o["tokens"] for o in outs)
            gen_secs += outs[0]["secs"]
            decode_steps += steps
        mean_secs = (mean_secs * done + (time.time() - t_batch) / 2) / (done + 1)       # one arm's batch, on average
        done += 1
        log(f"eval (process {rank}): batch {done}/{math.ceil(len(items) / cfg['eval_batch'])}, {gen_tokens / max(gen_secs, 1e-9):.1f} tok/s over this process's batches")
    (OUT / f"{prefix}eval-rank{rank}.json").write_text(json.dumps({"rank": rank, "batches": done, "gen_tokens": gen_tokens, "gen_secs": round(gen_secs, 1), "decode_steps": decode_steps,
                                                                 "gen_step_s": round(gen_secs / max(decode_steps, 1), 4), "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2)}))


# ---- the export (optional, CPU) ---------------------------------------------------------------------------------------------------------------

def export_gguf(cfg: dict, adapter: pathlib.Path) -> dict:
    """The adapter merged into the student's fp16 weights on the CPU, converted by llama.cpp's own `convert_hf_to_gguf.py` and quantized to Q4_K_M (`llama-quantize`, built here for the CPU: the prebuilt
    llama-bin of the teacher kernel holds llama-server only): `/kaggle/working/gguf/lora-Q4_K_M.gguf`. NOT what was evaluated (4-bit NF4 base + adapter): the merge is into fp16 weights and then Q4_K_M."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    merged, src, f16 = pathlib.Path("/tmp/merged"), pathlib.Path("/tmp/llama.cpp"), pathlib.Path("/tmp/lora-F16.gguf")
    model = AutoModelForCausalLM.from_pretrained(str(STUDENT_DIR), torch_dtype=torch.float16, device_map={"": "cpu"}, low_cpu_mem_usage=True)
    model = PeftModel.from_pretrained(model, str(adapter)).merge_and_unload()
    model.save_pretrained(str(merged), safe_serialization=True)
    AutoTokenizer.from_pretrained(str(STUDENT_DIR)).save_pretrained(str(merged))
    del model
    run_teacher.sh(f"git clone --depth 1 https://github.com/ggml-org/llama.cpp {src}")
    run_teacher.sh([sys.executable, "-m", "pip", "install", "-q", "numpy", "sentencepiece", f"{src}/gguf-py"])
    run_teacher.sh([sys.executable, str(src / "convert_hf_to_gguf.py"), str(merged), "--outtype", "f16", "--outfile", str(f16)])
    shutil.rmtree(merged, ignore_errors=True)
    run_teacher.sh([sys.executable, "-m", "pip", "install", "-q", "cmake"])
    run_teacher.sh(f"cd {src} && cmake -B build -DGGML_CUDA=OFF -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DCMAKE_BUILD_TYPE=Release > /tmp/cmake-q.log 2>&1 && cmake --build build --target llama-quantize -j 4 > /tmp/build-q.log 2>&1")
    dest = OUT / "gguf"
    dest.mkdir(exist_ok=True)
    q4 = dest / "lora-Q4_K_M.gguf"
    run_teacher.sh([str(src / "build" / "bin" / "llama-quantize"), str(f16), str(q4), "Q4_K_M"])
    f16.unlink()
    return {"gguf": str(q4), "size_gb": round(q4.stat().st_size / 1e9, 2)}


# ---- the kernel -------------------------------------------------------------------------------------------------------------------------------

def find_data(root: pathlib.Path = INPUT) -> pathlib.Path:
    return next(pathlib.Path(root).rglob("lora-config.json")).parent


def gpu_names() -> list:
    r = subprocess.run("nvidia-smi --query-gpu=name --format=csv,noheader", shell=True, capture_output=True, text=True)
    return [line.strip() for line in r.stdout.splitlines() if line.strip()]


def run_worker(stage: str, gpus: int, data_dir: pathlib.Path, env: dict) -> int:
    """`torchrun` over `gpus` processes of this file (`--worker <stage>`); the return code. Each process is told where the data is and when to stop."""
    cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={gpus}", str(pathlib.Path(__file__).resolve()), "--worker", stage]
    log("$", " ".join(cmd))
    return subprocess.run(cmd, env={**os.environ, "LORA_DATA_DIR": str(data_dir), **env}).returncode


def worker(stage: str) -> None:
    data_dir = pathlib.Path(os.environ["LORA_DATA_DIR"])
    cfg = resolve_config(json.loads((data_dir / "lora-config.json").read_text()))
    (train_worker if stage == "train" else eval_worker)(cfg, data_dir)


def main() -> None:
    if "--worker" in sys.argv:
        return worker(sys.argv[sys.argv.index("--worker") + 1])
    data_dir = find_data()
    cfg = resolve_config(json.loads((data_dir / "lora-config.json").read_text()))
    deadline = T0 + cfg["deadline_h"] * 3600
    prefix = "smoke-" if cfg["smoke"] else ""
    log("config", cfg)
    subprocess.run("nvidia-smi --query-gpu=name,memory.total --format=csv || ls /dev/nvidia*", shell=True)
    names = gpu_names()
    if not names:
        raise SystemExit("no GPU: turn the accelerator on in the kernel's settings (kernel-metadata.json: enable_gpu)")
    if cfg["gpu_name"] and not all(cfg["gpu_name"] in n for n in names):
        raise SystemExit(f"the kernel got {names}, not {cfg['gpu_name']}: set the accelerator to GPU T4 x2 in the kernel's settings (or \"gpu_name\": \"\" in lora-config.json to run on whatever it is)")
    gpus = min(cfg["gpus"], len(names))
    run_teacher.sh([sys.executable, "-m", "pip", "install", "-q", *cfg["pins"], *(["flash-linear-attention"] if cfg["fla"] else [])])
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    from huggingface_hub import snapshot_download
    snapshot_download(cfg["student"], local_dir=str(STUDENT_DIR), allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.txt", "tokenizer*"])
    log("student downloaded", cfg["student"])
    info = {"smoke": cfg["smoke"], "gpus": gpus, "started": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(T0)), "config": cfg}
    existing = None if cfg["smoke"] else find_adapter([OUT, INPUT])
    if existing is None:
        stop = train_deadline(time.time(), deadline, cfg["train_hours"], cfg["eval_reserve_s"])
        info["train_rc"] = run_worker("train", gpus, data_dir, {"LORA_TRAIN_DEADLINE": str(stop)})
    else:
        log(f"finished adapter found: {existing}: training skipped")
    final = (OUT / "adapter-smoke") if cfg["smoke"] else find_adapter([OUT, INPUT])
    if final is None or not (final / "train.json").exists():
        info["error"] = "training left no adapter"
    elif time.time() < deadline:
        info["eval_rc"] = run_worker("eval", gpus, data_dir, {"LORA_DEADLINE": str(deadline - cfg["export_reserve_s"] if cfg["export_gguf"] else deadline)})
        info["replies"] = {arm: merge_shards(OUT, arm, prefix) for arm in ARMS}
    else:
        info["error"] = "the deadline came before the evaluation"
    if cfg["export_gguf"] and final and not cfg["smoke"] and "error" not in info and time.time() + cfg["export_reserve_s"] < deadline:
        try:
            info["gguf"] = export_gguf(cfg, final)
        except Exception as e:      # the adapter and the replies are what the run is for; a failed export never takes them away
            info["gguf"] = {"error": f"{type(e).__name__}: {e}"}
    train_json = OUT / ("adapter-smoke" if cfg["smoke"] else "adapter") / "train.json"
    ranks = [json.loads(p.read_text()) for p in sorted(OUT.glob(f"{prefix}eval-rank*.json"))]
    info["train"] = json.loads(train_json.read_text()) if train_json.exists() else None
    info["measured"] = {**({"train_tok_per_s_per_gpu": info["train"]["tok_per_s_per_gpu"], "step_s": info["train"]["step_s"], "train_peak_gib": info["train"]["peak_gib"]} if info["train"] else {}),
                        **({"gen_tok_per_s_per_gpu": round(sum(r["gen_tokens"] for r in ranks) / max(sum(r["gen_secs"] for r in ranks), 1e-9), 1), "gen_step_s": ranks[0]["gen_step_s"],
                            "gen_peak_gib": max(r["peak_gib"] for r in ranks)} if ranks else {})}
    info["session_seconds"] = round(time.time() - T0)
    (OUT / "run.json").write_text(json.dumps(info, indent=1))
    log("finished", {k: v for k, v in info.items() if k != "config"})


if __name__ == "__main__":
    main()
