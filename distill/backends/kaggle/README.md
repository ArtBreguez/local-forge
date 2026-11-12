# The Kaggle kernel (`distill/backends/kaggle/`)

Two stdlib-only files for the teacher run (the LoRA training and held-out evaluation kit, `build_lora.py` / `lora_eval.py` / `score_lora.py`, has [its own section](#lora-training-and-held-out-evaluation-on-kaggle) at the end): `build_tasks.py` (on this machine: picks the tasks, writes the dataset, the run's `config.json` and the kernel metadata) and `run_teacher.py` (the kernel itself, pushed as `run.py`: llama-server, the requests, the checkpoint, `run.json`). The commands for REA, the config keys, resume and scoring are in [`tools/kaggle/README.md`](../../../tools/kaggle/README.md); this file is what the kernel does with them that is not obvious: the reply budget of a family, the server it starts for it, and the llama-server it reuses instead of building.

## A reply budget per family

`config.json` has `max_tokens` (every task) and, when a family needs more, `max_tokens_by_family` (`{"cipher": 8192, ...}`). The thinking pilot (57 tasks x 2 replies, 4096 tokens) cut off every cipher reply, 2 of 6 magic and password replies and 13% of all of them. The map comes from a run config (`--run-config distill/configs/rea-thinking.json`, the same file the Modal stage takes with `--run-config`) or `--max-tokens-by-family cipher=8192,magic=8192,password=8192`; the family names are the tasks' `kind`, and a name no task has is refused.

The kernel gives each task the budget of its family: the request's `max_tokens`, the prompt-fit check (`prompt tokens + budget <= ctx_per_slot`, else the task is skipped and listed in `run.json` as `skipped_too_long`) and the request timeout (`request_timeout_s` is for `max_tokens`; a longer budget gets a proportionally longer one). `kaggle-score` reads the map from `run.json`: the per-family table shows each family's budget, and the share cut off is read against it.

## What a resume brings with it, and what it asks again

An output attached with `--resume-from` brings its **replies** with it, not only its `llama-server`. Each reply in `replies.jsonl` carries the `prompt_sha` of the messages it answered and the `max_tokens` it was asked with, so a line that mixes sessions stays honest (the line's own `prompt_sha` is only the prompt of the session that wrote it). The kernel asks a sample again when it is **missing, failed, or stale**:

- stale by prompt: the reply's `prompt_sha` (its own, else its line's for a reply that has no `prompt_sha` key) is not the fingerprint of the task's messages now, **or the reply has none**: a reply whose prompt is not known cannot be shown to have answered this one. The pilot's `out2` has no fingerprint anywhere (not on its 57 lines, not on its 114 replies) and was asked under the old `SYSTEM_REASONING` (`system_sha bd6b48eaa3a0131b`): a resume from it asks **every reply again** under the current prompt (`tests/test_rea_thinking_resume.py`, `Out2Shape`), and the log says how many (`N of them carry no fingerprint at all`; `run.json` has `unfingerprinted`);
- stale by budget: `finish` is `length` and its `max_tokens` is below the task's budget now, **or it does not say** (a reply of a kernel that wrote no `max_tokens`: the budget it was cut off at is not known, so the base budget of this config is not assumed for it). A cipher reply cut off at 4096 is asked again when cipher has 8192; one cut off at 8192 is not asked again at 8192 (same seed and budget, the same cut).

A reply that stopped by itself is never stale for its budget. A reply that could not be asked again before the deadline stays in the line with its own fingerprint, **or with none** (`"prompt_sha": null`, for a reply of a file that had no fingerprint, like `out2`: the line's fingerprint, the task's prompt of the session that wrote it, never speaks for a reply of another session); `kaggle-score` checks the fingerprint reply by reply and skips and counts the ones of another prompt (`prompts.partial`) and the ones with no fingerprint (`prompts.unchecked`: skipped unless `--keep-unchecked`). A reply of an older kernel that wrote one fingerprint per line takes its line's.

**When you know the file was asked under this very prompt** (a resume of `out2` under the prompt it was asked with, say), say so: `"trust_unfingerprinted": true` in `config.json` (`build_tasks.py --trust-unfingerprinted`, or the run config's `trust_unfingerprinted`) makes a reply that has no fingerprint take the task's prompt now and the config's `max_tokens`, and stay; each such reply is written with the two fields from then on, in the line of every task of the run whether or not the session asks it again (so no later reader takes the line's fingerprint for it, and `kaggle-score` needs no `--keep-unchecked`: one flag, one decision), and `run.json` records the flag. It is your word, not a check: nothing in the file can contradict it. **To get only the server** of an earlier output, publish its `llama-bin` as a dataset and use `--llama-dataset`; leave `--resume-from` out.

## One server, fewer and longer slots (and why not two passes)

`llama-server -c` is `slots x ctx_per_slot`; the config says `slots` 4 and `ctx_per_slot` 8192 (`slot_context`: a prompt of up to 4096 tokens and a reply of up to 4096). A task of 8192 tokens does not fit that slot. `run_teacher.server_geometry(cfg, tasks)` decides what the server really gets:

- no task of the run has more than `max_tokens`: the config's own numbers, 4 x 8192 (a hand-made config that says 4 x 7168 keeps 4 x 7168);
- some task has `longest` > `max_tokens`: each slot grows by the difference (up to a multiple of 1024) and the slots become as many as the **same total context** holds: 4 x 8192 with an 8192-token family -> **2 x 12288** (a prompt of up to 4096 tokens plus 8192; 24576 tokens of context in all, less than the 4 x 8192 of the shipped default and than the 4 x 7168 the pilot ran). A hand-made 4 x 7168 becomes 2 x 11264.

It is `distill.config.server_plan` (the Modal stage uses it too; a test pins the two together). `run.json` records what was used (`slots`, `ctx_per_slot`, and `server.configured`), and `build_tasks.py` prints it.

**The cost is throughput:** the whole run goes at 2 requests in flight, not only the long families (the aggregate speed of the pilot was 18-23 tokens/s at 4 slots; at 2 it is probably lower, **not measured**). The alternative, a second pass with 2 slots for the long tasks after the short ones at 4, would keep 4 slots for most of the work, but it needs a second server start (about 70 s and the model re-read), a checkpoint that two passes share and a deadline split between them. It was not built; the same result is available with two builds of the kit, **which is what to do when the short families are most of the run**: `build_tasks.py ... --quota <the short families>` and a second kernel (`--kernel-slug`, `--dataset-slug`) with `--quota cipher=...,magic=...,password=...`. A run whose quota has none of the long families starts exactly as before.

## Reusing a built llama-server

The llama.cpp build is about 30 minutes of the 12-hour session (1,890 s and 2,056 s in the two pilots). Every session copies what it used into its own output (`/kaggle/working/llama-bin/`: `llama-server` and its `*.so*`, about 280 MB: the build's symbolic links are written as copies, three of each library). A later kernel skips the build when that folder is attached:

1. `find_prebuilt` looks under `/kaggle/input` for a `llama-server` file, `llama-bin/llama-server` first (the folder of a previous kernel's output, or of a dataset made from it, or a dataset that holds the files at its root).
2. `use_prebuilt` copies `llama-server` and the `*.so*` beside it to `/tmp/llama-prebuilt` (an attached mount is read-only and may have lost the executable bit), puts that folder at the front of `LD_LIBRARY_PATH` (the binary's RUNPATH names `/tmp/llama.cpp/build/bin` of the run that built it; `LD_LIBRARY_PATH` is searched first) and runs `llama-server --version`.
3. If that runs, the server is used (`run.json`: `llama_cpp` says `prebuilt <path>`). If not (a library of that build is missing here, the CUDA of this image differs), `LD_LIBRARY_PATH` is put back as it was and the kernel builds as usual: a wrong attachment costs the build, never the run.

The binary is built for the T4 (sm_75) against the CUDA of Kaggle's image of that day: it is good for T4 sessions and while the image does not change. A model that needs a newer llama.cpp needs a build: leave the dataset off that kernel.

### Publish the previous output's `llama-bin` as a dataset, and attach it

```bash
# 1. the output of the session that built it (replies.jsonl, run.json, run_log.txt, llama-bin/; the folder is ~280 MB, look at the free disk first)
kaggle kernels output <user>/rea-teacher -p ~/work/kaggle-rea/out2
# 2. a folder with the files of llama-bin and the dataset metadata (llama.cpp is MIT licensed)
mkdir -p ~/work/llama-dataset && cp ~/work/kaggle-rea/out2/llama-bin/* ~/work/llama-dataset/
echo '{"title": "llama-bin-t4", "id": "<user>/llama-bin-t4", "licenses": [{"name": "MIT"}]}' > ~/work/llama-dataset/dataset-metadata.json
# 3. upload it (private; the Kaggle CLI uploads the files of the folder and skips sub-folders unless it is given -r zip)
kaggle datasets create -p ~/work/llama-dataset
# 4. attach it to the next kernel: the flag adds it to the kernel's dataset_sources next to the tasks
python3 tools/kaggle/build_tasks.py ... --llama-dataset <user>/llama-bin-t4 --force
kaggle kernels push -p ~/work/kaggle-rea/kernel
```

`run_log.txt` of that kernel starts with `llama-server from an attached output: /kaggle/input/...` and no `git clone`; if it says `does not start here; building llama.cpp`, read what came before. Keeping `llama-bin/` as a folder in the dataset also works (`cp -r` it into step 2's folder and upload with `kaggle datasets create -p ... -r zip`, which Kaggle should unpack again): **not tried**, the flat folder is the form the steps above rely on, and `find_prebuilt` takes both. A dataset update after a rebuild: `kaggle datasets version -p ~/work/llama-dataset -m "<llama.cpp commit>"`.

**Not run on Kaggle.** The code path (find, copy, `LD_LIBRARY_PATH`, `--version`, fall back) is tested with a stand-in server script (`tests/test_rea_kaggle_llama_bin.py`); that Kaggle mounts a dataset where `find_prebuilt` looks, keeps the files byte for byte and lets the copy start on the T4 session is the first thing to read in `run_log.txt`.

## LoRA training and held-out evaluation on Kaggle

Does the thinking dataset make the student better? This kit trains the **same student as `modal_train.py`** (`xtools-at/DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic`, `distill.config.DEFAULT_STUDENT`, the base of REA's LoRA) on an SFT file of `kaggle-score --thinking`, on the free **2 x T4 (16 GB)**, then has the base student and base + adapter answer the **same held-out tasks** (seed 1001) with thinking, and scores both here, where the truth is. Three files: `build_lora.py` (your machine: checks, config, dataset, kernel, the cost), `lora_eval.py` (the kernel), `score_lora.py` (your machine: the report). Tests: `tests/test_kaggle_lora.py`. This kit was **not run on a GPU** and nothing was pushed to Kaggle: every figure below that is assumed has to be measured by the smoke run first.

### The commands

The held-out tasks are the decompiled seed-1001 file of the decompile stage (30 per new family; `tools/rea-bench/rea_bench/heldout-1001.txt` pins the tasks of the split). The SFT file is what `kaggle-score --thinking` wrote.

```bash
modal run tools/modal/modal_rea_data.py --stage decompile --seed 1001 --family all --n 330    # only if results/phase6/rea-data/decompiled-seed1001-all.jsonl does not exist
```

1. Build the **smoke** run (4 steps on the longest records, two replies per process: ~0.7 h of quota) and look at the estimate it prints:

```bash
python3 -m distill.backends.kaggle.build_lora --domain rea --owner <kaggle-user> --out ~/work/kaggle-lora-smoke --smoke \
    --sft results/phase6/rea-data/sft-<teacher>-<tag>-reasoning-think.jsonl --heldout results/phase6/rea-data/decompiled-seed1001-all.jsonl
```

2. Create the dataset and push the kernel; **set the kernel's accelerator to GPU T4 x2** (`kernel-metadata.json` cannot choose it: `enable_gpu` alone may give a P100, and the kernel stops before the download when a GPU is not a T4):

```bash
kaggle datasets create -p ~/work/kaggle-lora-smoke/data
kaggle kernels push -p ~/work/kaggle-lora-smoke/kernel
kaggle kernels status <kaggle-user>/rea-lora-eval
kaggle kernels output <kaggle-user>/rea-lora-eval -p ~/work/kaggle-lora-smoke/out
```

3. Read `out/run.json`: `measured` has the training speed per GPU, the step time, the generation step time and the **peak memory** of both workers. If the peak is above ~14.5 GiB lower `--max-len` (3072, 2048) or `--rank` before the real run. Then build the real run with what was measured (`--measured` replaces the assumed rates of the estimate):

```bash
python3 -m distill.backends.kaggle.build_lora --domain rea --owner <kaggle-user> --out ~/work/kaggle-lora \
    --sft results/phase6/rea-data/sft-<teacher>-<tag>-reasoning-think.jsonl --heldout results/phase6/rea-data/decompiled-seed1001-all.jsonl \
    --measured ~/work/kaggle-lora-smoke/out/run.json --per-family 10 --max-len 4096 --k 1 --train-hours 3
```

4. Push it the same way (`kaggle datasets create -p ~/work/kaggle-lora/data`, `kaggle kernels push -p ~/work/kaggle-lora/kernel`, the same accelerator), wait, fetch the output, and score it:

```bash
python3 -m distill.backends.kaggle.score_lora --domain rea --heldout results/phase6/rea-data/decompiled-seed1001-all.jsonl \
    --base ~/work/kaggle-lora/out/replies-base.jsonl --lora ~/work/kaggle-lora/out/replies-lora.jsonl --run-info ~/work/kaggle-lora/out/run.json --out ~/work/kaggle-lora/out/report.json
```

A session that ends early (quota, crash, the deadline) is **resumed**, not restarted: build again with `--resume-from <kaggle-user>/rea-lora-eval --force` (same options), push, and the new kernel finds the newest complete `ckpt/step-N` or the finished adapter and the replies of the output it has attached, and asks only what is missing, failed or stale (`run_teacher.pending`, the teacher kernel's rule). `--export-gguf` (off) also merges the adapter into fp16 weights and writes `gguf/lora-Q4_K_M.gguf` (~5.6 GB, far under the 20 GB output limit) at the end, time permitting: **not tried**, and not what was evaluated (see below).

### What it checks before anything is written

`build_lora.py` refuses: a training record of a **held-out** task (`domain.heldout_ids`: for REA the ids that end in `-1001-<n>`), a training record of a task in the **evaluation subset**, a `--heldout` file that holds a task that is not held-out, a teacher-reasoning record whose **label is a teacher that may only filter** (`teachers.require_reasoning`, the rule `kaggle-score` applies: gpt-oss, claude and the rest of `FILTER_ONLY_MARKERS`; a truth-template record's label is only the filter), an SFT file with no thinking record, and an earlier build without `--force`. The subset is `--quota structs=10,crc=10` or, by default, `--per-family` (10) tasks of each family the training records have (the first by id, round robin).

### What runs, and the decisions

| | |
|---|---|
| training | `lora_eval.py --worker train` under `torchrun`, one process per T4, each with the 4-bit student (**NF4**, **fp16**: a T4 has no bf16); gradient checkpointing; batch 1 per process, 8 accumulation steps, 16 sequences per step as in `modal_train.py`; AdamW, lr 1.5e-4, cosine, 5% warmup; LoRA on `distill.sft.LORA_TARGETS`, alpha 2 x rank, dropout 0.05; rank **16** (Modal: 32: the optimizer state of the adapter is fp32); window **4096** (Modal's thinking config: 8192, which a T4 cannot hold). The records become training examples with `distill/sft.py`, the code `modal_train.py` runs (moved there unchanged; the Kaggle dataset ships that file), so the thinking format, the loss mask and the `--mix` rule are the Modal run's |
| the loss | computed in **chunks** of 512 target positions inside `torch.utils.checkpoint`: the student's vocabulary is 248,320 tokens, so the logits of one 4096-token sequence are 4.07 GB in fp32 (and the cross-entropy keeps copies), more than a T4 has next to an ~8 GB model; with chunks they never exist whole. The processes average the adapter's gradients by hand, before the scaler unscales them (an overflow on one is an overflow on all) |
| evaluation | **HF transformers**, not llama-server: the base arm is `model.disable_adapter()` on the very weights the adapter sits on, so one load, no GGUF conversion, and both arms see the same quantization. The prebuilt `arthurbreguez/llama-bin-t4` holds `llama-server` only (no `llama-quantize`, no convert script), the base GGUF of this student **failed to convert** in the Modal smoke run (`model-mtp.safetensors`), and two conversions + quantizations would cost ~40 min before the first reply. **Nothing here serves a GGUF, so that dataset is not attached.** Cost of the choice: the arms run on NF4, not on the Q4_K_M that `modal_lora` makes |
| sampling | the thinking template (`enable_thinking`, the prompt ends in `<think>\n`), Qwen3's thinking settings (never greedy), `--k` replies per task and arm, `--max-new-tokens` 4096 (the pilot cut every cipher reply off at 4096: read the cut-off column), batches of 8, sorted by prompt length. `generate` seeds a batch, not a reply |
| scoring | `score_lora.py`: `score_replies.reply_of` and `traces.verdict` (the code of `kaggle-score`; REA's `domain.verify` is `scoring.score_strict`, equal to rea-bench's held-out `score` for the generated families), a reply to another prompt refused. Per family and arm: accuracy with its 95% **Wilson** interval, wrong / no `Answer:` line / truncated / error, the share cut off at the budget, mean tokens; and the arms **paired** (only base, only adapter, exact sign test). Ten tasks per family give wide intervals: read the overall paired line first |
| deadline | nothing is started after `--deadline-h` (7.5 h: the weekly quota, not the 12 h session, is the limit); training stops at `--train-hours` (3) or early enough to leave the evaluation's upper bound before the deadline; a generation batch is not started when 2 x the mean batch would pass it; replies are appended per batch and the adapter is checkpointed every 10 steps; a partial adapter (`status: partial`) is evaluated and said so |

### The estimate (assumed, not measured)

`build_lora.py` prints it. With the defaults, 70 held-out tasks and ~1,200 thinking records of ~3,000 tokens it says: 220 tok/s per GPU (6 x 9e9 FLOP per token at ~12 TFLOP/s of the T4's 65) -> ~109 s per step -> the 3 h budget caps training at ~99 steps (~1.3 epochs); evaluation 70 replies per arm, 5 batches per process and arm, up to 4096 steps of 0.12 s -> 1.4 h at most (~1.1 h typically); setup 0.33 h and two model loads 0.17 h: **~4.9 h at most of the 8 h** (the rest is room for `--k 2`). Every one of these figures is an assumption until the smoke run measures it; if the verdict line says `IT DOES NOT FIT`, lower `--train-hours`, `--k`, `--per-family` or `--max-new-tokens`.

### Not tried, and what to read first

- **Memory.** The estimate per GPU is ~8 GB of weights (embedding and `lm_head` in fp16: 4.1 GB), ~1 GB of adapter and optimizer state, ~1 GB of checkpointed activations, ~2 GB of chunked loss and workspace: about 13.5 of 15 GB, from the config alone. Nothing was run. The smoke run builds on the longest records for that reason.
- **fp16.** The student was trained in bf16; its activations may overflow in fp16. The kernel stops after five non-finite losses in a row instead of burning the quota (`nothing is saved`).
- **Linear attention.** 24 of the 32 layers are linear-attention layers: without `flash-linear-attention` transformers uses a slow PyTorch path (time and memory); `--fla` installs it and is not tried on a T4.
- **Template.** `sft.render_record(check=True)` refuses a tokenizer whose generation prompt does not end in `<think>\n`; the generation is decoded with the special tokens kept, split at `</think>`, and a block that never closed is a cut-off reply.
- **Pins.** `lora_eval.DEFAULTS["pins"]` are `modal_train.py`'s (`transformers==5.22.2`, `peft==0.21.2`; a test compares them) plus `bitsandbytes`; Kaggle's torch may be replaced by pip, and `from_pretrained(torch_dtype=...)` is the call the Modal run uses.
- **The GGUF** (`--export-gguf`) merges into fp16 weights and quantizes to Q4_K_M: not the NF4 model that was evaluated, and its conversion is not tried (`llama-quantize` is built for the CPU in the kernel).
