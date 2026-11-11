# The Kaggle kernel (`distill/backends/kaggle/`)

Two stdlib-only files: `build_tasks.py` (on this machine: picks the tasks, writes the dataset, the run's `config.json` and the kernel metadata) and `run_teacher.py` (the kernel itself, pushed as `run.py`: llama-server, the requests, the checkpoint, `run.json`). The commands for REA, the config keys, resume and scoring are in [`tools/kaggle/README.md`](../../../tools/kaggle/README.md); this file is what the kernel does with them that is not obvious: the reply budget of a family, the server it starts for it, and the llama-server it reuses instead of building.

## A reply budget per family

`config.json` has `max_tokens` (every task) and, when a family needs more, `max_tokens_by_family` (`{"cipher": 8192, ...}`). The thinking pilot (57 tasks x 2 replies, 4096 tokens) cut off every cipher reply, 2 of 6 magic and password replies and 13% of all of them. The map comes from a run config (`--run-config distill/configs/rea-thinking.json`, the same file the Modal stage takes with `--run-config`) or `--max-tokens-by-family cipher=8192,magic=8192,password=8192`; the family names are the tasks' `kind`, and a name no task has is refused.

The kernel gives each task the budget of its family: the request's `max_tokens`, the prompt-fit check (`prompt tokens + budget <= ctx_per_slot`, else the task is skipped and listed in `run.json` as `skipped_too_long`) and the request timeout (`request_timeout_s` is for `max_tokens`; a longer budget gets a proportionally longer one). `kaggle-score` reads the map from `run.json`: the per-family table shows each family's budget, and the share cut off is read against it.

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

**Not run on Kaggle.** The code path (find, copy, `LD_LIBRARY_PATH`, `--version`, fall back) is tested with a stand-in server script (`tests/test_rea_thinking_pilot.py`); that Kaggle mounts a dataset where `find_prebuilt` looks, keeps the files byte for byte and lets the copy start on the T4 session is the first thing to read in `run_log.txt`.
