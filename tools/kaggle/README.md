# Kaggle teacher runs (free GPU hours for the thinking dataset)

A Kaggle notebook (T4 x2) can sample the teacher for free; the replies come back as a file and are scored on this machine into the same SFT records the Modal stage would write. The kit is part of the distillation package (`distill/backends/kaggle/`, [`distill/README.md`](../../distill/README.md)) and works for any subject; this directory keeps the REA command `build_tasks.py` (a wrapper with REA bound) and this README. Three pieces, nothing else:

| file | what it does |
|---|---|
| `build_tasks.py` (`distill/backends/kaggle/build_tasks.py`; `--domain`, `--teacher`) | picks the tasks by family with a quota, writes `data/tasks.jsonl` (the exact messages of `domain.messages(task, mode)`; for REA `SYSTEM_REASONING` + the user message; nothing of the truth), `data/config.json`, `data/dataset-metadata.json`, and `kernel/run.py` + `kernel/kernel-metadata.json` |
| `run_teacher.py` (`distill/backends/kaggle/`) | the kernel: builds llama.cpp with CUDA (or reuses a built one), downloads the GGUF, runs `llama-server --jinja --reasoning-format deepseek`, asks every task `k` times with `enable_thinking` and the thinking sampling, checkpoints `replies.jsonl`, resumes, stops at the deadline, writes `run.json` |
| `python3 tools/rea-bench kaggle-score` | `distill/score_replies.py` (`python3 -m distill.score_replies --domain rea`, the same code; `rea_bench/kaggle_score.py` is the REA wrapper): `replies.jsonl` + the decompiled tasks -> `sft-<model>-<tag>-reasoning-think.jsonl` and `.stats.json` (README of `tools/modal`, "Scoring a Kaggle run without Modal") |

`tests/test_rea_kaggle_kit.py` runs the whole chain against a stub server (no network, no Kaggle): build, ask, store, resume, deadline, score.

## The token

The Kaggle CLI reads its token from `~/.kaggle/access_token`. **Never print it**: no `cat`, no `echo`, not in a command line, not in a commit, no `set -x` around a Kaggle call, no pasting a log that shows the environment. The files of this directory and of `build_tasks.py`'s output hold no token (a test looks for one).

## One run

```bash
# 1. the inputs: 7 families, the quota of the recipe; --owner is your Kaggle user name
python3 tools/kaggle/build_tasks.py --tasks results/phase6/rea-data/decompiled-seed9-all.jsonl --owner <user> --out ~/work/kaggle-rea \
    --quota structs=10,crc=10,antidebug=10,bounds=10,fsm=10,pktsum=5,encoding=5 --k 4
# 2. the dataset (first time; later: kaggle datasets version -p ~/work/kaggle-rea/data -m "tasks")
kaggle datasets create -p ~/work/kaggle-rea/data
# 3. push the kernel: this starts the run (private, GPU, internet on, the dataset attached)
kaggle kernels push -p ~/work/kaggle-rea/kernel
# 4. status (COMPLETE, RUNNING, ERROR)
kaggle kernels status <user>/rea-teacher
# 5. the output: replies.jsonl, run.json, run_log.txt, llama-bin/
kaggle kernels output <user>/rea-teacher -p ~/work/kaggle-rea/out1
# 6. the records, here
python3 tools/rea-bench kaggle-score --replies ~/work/kaggle-rea/out1/replies.jsonl --tasks results/phase6/rea-data/decompiled-seed9-all.jsonl \
    --model codeforge-27b-heretic-q4 --thinking --max-traces 3
```

`kaggle-score` finds `run.json` beside the replies (or `--run-info`), puts the sampling settings it records into the stats, and refuses a `--thinking` that disagrees with it. It skips, and counts, a replies line whose `prompt_sha` is not the one of the messages it rebuilds. It prints the **projected fit**: of the teacher records, how many fit a 4096 and an 8192 training window (prompt tokens + completion tokens as the server counted them).

## What the kernel does, and what you set

`config.json` (all of it is `build_tasks.py` flags; whatever it leaves out follows `thinking`):

| key | default | |
|---|---|---|
| `thinking` | true | `enable_thinking`: the model's native reasoning. `--no-thinking`: visible reasoning, 768 tokens, temperature 0.7 |
| `sampling` | Qwen3 thinking: temperature 0.6, top_p 0.95, top_k 20, min_p 0 | never greedy (`build_tasks.py` and the kernel refuse temperature 0 and top_k 1 for a thinking run) |
| `max_tokens` | 4096 | per reply; the reasoning counts against it; a reply cut off at it is rejected downstream and counted |
| `k` | 4 | replies per task (RFT: several paths per task; at most 3 become traces) |
| `slots`, `ctx_per_slot` | 4, 8192 | requests in flight; tokens one request may take: the prompt and `max_tokens`. A task whose prompt leaves less than `max_tokens` is not asked (`run.json`: `skipped_too_long`) |
| `deadline_h` | 10.5 | hours after the start at which no new request is started (a GPU session ends at 12 h; a request in flight may take `request_timeout_s`, 1800, more) |
| `seed_base` | 1234 | sample `i` is asked with seed `seed_base + i`, as the Modal stage does |
| `model_repo`, `model_file` | CodeForge 27B heretic i1-Q4_K_M | `--model-repo`, `--model-file` |

Each line of `replies.jsonl`: `{"id", "kind", "prompt_sha", "outs": [{"text", "finish", "tokens", "prompt_tokens", "secs"}]}`. In a thinking run `text` is `<think>\nreasoning\n</think>\n\nvisible reply` (the server's `reasoning_content` and `content`, the form `kaggle-score --thinking` reads); `finish` is `stop`, `length` (cut off: rejected) or `error` (a failed request, `[error ...]` as the text: never scored, asked again by a resume).

## Resume (a session ends at 12 h, the weekly GPU quota is 30 h)

A session that hits the deadline keeps what it answered (the samples of a task that were done are in its file; the ones left are not). Continue with a **second kernel** that has the first one's output attached:

```bash
python3 tools/kaggle/build_tasks.py --tasks <same> --owner <user> --out ~/work/kaggle-rea --quota <same> --k 4 --kernel-slug rea-teacher-2 --resume-from <user>/rea-teacher --force
kaggle datasets version -p ~/work/kaggle-rea/data -m "same tasks"        # only if the tasks or the config changed
kaggle kernels push -p ~/work/kaggle-rea/kernel
```

The kernel reads every `replies.jsonl` under `/kaggle/input` (the earlier output), copies them into its own file, and asks **only the samples that are missing or failed**: a task with every sample answered is not asked, a cut-off reply is an answer. It also looks for `llama-bin/llama-server` in that output, copies it to `/tmp`, makes it executable, and when `--version` runs **skips the llama.cpp build** (about 30 minutes of the 12 hours); if it is not there, or does not start, it builds as usual. Every session saves the server it used into its own output, so a chain of sessions keeps it. **Unverified:** the downloaded output of the first pilot has no `llama-bin` folder, so that the copy lands in the output is not confirmed; and `kaggle kernels output` downloads it with everything else (the CUDA libraries are several hundred MB: look at the free disk first). Score the **last** session's `replies.jsonl`: it holds the earlier ones too (the last line of an id wins). Alternative with no kernel source: download the output, copy `replies.jsonl` into `data/`, `kaggle datasets version`, push again.

## What is known about the speed

From the pilot of 2026-11-11 (57 tasks x 2 visible-reasoning replies of at most 768 tokens, CodeForge 27B heretic Q4_K_M on the T4 x2, 4 slots, `-fa on`): the llama.cpp build took 1,890 s, the pip install and the 15.8 GB download 97 s, the server start 72 s, and the aggregate speed was **18.4 tokens/s** (about 4.6 per slot). **Thinking replies are not measured.** If one is about 2,500 tokens (an assumption, between the 768 of visible reasoning and the 4096 cap), a 10-hour session at that speed writes about 270 replies: about 65 tasks at k=4, so a thinking run on Kaggle is a **pilot-sized** dataset (a few sessions for a few hundred tasks), and the 1,260-task run of the recipe belongs on Modal. A 4096-token reply takes about 15 minutes of one slot: `request_timeout_s` is 1800.

## Not done here

Nothing was run on Kaggle for this version of the kit: the build, the libcuda path and the server flags are those of the pilot notebook, the thinking request and `--reasoning-format deepseek` are **not tried on a GPU** (the first thing to read in `run_log.txt` and in the `kaggle-score` output is the share of replies that came back without reasoning, `no_think`, and the share cut off at `max_tokens`).
