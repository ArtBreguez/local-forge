# Kaggle teacher runs (free GPU hours for the thinking dataset)

A Kaggle notebook (T4 x2) can sample the teacher for free; the replies come back as a file and are scored on this machine into the same SFT records the Modal stage would write. The kit is part of the distillation package (`distill/backends/kaggle/`, [`distill/README.md`](../../distill/README.md)) and works for any subject; this directory keeps the REA command `build_tasks.py` (a wrapper with REA bound) and this README. Three pieces, nothing else:

| file | what it does |
|---|---|
| `build_tasks.py` (`distill/backends/kaggle/build_tasks.py`; `--domain`, `--teacher`, `--run-config`, `--max-tokens-by-family`, `--llama-dataset`) | picks the tasks by family with a quota, writes `data/tasks.jsonl` (the exact messages of `domain.messages(task, mode)`; for REA `SYSTEM_REASONING` + the user message; nothing of the truth), `data/config.json`, `data/dataset-metadata.json`, and `kernel/run.py` + `kernel/kernel-metadata.json` |
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

`kaggle-score` finds `run.json` beside the replies (or `--run-info`), puts the sampling settings it records into the stats, and refuses a `--thinking` that disagrees with it. It skips, and counts, a replies line whose `prompt_sha` is not the one of the messages it rebuilds, and a reply that has **no** fingerprint (a file of an older kernel, the pilot's `out2`: nothing shows which prompt it answered): `prompts.unchecked` counts those lines and the command prints how many; `--keep-unchecked` keeps them on your word that they were asked under this prompt (the stats say so: `prompts.kept_unchecked`). It prints the **projected fit**: of the teacher records, how many fit a 4096 and an 8192 training window (prompt tokens + completion tokens as the server counted them).

## What the kernel does, and what you set

`config.json` (all of it is `build_tasks.py` flags; whatever it leaves out follows `thinking`):

| key | default | |
|---|---|---|
| `thinking` | true | `enable_thinking`: the model's native reasoning. `--no-thinking`: visible reasoning, 768 tokens, temperature 0.7 |
| `sampling` | Qwen3 thinking: temperature 0.6, top_p 0.95, top_k 20, min_p 0 | never greedy (`build_tasks.py` and the kernel refuse temperature 0 and top_k 1 for a thinking run) |
| `max_tokens` | 4096 | per reply; the reasoning counts against it; a reply cut off at it is rejected downstream and counted |
| `max_tokens_by_family` | none | `{"cipher": 8192, ...}` (`--max-tokens-by-family`, or a run config: `--run-config distill/configs/rea-thinking.json`): the budget of the tasks of those families instead of `max_tokens`. Written only when there is one. The kernel sizes its server for the longest budget (next row) |
| `k` | 4 | replies per task (RFT: several paths per task; at most 3 become traces) |
| `slots`, `ctx_per_slot` | 4, 8192 | requests in flight; tokens one request may take: the prompt and `max_tokens`. A task whose prompt leaves less than its budget is not asked (`run.json`: `skipped_too_long`). With a family budget above `max_tokens` the kernel starts 2 slots of 12288 instead (the same total context; `run.json` says what it used; [`distill/backends/kaggle/README.md`](../../distill/backends/kaggle/README.md)) |
| `deadline_h` | 10.5 | hours after the start at which no new request is started (a GPU session ends at 12 h; a request in flight may take `request_timeout_s`, 1800, more) |
| `seed_base` | 1234 | sample `i` is asked with seed `seed_base + i`, as the Modal stage does |
| `model_repo`, `model_file` | CodeForge 27B heretic i1-Q4_K_M | `--model-repo`, `--model-file` |
| `trust_unfingerprinted` | absent (false) | `--trust-unfingerprinted`, or the run config's key: a resume keeps the earlier replies that have no fingerprint, on your word that they were asked under this prompt and `max_tokens`; otherwise they are asked again (below) |

Each line of `replies.jsonl`: `{"id", "kind", "prompt_sha", "outs": [{"text", "finish", "tokens", "prompt_tokens", "secs", "prompt_sha", "max_tokens"}]}`: each reply says which prompt it answered and the budget it was asked with, because a line can hold replies of several sessions (the line's `prompt_sha` is the task's prompt of the session that wrote it). In a thinking run `text` is `<think>\nreasoning\n</think>\n\nvisible reply` (the server's `reasoning_content` and `content`, the form `kaggle-score --thinking` reads); `finish` is `stop`, `length` (cut off: rejected) or `error` (a failed request, `[error ...]` as the text: never scored, asked again by a resume).

## Resume (a session ends at 12 h, the weekly GPU quota is 30 h)

A session that hits the deadline keeps what it answered (the samples of a task that were done are in its file; the ones left are not). Continue with a **second kernel** that has the first one's output attached:

```bash
python3 tools/kaggle/build_tasks.py --tasks <same> --owner <user> --out ~/work/kaggle-rea --quota <same> --k 4 --kernel-slug rea-teacher-2 --resume-from <user>/rea-teacher --force
kaggle datasets version -p ~/work/kaggle-rea/data -m "same tasks"        # only if the tasks or the config changed
kaggle kernels push -p ~/work/kaggle-rea/kernel
```

The kernel reads every `replies.jsonl` under `/kaggle/input` (the earlier output), copies them into its own file, and asks **only the samples that are missing, failed or stale**: a task with none of those is not asked. A reply is *stale* when it answered another prompt than the task's now (its `prompt_sha` differs: a changed `SYSTEM_REASONING`, other tasks) or was cut off (`finish: length`) at a smaller budget than the task's now (its `max_tokens` is below the family's `max_tokens_by_family` entry: a longer budget gives a different reply, not the same cut); a cut-off reply asked with the budget it has now is an answer. A reply of an older kernel that wrote one fingerprint per line takes its line's; **a reply with no fingerprint at all (the pilot's `out2`) is stale too**, since what it answered is not known, and so is a cut-off reply that does not say its budget: both are asked again, unless `trust_unfingerprinted` (config.json; `build_tasks.py --trust-unfingerprinted`) says the file was asked under this prompt and `max_tokens`. **Attaching an earlier output brings its replies with it.** A resume under a new prompt or a longer budget asks again what it must (the log says `N earlier reply(ies) are asked again`, `run.json` has `stale`); the replies it does not get to before the deadline stay in the file with their own fingerprint, and `kaggle-score` skips each one that answered another prompt, counting it under `prompts`. To take **only the server** of an earlier output and none of its replies, attach the `llama-bin` dataset instead (`--llama-dataset`, below) and leave `--resume-from` out. It also looks for `llama-bin/llama-server` in that output, copies it to `/tmp`, makes it executable, and when `--version` runs **skips the llama.cpp build** (about 30 minutes of the 12 hours); if it is not there, or does not start, it builds as usual. Every session saves the server it used into its own output, so a chain of sessions keeps it: the output of the second pilot has the `llama-bin/` folder (about 280 MB, the libraries three times each), and `kaggle kernels output` downloads it with everything else (look at the free disk first). To keep it without chaining kernels, publish that folder as a dataset and attach it with `--llama-dataset <user>/<slug>`: the commands are in [`distill/backends/kaggle/README.md`](../../distill/backends/kaggle/README.md) (**not run on Kaggle**). Score the **last** session's `replies.jsonl`: it holds the earlier ones too (the last line of an id wins). Alternative with no kernel source: download the output, copy `replies.jsonl` into `data/`, `kaggle datasets version`, push again.

## What is known about the speed

From the pilot of 2026-11-11 (57 tasks x 2 visible-reasoning replies of at most 768 tokens, CodeForge 27B heretic Q4_K_M on the T4 x2, 4 slots, `-fa on`): the llama.cpp build took 1,890 s, the pip install and the 15.8 GB download 97 s, the server start 72 s, and the aggregate speed was **18.4 tokens/s** (about 4.6 per slot). **Thinking replies are not measured.** If one is about 2,500 tokens (an assumption, between the 768 of visible reasoning and the 4096 cap), a 10-hour session at that speed writes about 270 replies: about 65 tasks at k=4, so a thinking run on Kaggle is a **pilot-sized** dataset (a few sessions for a few hundred tasks), and the 1,260-task run of the recipe belongs on Modal. A 4096-token reply takes about 15 minutes of one slot: `request_timeout_s` is 1800.

## The thinking pilot (2026-11-11) and what it changed

57 tasks (3 per family) x 2 thinking replies at 4096 tokens, CodeForge 27B heretic Q4 on the T4 x2, 4 slots x 7168 tokens: 7,554 s for 172,692 tokens (22.9 tokens/s aggregate). Scored with `kaggle-score --thinking --max-traces 3`: **89 of 114 replies accepted, 10 wrong, 15 cut off, 13 of the accepted ones dropped as `repetition`**, 75 traces. Three findings, three fixes (branch `fix/thinking-pilot`):

- **antidebug 0 of 6 was the prompt, not the model.** With thinking it computed the right number in all 3 tasks, but the `Answer:` line held only the number (`8204729`) where the question asks for the technique and the number (`timing, 8204729`); `score_strict` was right to reject it. `SYSTEM_REASONING` now says that the last line gives every part the question asks for, in its order, comma-separated, and that a note on the form of one value (`a plain integer`) applies to that value only. In each of the two pilot files (thinking, and the older visible-reasoning one) 56 replies answer a multi-part question and exactly 6 of them, the antidebug ones, leave a part out; no other family does (the other wrong replies have a wrong value, not a missing part). The prompt changed, so its fingerprint did: a `replies.jsonl` asked under the old prompt is refused by `kaggle-score` (`prompts` in the stats).
- **The repetition filter counted the model quoting the question.** A model that reasons re-reads its question: 8-word windows such as `(one of: ptrace, timing, env, tracerpid), and above` occurred 4 to 12 times in replies that did not loop. Windows and lines that are in the task's own prompt (system and user message) are counted against a limit of their own, **more than 12 times** (`PROMPT_REPEAT_MAX`); the thresholds for everything else (a line more than 3 times, an 8-word window more than 3 times) are the same. Why 12 and not 8: four of the 12 replies that only quote the question do so 9, 10, 11 and 11 times (two `bounds` replies re-read `(a subtraction and a width, a bound one off)` 10 and 11 times, a `magic` one 11, another `magic` 9); a limit of 8 drops them (out2: 83 teacher records and one kept task without a trace, instead of 87 and none), 12 keeps all of them. Of the 13 replies dropped, 12 now pass and 1 (a `magic` reply that repeats its own computation four times) is still dropped; the cut-off replies that really loop (a `password` window 17 times, `pktsum`, `rotate`, `cipher`) still count as loops. **The hole that is left, stated:** a reply may re-read the question up to 12 times and be kept (the three that do it 10-11 times are kept as `teacher-reasoning` traces: whether a trace that repeats an instruction that often belongs in the data is a length rule, which this filter is not), and a short line (6 words or fewer) restated with a label that has a number changing each round (`check 3: <a line of the code>`) passes at any count, because every 8-word window of it holds a different number. Such a reply ends; it is not "a loop that does not end and is cut off". A line of 7 words or more with the label, a long prompt line (the windows inside it), and a block of the question copied over and over are caught.
- **4096 tokens was too few for three families.** Every cipher reply (6 of 6) and 2 of 6 of the magic and password replies hit the cap. `max_tokens_by_family` (above) gives them 8192 in `distill/configs/rea-thinking.json`; the cost is a slower server (2 slots, not 4) for a run that has one of them, and a training record of such a reply is longer than the 8192 window `modal_lora` trains on once prompt and reply are added: the length report says how many fit.

## Not done here

Nothing was run on Kaggle for this version of the kit: the build, the libcuda path and the server flags are those of the pilot notebook, the thinking request and `--reasoning-format deepseek` are **not tried on a GPU** (the first thing to read in `run_log.txt` and in the `kaggle-score` output is the share of replies that came back without reasoning, `no_think`, and the share cut off at `max_tokens`).
