# distill: teach a student model a subject by sampling a teacher

A teacher (an open-weight model) answers a subject's tasks; the answers are checked **by code**; what passes is filtered, de-duplicated and written as SFT records; a student is fine-tuned on them and evaluated on held-out tasks. This package is that process without the subject: reverse engineering (REA) is the first subject, and the same pipeline trains other students on other subjects.

Everything in `distill/*.py` is the standard library only and imports without Modal. Modal, Kaggle and the model libraries are in `backends/`. A subject is a `Domain` in `domains/`.

```
distill/
  domain.py          the interface a subject implements (Domain, MODES)
  traces.py          Answer-line extraction, think handling, the R1-style quality filters, duplicates, max traces
  records.py         record building, the record schema (check_records), the agreement rule, stats, the length rule
  config.py          RunConfig (one run as JSON), sampling of each mode, token budget, slot, quota
  outputs.py         file names by tag, the checkpoint, atomic writes, refuse-to-overwrite, the prompt fingerprint
  score_replies.py   `python -m distill.score_replies`: replies.jsonl + a domain -> SFT records + stats
  teachers.py        the teacher list (name, GGUF repo, file, Modal GPU) shared by Modal and Kaggle
  configs/           rea-plain.json, rea-reasoning.json, rea-thinking.json: today's REA runs as run configs
  backends/
    modal_sample.py  the sample stage (llama-server, checkpoint/resume, length rule, cost line); no Modal import
    modal_train.py   LoRA fine-tune of a student, merge, GGUF of base and trained model (needs Modal)
    kaggle/          build_tasks.py (inputs of a free teacher run), run_teacher.py (the kernel, stdlib only)
  domains/
    rea/             REA: the adapter between Domain and tools/rea-bench
    example_arith/   a TEST FIXTURE: arithmetic word problems; proves the core does not depend on REA
```

## The pipeline

| step | what | where | the rule |
|---|---|---|---|
| 1 tasks | the subject generates tasks with a ground truth it computed itself | the subject (REA: `tools/rea-bench`, then the Ghidra decompile stage of `tools/modal/modal_rea_data.py`) | the answer is computed from the task's own parameters, never by a model; held-out tasks are a split no training run draws |
| 2 sample | the teacher answers each task `k` times | `backends/modal_sample.py`, or a Kaggle notebook (`backends/kaggle`) | open-weight teachers only (`teachers.py`); a thinking run never samples greedily; a failed request is never scored or stored |
| 3 verify | the final answer is checked against the truth | `Domain.verify` | by code. In `plain` mode the whole reply is the answer; in `reasoning` / `thinking` the text after the last `Answer:` line (`traces.final_answer`), and a reply cut off at `max_tokens` is rejected |
| 4 filter | a task is kept when a sample is verified, and when two are for a task a guess solves often | `records.select*`, `Domain.needs_agreement` | agreement filters luck, it does not make a task hard |
| 5 traces | up to `max_traces` distinct clean verified replies per task | `traces.quality_drop`, `traces.pick_traces` | filters: mixed language, a repetition loop, an analysis that only restates the answer, a plain contradiction, and (thinking) a missing reasoning block; then duplicates (identical normalized text, or token-set Jaccard >= 0.9) |
| 6 records | `{messages, task, kind, model, target_source}` and nothing else | `records.assemble`, `records.check_records` | `target_source` is `truth-template` (rendered from the truth, the teacher's text is not stored) or `teacher-reasoning` (the reply verbatim). **The steps of a teacher record are not checked: some may be wrong while the answer is right; a dataset card must say so** |
| 7 length | a record longer than the training window is dropped and counted, never cut | `records.drop_over_length` (sampler's tokenizer) and `modal_train.encode_ids` (training tokenizer) | one window, `--max-len` |
| 8 train | LoRA on the student; `<think>` records are trained as the student's chat template renders them | `backends/modal_train.py` | `--mix` also trains the truth records, with an empty think block |
| 9 evaluate | the student against the base on the held-out split | the subject (REA: `tools/modal/modal_rea.py`, not generalized: see the end) | a gain is read on the strict score and its format-insensitive control |

Modes: `plain` (the teacher answers directly; the stored text is the domain's `truth_target`), `reasoning` (short steps, then `Answer: <value>`; the steps are stored), `thinking` (the model's native reasoning, stored as `<think>reasoning</think>` + the visible reply, the way the R1 distills are made).

## The interface of a subject

`distill/domain.py`. A `Domain` has:

| member | must guarantee |
|---|---|
| `name` | the package name under `domains/` |
| `load_tasks(source)` | the tasks of a path, a list of paths or a seed: dicts with a unique `id` and a `kind` (the family a quota counts by), in a stable order; the order is the order records come out in |
| `messages(task, mode)` | `[system, user]` for `plain`, `reasoning`, `thinking`; the same messages are asked and stored; **nothing of the ground truth in them** |
| `verify(final_answer, task)` | computed by code from the task's own truth, **never by a model**; False for anything it cannot parse, no exception on a reply; it sees only the final answer, whatever the teacher wrote before it. It is the strictest thing in the pipeline: a loose `verify` puts wrong answers in the data. Prefer a whitelist (what is an answer) to a blacklist (what is not) |
| `needs_agreement(task)` | True for a task a guess solves with a probability that is not small (a yes/no, a small integer): it is kept only when `records.AGREE_MIN` (2) samples are verified |
| `truth_target(task)` | the answer text rendered from the truth alone, accepted by `verify`; or None (then a `plain` run refuses: it has nothing to store, and a `reasoning` run stores traces only) |
| `heldout_ids()` | anything that supports `task_id in x`: the evaluation split. A held-out task is refused by the scorer and the kit; **a held-out task must also be disjoint from the training tasks as a problem, not only as an id** (REA hashes the program with its decoys stripped, and, for free-string answers, the answer: `tools/rea-bench/rea_bench/dedupe.py`) |

Optional, with a default that is right for a subject that does not need them: `check_task` (fail before a GPU is rented; default: `truth_target` passes `verify`), `contradicts`, `canon_word`, `task_words` (the trace filters that read the subject's answers; REA's read the numbers of its answer grammar), `kinds` (the names a `--quota` may use), `tasks_prefix` (a tasks file's name before the run tag). They exist because REA needs them to keep its behaviour exactly; a second subject that needs none of them leaves them alone.

## How to add a subject

1. Write its task generator and ground truth. The truth is computed by code and proven by a harness (REA compiles each program and runs it).
2. Create `distill/domains/<name>/__init__.py` with a `Domain` subclass and `DOMAIN = YourDomain()`. Start from `domains/example_arith` (about 60 lines). Wrap the subject's own library; do not copy it. Keep a held-out split that no training run draws, and say how it is disjoint.
3. Run the contract: add the subject to `Contract` in `tests/test_distill_core.py` (unique ids, `verify` accepts its own truth and rejects the empty answer, held-out ids disjoint from training, messages well formed).
4. Score a first sample by hand: `python3 -m distill.score_replies --domain <name> --replies replies.jsonl --tasks tasks.jsonl --model <teacher> --max-traces 3` on the output of any llama-server run (the Kaggle kit writes that file). Read the per-family table: the share of replies cut off, wrong, filtered, duplicate.
5. To sample on Modal, add a wrapper like `tools/modal/modal_rea_data.py`'s sampler (15 lines; the image needs llama.cpp, `huggingface_hub`, `distill/` and your library):

```python
@app.function(gpu=["L40S", "A100-40GB"], volumes={"/models": models_vol, "/data": data_vol}, timeout=modal_sample.SAMPLER_TIMEOUT_H * 3600)
def sample_answers(name, repo, fname, k, tag, resume=False, reasoning=False, max_tokens=0, max_len=0, quota="", thinking=False, max_traces=3, sampling=None):
    data_vol.reload()
    return modal_sample.sample_answers(DOMAIN, data_vol, tasks_source, name, repo, fname, k, tag, resume, reasoning, max_tokens, max_len, quota, thinking, max_traces, sampling)
```

6. Train: `modal run tools/modal/modal_lora.py --config <your dataset config> --tag <name> [--student ...] [--max-len ...]` on the uploaded records, then evaluate on the held-out split with the subject's own scorer.

## A different student

The student is the model the LoRA is trained on. `--student <huggingface repo>` of `modal_lora.py` (or `student` in a run config) replaces the default, the 9B uncensored that REA's LoRA is trained on. Both the trained model and the base are converted by the same path (llama.cpp convert, Q4_K_M), so the before/after compares like with like. What the student must have:

- a tokenizer whose chat template renders an assistant turn the way the Qwen3 family does for reasoning records: `<think>\nreasoning\n</think>\n\nanswer` after the last user message, and the generation prompt ending in `<think>\n` (or `<think>\n\n</think>\n\n` with `enable_thinking=False`). `modal_train.render_record(check=True)` raises when the template ends a prompt differently, instead of placing the loss mask wrongly. A visible-reasoning or plain file does not need this;
- an architecture llama.cpp can convert, and weights that fit one A100-40GB with gradient checkpointing at the window you choose (`modal_train.batch_plan`: 4 x 4 at 2048, 1 x 16 at 8192; only 2048 is measured);
- LoRA target modules: `train()` names the attention and MLP projections of the Qwen family and its linear-attention projections; check that list against the student's own module names (`model.named_modules()`) when it is another family.

The output is `/out/<tag>/` on the `lf-lora` volume and `results/phase6/lora/<tag>-train.json` (it records the student). `full` and `smoke` are reserved for the default config's default run, so a new student never overwrites phase A's output.

## Run configs

`config.RunConfig` is one run as JSON: `domain`, `teacher` (a name in `teachers.py`), `student`, `k`, `mode`, `sampling` (None: the mode's), `max_tokens` (0: the mode's), `max_traces`, `max_len`, `mix`, `quota`. A key that is not a field is refused. `distill/configs/rea-*.json` are today's REA runs (`tests/test_distill_core.py` pins what each resolves to; `rea-thinking.json` carries the pilot quota). `modal_rea_data.py --run-config F` and `modal_lora.py --run-config F` take their flags from a file, and refuse a flag that is also given on the command line.

## The rules

- No output of an OpenAI or Anthropic model enters a dataset; every teacher is open-weight (`teachers.py`).
- The answer is verified by code against a truth the generator computed; a model never judges a model.
- Held-out tasks never enter training, as ids and as problems.
- The record format is closed: `check_records` refuses any other key, so nothing of a teacher leaks into a file by accident.
- A thinking run is never greedy; a reply cut off at `max_tokens` is rejected; a record over the window is dropped, never cut.

## References (already in the repo's recipe)

- Rejection sampling and self-taught reasoning: STaR (Zelikman et al. 2022, arXiv:2203.14465) keeps the rationales whose answer is right; the rule here is its filter, the final answer checked by code, with the reasoning kept as the model wrote it.
- RFT, rejection-sampling fine-tuning (Yuan et al. 2023, arXiv:2308.01825): several distinct correct paths per question are worth more than one; `max_traces` and the duplicate rule.
- DeepSeek-R1 (DeepSeek-AI 2025, arXiv:2501.12948): its SFT data is built by rejection sampling with filters on the chains of thought (mixed languages, and others); `traces.quality_drop` is the same kind of cleanup with our own thresholds, which are constants at the top of `traces.py`. The R1 distills are base models fine-tuned on reasoning traces a larger model wrote: the `thinking` mode.
- Qwen3 (model card): temperature 0.6, top_p 0.95, top_k 20, min_p 0 for thinking, never greedy (`config.THINKING_SAMPLING`).

## What is not generalized, on purpose

- **Evaluation.** `tools/modal/modal_rea.py` (the held-out arm, the 10 fixed questions, the agent arm, the frozen substring scorer) stays REA's. Its held-out arm reads REA's decompiled files and its scorer, and the 34/60 -> 50/60 numbers are comparable only while it does not change. A second subject writes its own evaluation against its `heldout_ids()` and its `verify`.
- **The task stage.** Generating tasks and computing their truth is the subject's (REA: the families, the compile and fact gates, Ghidra). `Domain.load_tasks` only reads what it wrote.
- **The cost line.** `modal_sample.project` scales the one measurement there is (REA's phase A on one L40S) by reply length. It is a budget guard, not a forecast, and another subject's replies have another length.
- **The Modal images.** A subject's wrapper brings its own image (REA's has Ghidra, gcc and clang); `modal_sample.sample_answers` is the body that runs in it.
