# Phase 3: does a speed setting change the answer?

A fixed set of 40 small coding problems (20 Python, 20 JavaScript) with hidden unit tests, and the tools to run them against one
server configuration at a time. Issue #6; the results and the decisions are in `docs/JOURNAL.md`, the raw outputs in `results/phase3/`.

## What is here

| Path | What |
|---|---|
| `problems/<id>/prompt.md` | the task, the only thing the model sees (written for this repo; nothing copied from HumanEval/MBPP) |
| `problems/<id>/tests.py` / `tests.js` | hidden tests (`python -m unittest`, `node --test`), standard library only |
| `_ref/<id>/good.*`, `bad.*` | a reference solution (all hidden tests pass) and a broken one (at least one fails). `tests/test_quality.py` checks both for every problem |
| `manifest.json` | id, language, entry points, difficulty, token budget, tags, `subtle` flag |
| `qbench.py` | prompt, code extraction, test execution, Wilson interval, exact McNemar |
| `run_quality.py` | one arm against one OpenAI-compatible endpoint, writes JSONL |
| `qgrid.py`, `arms.json` | the host driver: a fresh `forge serve` per arm, the runner, stop everything |
| `qreport.py` | the table for the journal |

Mix: 9 easy, 21 medium, 10 hard; string and format parsing, date and time edge cases (ISO weeks, month ends, leap days, UTC), small
algorithms, input validation with the exception type specified, and 15 problems that have one subtle edge case (marked `subtle`).
Every prompt names the functions the tests import, the error types, and the behaviour at the edges that the tests check. An
independent read of all prompts against all tests (one reviewer that saw only the files) found seven spots where a correct answer
could have failed unfairly or a requirement was not enforced; they are fixed.

## What one request looks like

* `POST <endpoint>/chat/completions`, temperature 0, `seed` 1234, `cache_prompt` false (a request never depends on the previous one),
  `stream` false. `max_tokens` is the problem's budget (512 to 1024) and, for a thinking arm, plus the thinking budget (default 2048).
* Two messages, identical for every arm: a system message ("one complete solution in a single fenced code block and nothing else") and
  the user message (a one-line language note + `prompt.md`).
* **Thinking is a template switch, sent on every request:** `"chat_template_kwargs": {"enable_thinking": false | true}`. This model's
  chat template (Qwen3.5 family, read from the GGUF) ends the generation prompt with `<think>\n` unless `enable_thinking` is explicitly
  false, in which case it appends an empty `<think>\n\n</think>\n\n`. So the default, with no kwarg, is thinking ON. There is no
  `/think` tag in this template.
* Code is taken from the first fenced block whose tag fits the language (else the first block, else an unterminated fence, else the whole
  answer if it looks like code). Reasoning is removed first and never searched for code; reasoning that never closes counts as no answer.
  For JavaScript, `export` keywords are dropped and a CommonJS `module.exports` line is added when the answer has none (recorded as
  `normalized_exports`); no other repair is made.
* The code is written to a fresh temp dir next to the hidden tests and run with a 30 s timeout that kills the whole process tree.
  Failure kinds: `no_code`, `syntax` (compile / `node --check`), `test_fail`, `timeout`, `api_error`. A pass needs every hidden test.
  `cap_hit` is `finish_reason == "length"`.

**Safety:** the runner executes model-written code with no sandbox beyond a temp dir, a timeout and a closed stdin. Run it only on a
machine where that is acceptable.

## Running

```
# anywhere with a server: one arm
python run_quality.py --endpoint http://127.0.0.1:8080/v1 --arm A --thinking off --out ../../results/phase3/A.jsonl
python qreport.py ../../results/phase3 --baseline A

# on the host (Windows), every arm in arms.json order, as a scheduled task so an SSH drop does not matter
powershell -File tools\phase3-host.ps1 -Arms A,B,C,D,E -Tag full
```

`qgrid.py` starts `python -m forge serve <flags of the arm>` (so the planner places the layers), waits for `/health`, stores what
`forge serve` printed (exact llama-server command line, chosen layers) in `<arm>.serve.json`, runs the runner, then kills the process
tree and checks that no `llama-server` is left. A rerun resumes: the runner skips (problem, sample) pairs already in the file.

Tests of all of this: `python -m unittest tests.test_quality` (about 30 s; needs `node`).

## What 40 problems can and cannot show

Pass rates come with a Wilson 95% interval and arms are compared with an exact McNemar test on the same problems. With 40 problems a
single problem moves the rate by 2.5 points, an interval is about +-13 points wide, and the paired test needs roughly six or more
problems that flip in the same direction (and almost none in the other) to reach p < 0.05. A difference smaller than that is reported
as within noise, which is a statement about this benchmark's resolution, not a proof that two settings are equal.
