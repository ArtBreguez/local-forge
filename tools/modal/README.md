# Modal pipeline (rented GPU lab)

Everything local-forge ran on [Modal](https://modal.com): model comparison, the REA benchmark, the REA training data, the LoRA, the refusals check and the Hugging Face upload. The host (4 GB GPU) only measures speed and fit; quality is measured here, every model in the same container, same llama.cpp build (pinned by digest), same prompts, temperature 0.

Numbers from Modal (CUDA, llama.cpp b12211) are not directly comparable with the host (Vulkan, b12207): compare models within one environment.

## Setup (once)

```bash
uv tool install modal
modal token set --token-id <id> --token-secret <secret> --profile <name>   # type it yourself, never paste tokens in chat
modal profile activate <name>
# Hugging Face write token, for the gated/private dataset and the model upload:
read -rs HF; modal secret create hf-write HF_TOKEN="$HF"; unset HF
```

Volumes are created on first use: `lf-models` (downloaded GGUFs, checked against SHA-256), `lf-lora` (training output and our own GGUFs).

## The steps, in order

| # | Script | What it does | Writes | Cost seen (Nov 2026) |
|---|---|---|---|---|
| 1 | `modal_qbench.py` | 40 coding problems with hidden tests (same runner as phase 3, `03-agent/quality`): first try at temp 0 and 10 samples at temp 0.7, per model in `MODELS` | `results/phase6/modal/` | US$ 3–4 per 6 models |
| 2 | `modal_rea.py` | REA benchmark: 10 fixed questions on one stripped binary (`rea-bench/targets.c`), recipe arm (Ghidra pseudocode given) and agent arm (`funcs`, `decomp`, `decomp-all`, `refdata` tools, 12 steps), plus the generated held-out set (seed 1001) as a third arm, reported on its own line with a strict score (headline only) and a format-insensitive control (any line of the reply) | `results/phase6/rea/` | ~US$ 1 per model |
| 3 | `modal_rea_data.py --stage decompile [--family legacy\|all\|crc,cipher,...] [--cc mix\|gcc\|clang] [--exclude-seed 1001] [--force]` | Generate tasks with `tools/rea-bench` (11 families, automatic answer key; default `legacy` = the original 10 templates, whole-reply scoring: prefer `--family all`), drop every task that is in the held-out list, compile stripped with each task's own compiler and -O level, re-check that the asked constant is in that binary, decompile `proc` (and the helpers it calls) with Ghidra, resolve referenced bytes. A seed-1001 run is refused above 120 tasks per family; an error summary says why tasks failed (fact gate / timeout / no pseudocode) per family | `results/phase6/rea-data/decompiled-seed<seed>-<family>.jsonl` | ~US$ 1 per 1,200 |
| 4 | `modal_rea_data.py --stage sample --model <teacher> --k 4 [--seed 7 --family legacy] [--resume] [--force]` | The teacher (uncensored, e.g. CodeForge 27B heretic Q4) answers each task k times; only answers whose headline (first line) matches the key are kept, and what is stored is the headline plus one justification sentence; a task a guess solves 20% of the time or more needs 2 right samples (k >= 2); checkpointed per task on the volume, `--resume` continues a timed-out run; yield per family in `.stats.json` | `results/phase6/rea-data/sft-<model>-seed<seed>-<family>.jsonl` | ~US$ 5 per 4,800 answers |
| 5 | (upload) | The SFT file went to the private dataset `HiddenTerps/local-forge-agent-sft` (`rea/sft_train.jsonl`, `rea/tasks_*.jsonl`) | Hugging Face | — |
| 6 | `modal_lora.py [--smoke]` | LoRA (rank 32, all attention/MLP projections) on the base safetensors, merge, convert and quantize **both** the trained and the base model to Q4_K_M by the same path | volume `lf-lora`, `results/phase6/lora/*.json` | ~US$ 2–3 (A100, 44 min) |
| 7 | `modal_rea.py --only heretic-9b-base-q4km,heretic-9b-rea-lora-q4km` | Before/after on the REA benchmark, both GGUFs from step 6 | `results/phase6/rea/` | ~US$ 1 |
| 8 | `modal_censor.py` | Refusals check on base and trained (the model must stay uncensored) | stdout, `results/phase6/censor.log` | < US$ 0.5 |
| 9 | `modal_hf_upload.py` | Upload merged weights + LoRA adapter, and the Q4_K_M GGUF, to private repos | Hugging Face | CPU only |

Run each with `modal run tools/modal/<script>.py [options]`; `--help` lists the options. Long runs are best started detached from the shell (e.g. `systemd-run --user --unit=modal-x ...`) so a dropped session does not kill them.

## REA task generator (`tools/rea-bench`)

Every task is a small C program whose function under question is named `proc`; the answer is computed by the generator from its own parameters (never by a model) and each family ships a harness that proves it. One command regenerates any family, from the repo root (stdlib only):

```bash
python3 tools/rea-bench list                                                  # families and variants
python3 tools/rea-bench gen --family crc --n 60 --seed 1001 --out /tmp/crc    # <id>.c + <id>.json per task; --family a,b,c or all
python3 tools/rea-bench check --family all --n 330 --seed 1001                # compile -O0..-O3/-Os -Wall -Werror (gcc, clang if installed), run the harnesses, check the asked constant is in the compiled code, report duplicates
python3 tools/rea-bench survival --family all --n 120                         # per variant: how often the asked constant is in the code, gcc and clang, every -O level
python3 tools/rea-bench stats --seed 7 --n 3000                               # distinct programs per family and overlap with the held-out list
python3 tools/rea-bench heldout [--write]                                     # verify / regenerate the pinned held-out list
python3 tools/rea-bench chance [--write]                                      # per-variant chance level (the `chance` field)
modal run tools/modal/modal_rea_data.py --stage decompile --family crc --n 60 --seed 1001
```

Without `--family`, `modal_rea_data.py` still generates the original 10 templates; `python3 tools/rea-bench --n 60 --seed 1001 --out D` (the old `gen_rea_tasks.py` command line) means `--family legacy`. The `legacy` output is frozen: a test pins its SHA-256 for seeds 1001 (held-out) and 7. `gen` refuses an `--out` that already holds task files unless `--force`. The fact gate needs gcc and objdump (clang too for `--cc mix|clang`) but looks for them only when a task with facts is drawn: `legacy` runs with no compiler at all, and on a machine without clang `--cc mix` gates the gcc draws, warns once and counts the clang draws it could not check (`--cc gcc` gates everything with gcc alone); `--no-survive` skips the gate.

**Held-out and training stay disjoint, as programs.** The held-out split is seed 1001. `tools/rea-bench/rea_bench/heldout-1001.txt` pins the keys of its first 120 tasks per family (the standard `--family all --n 330` split uses 30; a seed-1001 run above 120 per family is refused, since the list does not cover it). A key is the SHA-256 of the task's C body **without its decoy statements** (every decoy line ends with `// decoy` and is stripped), invented identifiers renamed (v0, v1, ...), and its question, so a renamed copy or the same program under other decoys counts as the same task. The decompile stage drops every generated task whose key is in the list or is any later draw the fact gate could have kept in a held-out slot (`--exclude-seed`, default 1001 for any other seed, 0 = off) and prints how many it dropped; `check` reports verbatim repeats inside a seed. Where a family has so few programs that this would drop a large share of training data, the held-out seed draws from reserved parameter values that training seeds never use: seven standard CRC algorithms, the ChaCha/Salsa round counts 10, 14, 18, ... 38, return codes and packet magics that are multiples of 5, and shuffled alphabets only (training also has the standard ones). Measured with the review's `overlap.py` on n=3000 per family at seeds 7 and 8 against the first 120 held-out tasks: **0 training tasks equal a held-out program, decoys stripped, and 0 held-out tasks have a training twin, for every new family** (before: encoding 60%, crc 32%, cipher 28%, antidebug 15% of the held-out tasks had a twin). The frozen `legacy` templates do repeat programs (about 19% of a seed-7 n=3000 set are one of the 120 held-out programs); the exclusion removes them from new training sets.

**Outputs are keyed by seed and family** and never overwritten without `--force`: `decompiled-seed<seed>-<family>.jsonl` (and `/decompiled-seed<seed>-<family>/` on the volume, wiped per run), `sft-<model>-seed<seed>-<family>.jsonl`; a family list joins with `+`, `--family all` is `all`. The sample stage takes the same `--seed` and `--family` as its decompile run. `modal_rea.py` reads every `decompiled-seed1001*.jsonl` for its held-out arm.

| family | variants | the question asks | the harness proves it with |
|---|---|---|---|
| `legacy` | 10 templates | the original ten (xor key, password, magic, LCG, hash, skip byte, key count, alphabet, rotate, threshold) | pinned hash |
| `structs` | 6 | field offset (hex) in a plain, nested or linked struct, which offset gets a constant, struct size (array stride, allocation) | `offsetof`/`sizeof` and `proc` on values it wrote |
| `fsm` | 6 | shortest input reaching a state, number of states (table or `switch` machine); opcode of an operation, number of opcodes (dispatcher) | every input up to the shortest length enumerated against a Python BFS |
| `crc` | 6 | polynomial as written + reflected/normal, initial value, final XOR; 16/32-bit, 10 standard algorithms or random polynomials, bitwise or runtime table | Python Rocksoft model (anchored to `zlib`, `binascii`, published check values) |
| `cipher` | 7 | RC4 + key length; TEA/XTEA + encrypt/decrypt + loop count; ChaCha/Salsa + total rounds | Python references (RC4 vector, RFC 7539, zero-key TEA/XTEA vectors) |
| `encoding` | 6 | base64/base58/base32 x encode/decode (standard or shuffled alphabet): which one, or the alphabet | stdlib `base64` + big-int base58, encode or round-trip |
| `strobf` | 7 | the plaintext written by a stack string, xor (constant, rolling, alternating, reversed), subtraction or rot-N | the C is run and its output compared |
| `pktsum` | 7 | where the checksummed range starts, bytes excluded at the end, byte order of the stored checksum, offset of the length byte (sum8, xor8, sum16, Fletcher-16, CRC-16) | valid and mutated packets against a Python model |
| `bounds` | 6 | lowest/highest accepted value (five spellings, unsigned window, clamp), table index limits, loop iteration count, count of accepted byte values | the whole domain scanned |
| `antidebug` | 6 | technique (`ptrace`, `timing`, `env`, `tracerpid`) + return value, env variable or threshold; ptrace via libc or raw syscall, `clock_gettime` or `gettimeofday` | libc call swapped for a fake: request code, syscall number, path, threshold and unit |
| `multi` | 5 | value that needs the helper's body (arithmetic, XOR key, decoded secret, two-hop chain, length limit) | evaluation in Python; `objdump` shows `proc` calls the `noinline` helper at every -O level |

Gated compiler and -O mix (seed 99, 60 tasks per variant, `survive.Gate`; design 8/25/33/17/17 % for -O0/-O1/-O2/-O3/-Os and 60 % gcc):

| family | -O0/-O1/-O2/-O3/-Os % | gcc % | | family | -O0/-O1/-O2/-O3/-Os % | gcc % |
|---|---|---|---|---|---|---|
| structs | 10/24/35/17/14 | 61 | | encoding | 11/27/35/14/12 | 59 |
| fsm | 9/20/33/19/19 | 59 | | strobf | 9/25/31/18/17 | 58 |
| crc | 8/25/32/19/16 | 62 | | pktsum | 9/25/34/20/11 | 65 |
| cipher | 7/30/30/17/15 | 57 | | bounds | 9/26/31/16/18 | 57 |
| antidebug | 9/23/33/18/17 | 60 | | multi | 12/22/29/20/18 | 59 |

Per family the mix is within a few points of the design. One variant is not: `crc/table_poly` keeps 83% gcc (design 60%), because clang folds or vectorises the table-building loop for most parameters and -O levels, so few clang draws keep the polynomial in the code. The other variants the review looked at (`crc/bitwise_poly`, `bounds/char_class`, `structs/init_magic`) are back near the design mix (gcc 64/58/61% against the ungated 59/61/61%; the review measured 68/38/61% after gating) now that a redraw keeps its flags.

Task JSON (new families): `id, kind, question, opt, cc, function, variant, score, accept` and, where needed, `reject`, `cand`, `helpers`, `facts` and `chance`. `cc` is `gcc` or `clang` (60/40 by design, measured below) and `opt` one of -O0..-O3/-Os; `decompile_task` compiles with `task.get('cc','gcc')` and the Modal image has clang. `helpers` names the functions `proc` calls; their pseudocode is appended after `proc`'s in the prompt. `facts` are groups of alternatives (numbers, or strings) that must be in the code compiled at the task's own cc/opt; the generator draws a task again when they are not (`survive.py`; a compiler can fold a loop count into a compare, a polynomial into a vector constant or a table, a character class into one bit mask), and `decompile_task` re-checks them with the Modal image's own compilers. A number counts as present as itself or as the 16/32/64-bit pattern of its negation, never as a truncation of itself (round 2: `0xa001` was "present" because of a `cmp rsi, 1`). A value below 0x20 is in most programs, so for those (loop and round counts, return codes, thresholds, the CRC polynomial and initial value) a group ends with a probe: the gate puts a marker value into the source at the asked place and requires the marker in the compiled code, which a folded constant fails; a CRC polynomial also fails when the object holds a 256-entry table the source does not. ChaCha/Salsa `proc` returns the number of rounds it ran, like TEA's returns its loop count. A task redrawn by the gate keeps its compiler and -O level for the first 8 redraws, so the gate does not skew them. `chance` is the estimated probability that a guess is right (`chance.py`): the share of tasks with the same question wording that have the same answer. Tasks whose answer is computed rather than read (fsm, strobf, multi, loop counts, "which base") carry no `facts`. Where a derived form counts (a range shown as a subtraction and a width, a loop bound one off, `>= N + 1`) the question says so. ARM64 and Windows PE are not generated yet.

Scoring (`tools/rea-bench/rea_bench/scoring.py`, the one scorer `modal_rea_data.py` and `modal_rea.py` import; an unknown mode raises before any GPU is rented): the original modes `exact`, `any`, `all`, `all_groups` are unchanged and still score the whole reply, so the 34/60 → 50/60 numbers stay comparable. The modes of the new families score the **headline only** and apply one rule to all of them:

- the headline is the first non-empty line, markdown removed; a bare lead-in line ("Answer:", "Candidates:") is skipped only when exactly one answer line follows before prose, so a list under a lead-in has no headline;
- `tokens` (numbers, names, directions): every group must match a whole token (numbers by value: `0x08` = `0x8`, `16u` = 16, `250,000,000` = 250000000; a bare hex byte such as `3f` or `\x3f` counts). The headline fails if it hedges or negates (`or`, `either`, `nor`, `not`, `n't`, `instead`, `rather than`, `vs`, `maybe`, `perhaps`, `possibly`, `probably`, `likely`, `might`, `could`), holds an expression ("40 + 40"), or holds a second candidate of the key's shape: another number, another name of the family's vocabulary (all names it can produce plus confusables such as XSalsa20, XChaCha20, Camellia, AES, base36, base45; any `baseNN`), another fsm input string or environment variable name. So "TEA (not XTEA)", "RC4 or Camellia", "not 64" and "0 1 2" fail. Numbers that belong to a unit or a standard name the question did not ask about are not candidates: "(128-bit key)", "(88 bits)", "CRC-32", "x86-64", "8-byte aligned". "Extended TEA" may explain XTEA; a bare "TEA" next to it may not. Lowest/highest must come in order;
- `exact_case` (alphabets) and `exact_head` (plaintexts, secrets): the whole headline must be the key. Quotes, backticks, markdown, trailing punctuation and a short lead-in made of function words ("The string is", "Alphabet:") are stripped, then nothing but the key may be left. Case and spaces are significant, and the key must be delimited: `xkey`, `key7`, `/etc/key`, `KEY`, `key or key2` are other strings.

What is stored (`scoring.sft_target`, `rea_bench/sft.py`) is not the reply but its headline (with its lead-in line, if any) and at most one justification sentence; the rest is cut off and the record says `truncated: true`. A sentence that names another candidate value, hedges or negates, or says wait / actually / correction / however / but vetoes the reply, so a right first line cannot carry a wrong or contradicting body into the data. A task whose chance level is 0.2 or more (CRC init and xorout 0.4-0.6, pktsum 0.2-0.37, fsm 0.17-0.27) is kept only when at least 2 of the k samples are right; that filters one lucky guess, it does not make a 0.5 task hard. The legacy tasks are cut the same way (the cut reply must still score) but keep their whole-reply scoring. A reply that answers in a later line, or restates another number on the answer line, is rejected: the cost is yield, not poisoned data. The review's attack scripts (candidate lists, `X or Y`, negations, supersets, a lead-in over a list, a right first line over a contradicting body) are in `tests/rea_attack_matrix.py` and run as tests.

The held-out arm of `modal_rea.py` is reported apart from the 10 fixed questions: a strict score (headline only) and a format-insensitive control (any line of the reply scores). A base model that thinks aloud on line 1 scores low on the strict one only; read a base-vs-LoRA gain on the strict score together with the control.

Tests: `python3 -m unittest tests.test_rea_bench tests.test_rea_scoring tests.test_rea_gate tests.test_rea_modal` (about 5 minutes in all; compiles and runs a sample of every family, checks the scorer against content-free replies and the review's attack shapes, mutates asked constants against the gate; the clang tests skip when clang is absent).

### Known limits of the generated data

- **The gate depends on the machine that generated the data.** Which draw a slot keeps depends on the gcc and clang versions that checked it (here gcc 15.2, clang 22.1); the Modal image has the apt compilers of its own base. A fact that fails there becomes an error record (it is not drawn again), so the held-out and training sets can shrink unevenly per family. The Modal-side re-check is the same code (`survive.check`, probes included); the error summary of the decompile stage says how many tasks it cost.
- **A first line that is not the answer is rejected by design** ("Looking at the code...", an answer on line 2 scores 0). That is right for data quality and costs yield; the yield of real teacher output has not been measured (no teacher run since this scorer). The measured pass rates are for synthetic realistic phrasings (28 formats, `tests/rea_attack_matrix.py`), not for sampled replies.
- **The gate is a necessary condition, not proof the decompiler shows the value.** It reads `objdump` of the compiled object, not Ghidra pseudocode; a constant a compiler writes as a shift or an lea is not accepted as a probed answer, but the pseudocode may still fold things the instructions show.
- **Small constants** (below 0x20: in every program) are not gated by presence; where a family asks for one (loop counts, return codes, thresholds) the fact carries a probe: the gate puts a marker value in the source at the asked place and requires it in the code. RC4's key length has no probe (it is the size of an array, not a literal) and is only checked by presence of values the gate treats as vacuous: the harness proves the key length, the gate does not prove it is visible.
- **The default family is still `legacy`**: its frozen whole-reply substring scoring accepts a content-free reply on about 6% of its tasks (`check` reports it, `--family all` avoids it). Stored targets are cut and re-scored like any other, but the scoring itself is unchanged on purpose.
- **A task a guess solves often stays easy.** Agreement of two samples removes lucky guesses, not a model that always answers "all ones" (CRC init, xorout).
- **`crc/table_poly` skews towards gcc**: clang vectorises or folds the table-building loop for most parameters and -O levels, so the gate keeps few clang draws of this variant (see the distribution below).
- **Held-out is disjoint as a program, for the reserved parameter values**: standard CRC algorithms, ChaCha/Salsa round counts, return codes, packet magics and standard alphabets are split between the held-out seed and training seeds (a held-out seed tests unseen standard algorithms, unseen round counts and shuffled alphabets only). The frozen `legacy` family still repeats programs.

## Rules the scripts keep

- Only uncensored candidates are chosen; censored models appear only as reference rows.
- Held-out tasks (seed 1001) and the 10 fixed questions never enter training.
- Teacher answers enter the data only when they match the automatic answer key.
- No outputs of OpenAI or Anthropic models in the data.
- GGUFs downloaded by SHA-256; llama.cpp image pinned by digest.

## Results so far

- Coding (40 problems, first try / 10 samples): CodeForge 27B heretic Q4 92% / 100%; 9B heretic 25% / 68%.
- REA, 9B heretic, base vs LoRA (same Q4_K_M path): held-out 34/60 → **50/60** (McNemar exact p = 0.0001), fixed questions 9/10 → 10/10, agent arm 4/10 → 5/10. Gains concentrated in LCG (0→6/6), magic constant (2→6/6) and rotate (2→6/6); hash (1/6) and threshold (0→2/6) still weak.
- Limits: 10 generated patterns, one function per binary, x86-64 Linux/gcc only. Not a general reverse-engineering model.

The training data itself (`results/phase6/rea-data/*.jsonl`) is not in the repo; it lives in the private dataset above.
