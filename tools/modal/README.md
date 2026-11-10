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
| 2 | `modal_rea.py` | REA benchmark: 10 fixed questions on one stripped binary (`rea-bench/targets.c`), recipe arm (Ghidra pseudocode given) and agent arm (`funcs`, `decomp`, `decomp-all`, `refdata` tools, 12 steps), plus the 60 generated held-out tasks (seed 1001) | `results/phase6/rea/` | ~US$ 1 per model |
| 3 | `modal_rea_data.py --stage decompile [--family legacy\|all\|crc,cipher,...] [--cc mix\|gcc\|clang] [--exclude-seed 1001] [--force]` | Generate tasks with `tools/rea-bench` (11 families, automatic answer key; default `legacy` = the original 10 templates), drop every task that is in the held-out list, compile stripped with each task's own compiler and -O level, re-check that the asked constant is in that binary, decompile `proc` (and the helpers it calls) with Ghidra, resolve referenced bytes | `results/phase6/rea-data/decompiled-seed<seed>-<family>.jsonl` | ~US$ 1 per 1,200 |
| 4 | `modal_rea_data.py --stage sample --model <teacher> --k 4 [--seed 7 --family legacy] [--force]` | The teacher (uncensored, e.g. CodeForge 27B heretic Q4) answers each task k times; only answers whose headline (first line) matches the key are kept | `results/phase6/rea-data/sft-<model>-seed<seed>-<family>.jsonl` | ~US$ 5 per 4,800 answers |
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

Without `--family`, `modal_rea_data.py` still generates the original 10 templates; `python3 tools/rea-bench --n 60 --seed 1001 --out D` (the old `gen_rea_tasks.py` command line) means `--family legacy`. The `legacy` output is frozen: a test pins its SHA-256 for seeds 1001 (held-out) and 7. `gen` refuses an `--out` that already holds task files unless `--force`, and needs gcc and objdump (clang too for `--cc mix|clang`) for the fact gate; `--no-survive` skips the gate.

**Held-out and training stay disjoint.** The held-out split is seed 1001. `tools/rea-bench/rea_bench/heldout-1001.txt` pins the keys of its first 120 tasks per family (the standard `--family all --n 330` split uses 30); a key is the SHA-256 of the task's C body with invented identifiers renamed (v0, v1, ...) and its question, so a renamed copy counts as the same program. The decompile stage drops every generated task whose key is in the list (`--exclude-seed`, default 1001 for any other seed, 0 = off) and prints how many it dropped; `check` reports duplicates inside a seed. The families draw their constants from wide ranges and add decoy statements (volatile locals, static counters), so a new family's seed-7 and seed-1001 sets share no program (measured on n=3000: 0 overlap, 2973..3000 distinct bodies). The frozen `legacy` templates do repeat programs (about 19% of a seed-7 n=3000 set are one of the 120 held-out programs); the exclusion removes them from new training sets.

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

Task JSON (new families): `id, kind, question, opt, cc, function, variant, score, accept` and, where needed, `reject`, `cand`, `helpers`, `facts` and `chance`. `cc` is `gcc` or `clang` (about 60/40) and `opt` one of -O0..-O3/-Os; `decompile_task` compiles with `task.get('cc','gcc')` and the Modal image has clang. `helpers` names the functions `proc` calls; their pseudocode is appended after `proc`'s in the prompt. `facts` are groups of alternatives (numbers, or strings) that must be in the code compiled at the task's own cc/opt; the generator draws a task again when they are not (`survive.py`; a compiler can fold a loop count into a compare, a polynomial into a vector constant, a character class into one bit mask), and `decompile_task` re-checks them with the Modal image's own compilers. `chance` is the estimated probability that a guess is right (`chance.py`): the share of tasks with the same question wording that have the same answer. Tasks whose answer is computed rather than read (fsm, strobf, multi, loop counts, "which base") carry no `facts`. Where a derived form counts (a range shown as a subtraction and a width, a loop bound one off, `>= N + 1`) the question says so. ARM64 and Windows PE are not generated yet.

Scoring (`tools/rea-bench/rea_bench/scoring.py`, the one scorer `modal_rea_data.py` and `modal_rea.py` import; an unknown mode raises before any GPU is rented): the original modes `exact`, `any`, `all`, `all_groups` are unchanged and still score the whole reply, so the 34/60 → 50/60 numbers stay comparable. The modes of the new families score the **headline only** (first non-empty line, markdown removed, a bare "Answer:" skipped), because the sampler's prompt puts the answer there and a justification after it: `tokens` (every group must match a whole token; numbers by value, so `0x08` = `0x8`, `16u` = 16, `250,000,000` = 250000000; a `reject` token or a number the key does not contain fails it, so "XTEA" is not "TEA", "0x18" is not 8 and "0 1 2" is no answer to anything; lowest/highest must come in order), `exact_case` (alphabets) and `exact_head` (plaintexts, secrets). A sampled reply is kept for SFT only when its headline matched. A reply that answers in a later line, or restates another number on the answer line, is rejected: the cost is yield, not poisoned data.

Tests: `python3 -m unittest tests.test_rea_bench` (about 2.5 minutes; compiles and runs a sample of every family and checks the scorer against content-free replies; the clang tests skip when clang is absent).

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
