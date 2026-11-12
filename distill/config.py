"""One run's settings, and the sampling rules every run follows. Stdlib only.

`RunConfig` is a run as a JSON file (`distill/configs/*.json`): the subject, the teacher that is sampled, the student that is trained, how many samples, in which mode, with what sampling, how long a reply and a
training record may be. The rest of this file is what those settings mean, moved from `rea_bench/reasoning.py` unchanged: the sampling of each mode (Qwen3's thinking settings are never greedy), the token budget,
the slot of the teacher's server, the length windows the sample stage reports, and the `--quota` rule.

The three configs shipped for REA (`rea-plain.json`, `rea-reasoning.json`, `rea-thinking.json`) are today's runs: `tests/test_distill_config.py` checks that they resolve to the settings the flags of the Modal stage
give for the same mode.
"""
from __future__ import annotations

import collections
import dataclasses
import json
import pathlib

from .traces import DEFAULT_MAX_TRACES

MAX_LEN = 2048          # tokens of prompt + answer + EOS that one training example may have (the default; a longer window is a reasoning run's: records are several times longer)
DEFAULT_STUDENT = "xtools-at/DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic"      # the base model REA's LoRA is trained on
DEFAULT_TEACHER = "codeforge-27b-heretic-q4"
MODES = ("plain", "reasoning", "thinking")

DEFAULT_MAX_TOKENS = 768          # `--max-tokens` in reasoning mode (the phase-A sampler used 300 for a headline and one sentence)
REPORT_WINDOWS = (2048, 3072, 4096)     # the sample stage reports how many teacher records fit each of these training windows

# ---- the thinking run's settings (round 13) ----------------------------------------------------------------------------------------------------------
DEFAULT_SAMPLING = {"temperature": 0.7}          # every run that does not think: what phase A sampled with, unchanged
# Qwen3's recommended settings for thinking mode (model card: temperature 0.6, top_p 0.95, top_k 20, min_p 0; "do not use greedy decoding, it can lead to performance degradation and endless
# repetitions"). The order is the order of the request body.
THINKING_SAMPLING = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0}
DEFAULT_THINKING_MAX_TOKENS = 4096               # `--max-tokens` of a thinking run: the reasoning counts against it; a reply cut off at it is rejected and counted (`TRUNCATED`)
THINKING_REPORT_WINDOWS = (4096, 8192)           # the length report of a thinking run: how many records fit each of these training windows
THINKING_WINDOW = 8192                           # the recommended `--max-len` (sample stage and `modal_lora.py`) for thinking records
SLOT_CONTEXT = 8192                              # tokens of one slot of the sampler's llama-server (`-c` is slots x this); the phase-A and visible-reasoning runs have always used it
SLOT_PROMPT_ROOM = 4096                          # what the prompt may take next to a reply of `max_tokens`: the slot holds prompt + max_tokens (4096 + 4096 = 8192)
SLOT_ROUND = 1024                                # a slot's size is a multiple of this
SAMPLING_KEYS = tuple(THINKING_SAMPLING)


def slot_context(max_tokens: int) -> int:
    """Tokens per slot of the sampler's server: `SLOT_CONTEXT`, or more when `max_tokens` + `SLOT_PROMPT_ROOM` does not fit (rounded up to a multiple of 1024). A thinking run at the default 4096
    tokens needs exactly 8192: a prompt of up to 4096 tokens and a reply of up to 4096."""
    return max(SLOT_CONTEXT, -(-(max_tokens + SLOT_PROMPT_ROOM) // 1024) * 1024)


def server_plan(max_tokens: int, longest: int = 0, slots: int = 4) -> tuple:
    """(slots, tokens per slot) of the sampler's server. The base is `slots` slots of `slot_context(max_tokens)` tokens (4 x 8192 for every run so far). When a task of the run has a budget `longest`
    above `max_tokens` (a family of `RunConfig.max_tokens_by_family`), each slot grows by the difference (up to a multiple of 1024: 4096 -> 8192 tokens makes the 8192-token slot a 12288-token one) and the
    server keeps the SAME total context, so it holds fewer slots (4 x 8192 -> 2 x 12288): a longer reply costs requests in flight, not memory. Without such a task the base is returned unchanged."""
    ctx = slot_context(max_tokens)
    if longest > max_tokens:
        total = slots * ctx
        ctx = -(-(ctx + longest - max_tokens) // SLOT_ROUND) * SLOT_ROUND
        slots = max(1, total // ctx)
    return slots, ctx


def prompt_fits(prompt_tokens: int, max_tokens: int, slot_ctx: int) -> bool:
    """True when a prompt of `prompt_tokens` and a reply of up to `max_tokens` fit one slot of `slot_ctx` tokens: otherwise the server cuts the reply off early (or refuses the request)."""
    return prompt_tokens + max_tokens <= slot_ctx


def task_max_tokens(kind: str, max_tokens: int, by_family: dict = None) -> int:
    """The reply budget of a task of family `kind`: its entry of `by_family` (`RunConfig.max_tokens_by_family`), else the run's `max_tokens`."""
    return (by_family or {}).get(kind, max_tokens)


def run_max_tokens(kinds, max_tokens: int, by_family: dict = None) -> int:
    """The longest reply budget among the tasks of a run (`kinds`: the family of each task): what the server's slot must hold. Families of `by_family` that no task has do not count."""
    return max([task_max_tokens(k, max_tokens, by_family) for k in kinds] or [max_tokens])


def check_family_tokens(by_family, known=None) -> dict:
    """`by_family` as a clean dict {family: tokens}: every key a family name (one of `known`, when given: a typo must not silently run the default) and every value an integer of at least 1 (a bool is
    not one). SystemExit otherwise."""
    if not isinstance(by_family, dict):
        raise SystemExit(f"max_tokens_by_family {by_family!r}: a map family -> tokens, e.g. {{\"cipher\": 8192}}")
    for kind, n in by_family.items():
        if not isinstance(kind, str) or not kind:
            raise SystemExit(f"max_tokens_by_family: {kind!r} is not a family name")
        if not isinstance(n, int) or isinstance(n, bool) or n < 1:
            raise SystemExit(f"max_tokens_by_family {kind}={n!r}: a number of tokens, at least 1")
        if known is not None and kind not in known:
            raise SystemExit(f"max_tokens_by_family names {kind!r}, which is no task kind; known: {', '.join(known)}")
    return dict(by_family)


def parse_family_tokens(text: str, known=None) -> dict:
    """`cipher=8192,magic=8192` -> {"cipher": 8192, "magic": 8192} (the form of `--max-tokens-by-family`; "" is no override). SystemExit on a part that is not kind=N, a repeated kind, and what
    `check_family_tokens` refuses."""
    out: dict = {}
    for part in [p.strip() for p in text.split(",") if p.strip()]:
        kind, eq, count = part.partition("=")
        kind = kind.strip()
        if not eq or not kind or not (count.strip().isascii() and count.strip().isdigit()):        # isdigit() alone takes '²', which int() refuses: a traceback, not this message
            raise SystemExit(f"bad --max-tokens-by-family entry {part!r}: write kind=N with N a number of tokens, e.g. --max-tokens-by-family cipher=8192,magic=8192")
        if kind in out:
            raise SystemExit(f"--max-tokens-by-family names {kind!r} twice")
        out[kind] = int(count)
    return check_family_tokens(out, known)


def family_tokens_text(by_family: dict) -> str:
    """The flag form of a map (`parse_family_tokens` is its inverse), "" for none."""
    return ",".join(f"{k}={n}" for k, n in (by_family or {}).items())


def default_max_tokens(reasoning: bool, thinking: bool) -> int:
    """`--max-tokens` when it is not given: 300 (phase A), DEFAULT_MAX_TOKENS (768) with `--reasoning`, DEFAULT_THINKING_MAX_TOKENS (4096) with `--thinking`."""
    return DEFAULT_THINKING_MAX_TOKENS if thinking else DEFAULT_MAX_TOKENS if reasoning else 300


def sampling_for(thinking: bool, temperature=None, top_p=None, top_k=None, min_p=None) -> dict:
    """The sampling settings of a run, in request-body order: the defaults of its mode (`THINKING_SAMPLING` with `--thinking`, `DEFAULT_SAMPLING` otherwise) and each value that is not None replaced.
    SystemExit for a value out of range, and, in a thinking run, for greedy decoding (temperature 0, or top_k 1), which Qwen3 says loops and repeats; nothing is rented before that."""
    out = dict(THINKING_SAMPLING if thinking else DEFAULT_SAMPLING)
    for key, value in (("temperature", temperature), ("top_p", top_p), ("top_k", top_k), ("min_p", min_p)):
        if value is not None:
            out[key] = value
    t = out["temperature"]
    if not 0 <= t <= 2 or ("top_p" in out and not 0 < out["top_p"] <= 1) or ("top_k" in out and out["top_k"] < 0) or ("min_p" in out and not 0 <= out["min_p"] < 1):
        raise SystemExit(f"sampling out of range: {out} (temperature 0-2, top_p in (0, 1], top_k >= 0 (0 = off), min_p in [0, 1))")
    if thinking and (t <= 0 or out.get("top_k") == 1):
        raise SystemExit(f"a thinking run is never greedy ({out}): temperature 0 or top_k 1 makes the reasoning loop and repeat; Qwen3's recommendation is {THINKING_SAMPLING}")
    return out


def sampling_record(thinking: bool, sampling: dict, max_tokens: int, **more) -> dict:
    """The `sampling` block of the stats of a reasoning run: whether it thought, every sampling value, the token budget, and what else the caller knows (seed base, slot size...)."""
    return {"thinking": bool(thinking), **sampling, "max_tokens": max_tokens, **more}


def parse_quota(text: str, known=None) -> dict:
    """`crc=250,structs=100` -> {"crc": 250, "structs": 100} (order kept). SystemExit on an empty name, a count that is not a positive integer, a repeated kind, or (when `known` is
    given) a kind no task has."""
    out: dict = {}
    for part in [p.strip() for p in text.split(",") if p.strip()]:
        kind, eq, count = part.partition("=")
        kind = kind.strip()
        if not eq or not kind or not (count.strip().isascii() and count.strip().isdigit()) or int(count) < 1:        # ASCII digits only: isdigit() takes '²', which int() refuses
            raise SystemExit(f"bad --quota entry {part!r}: write kind=N with N a positive integer, e.g. --quota crc=250,structs=100")
        if kind in out:
            raise SystemExit(f"--quota names {kind!r} twice")
        if known is not None and kind not in known:
            raise SystemExit(f"--quota names {kind!r}, which is no task kind; known: {', '.join(known)}")
        out[kind] = int(count)
    if not out:
        raise SystemExit("--quota is empty: write kind=N[,kind=N...], or leave it out to sample every decompiled task")
    return out


def apply_quota(tasks: list, quota: dict) -> list:
    """At most `quota[kind]` tasks of each listed kind (the first ones, in the order `tasks` has: sorted by id); tasks of other kinds are not sampled. The result goes round-robin over
    the kinds (every kind's first task, then every kind's second, ...), so a run that is stopped early, or times out and is not resumed, holds a balanced set. No quota: `tasks`."""
    if not quota:
        return tasks
    by = collections.defaultdict(list)
    for t in tasks:
        if t.get("kind") in quota:
            by[t["kind"]].append(t)
    chosen = {kind: by[kind][:n] for kind, n in quota.items()}
    out = []
    for rank in range(max((len(v) for v in chosen.values()), default=0)):
        out += [chosen[kind][rank] for kind in quota if rank < len(chosen[kind])]
    return out


# ---- the run as a file ---------------------------------------------------------------------------------------------------------------------

@dataclasses.dataclass
class RunConfig:
    """A run. `sampling` / `max_tokens` / `max_len` left out (None / 0) mean the default of the mode (`sampling_for`, `default_max_tokens`, `MAX_LEN`); `max_len` of a reasoning or thinking run has no default
    (the stage refuses it: a reasoning record is several times longer than a truth record, and what survives depends on the window). `mix` is a training choice: a thinking file's truth-template records are trained too.
    `max_tokens_by_family`: {family: tokens}, the reply budget of the tasks of a family whose reasoning does not fit `max_tokens` (the thinking pilot cut off every cipher reply at 4096); empty (the default): `max_tokens`
    for every task. The names are the tasks' `kind`; a family no task of the run has changes nothing."""
    domain: str = "rea"
    teacher: str = DEFAULT_TEACHER
    student: str = DEFAULT_STUDENT
    k: int = 6
    mode: str = "plain"
    sampling: dict | None = None
    max_tokens: int = 0
    max_traces: int = DEFAULT_MAX_TRACES
    max_len: int = 0
    mix: bool = False
    quota: str = ""
    max_tokens_by_family: dict = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        if self.mode not in MODES:
            raise SystemExit(f"run config: mode {self.mode!r} is not one of {', '.join(MODES)}")
        if not isinstance(self.k, int) or self.k < 1:
            raise SystemExit(f"run config: k {self.k!r}: at least one sample per task")
        if self.max_tokens < 0 or self.max_len < 0 or self.max_traces < 1:
            raise SystemExit(f"run config: max_tokens {self.max_tokens}, max_len {self.max_len}, max_traces {self.max_traces}: counts, and at least one trace")
        if self.sampling is not None and not set(self.sampling) <= set(SAMPLING_KEYS):
            raise SystemExit(f"run config: sampling holds {sorted(self.sampling)}; only {', '.join(SAMPLING_KEYS)} can be set")
        sampling_for(self.thinking, **(self.sampling or {}))          # out of range, or greedy in a thinking run: refused here, before anything is rented
        self.max_tokens_by_family = check_family_tokens(self.max_tokens_by_family)
        if self.max_tokens_by_family and not self.reasoning:
            raise SystemExit("run config: max_tokens_by_family is for a reasoning or thinking run (a plain run's reply is one short line)")

    @property
    def reasoning(self) -> bool:
        return self.mode != "plain"

    @property
    def thinking(self) -> bool:
        return self.mode == "thinking"

    @classmethod
    def load(cls, path) -> "RunConfig":
        """The config in a JSON file; a key that is not a field is refused (a typo must not silently run the defaults)."""
        p = pathlib.Path(path)
        try:
            raw = json.loads(p.read_text())
            assert isinstance(raw, dict)
        except (OSError, json.JSONDecodeError, AssertionError):
            raise SystemExit(f"{p} is not a run config: a JSON object with the fields of distill.config.RunConfig")
        fields = {f.name for f in dataclasses.fields(cls)}
        if set(raw) - fields:
            raise SystemExit(f"{p}: unknown key(s) {sorted(set(raw) - fields)}; fields: {sorted(fields)}")
        return cls(**raw)

    def dump(self) -> str:
        return json.dumps(dataclasses.asdict(self), indent=1) + "\n"

    def longest_budget(self) -> int:
        """The longest reply budget of a task of this run: `max_tokens`, or a family's own when the run has a task of that family (the families of the quota; without a quota, every family)."""
        tokens = self.max_tokens or default_max_tokens(self.reasoning, self.thinking)
        if self.quota:
            return run_max_tokens(list(parse_quota(self.quota)), tokens, self.max_tokens_by_family)
        return max([tokens, *self.max_tokens_by_family.values()])

    def resolved(self) -> dict:
        """What the run will use, with the mode's defaults filled in. `slot_context` and `slots` are for the longest budget of a task of the run (`longest_budget`, `server_plan`)."""
        tokens = self.max_tokens or default_max_tokens(self.reasoning, self.thinking)
        longest = self.longest_budget()
        return {"mode": self.mode, "sampling": sampling_for(self.thinking, **(self.sampling or {})), "max_tokens": tokens,
                "max_len": self.max_len or MAX_LEN, "slot_context": server_plan(tokens, longest)[1] if self.reasoning else SLOT_CONTEXT,
                "report_windows": (THINKING_REPORT_WINDOWS if self.thinking else REPORT_WINDOWS) if self.reasoning else (),
                **({"max_tokens_by_family": dict(self.max_tokens_by_family), "slots": server_plan(tokens, longest)[0]} if self.max_tokens_by_family else {})}

    def sample_kwargs(self) -> dict:
        """The keyword arguments of the sample stage (`modal_sample.sample_answers`) for this run."""
        return {"reasoning": self.reasoning, "thinking": self.thinking, "max_tokens": self.max_tokens, "max_len": self.max_len, "quota": self.quota, "max_traces": self.max_traces,
                "sampling": self.sampling or None, **({"max_tokens_by_family": dict(self.max_tokens_by_family)} if self.max_tokens_by_family else {})}


# ---- a run config in place of flags -----------------------------------------------------------------------------------------------------------------

FLAG_DEFAULTS = {"model": "", "k": 6, "reasoning": False, "thinking": False, "max_tokens": 0, "max_len": 0, "quota": "", "max_traces": DEFAULT_MAX_TRACES,
                 "temperature": -1.0, "top_p": -1.0, "top_k": -1, "min_p": -1.0, "max_tokens_by_family": ""}      # the defaults of the sample stage's flags (a negative sampling value is "not given"); a test pins them to the wrapper's
TRAIN_FLAG_DEFAULTS = {"student": DEFAULT_STUDENT, "max_len": MAX_LEN, "mix": False}      # the same for the training stage


def merge_flags(path, flags: dict, domain: str) -> dict:
    """The sample stage's flags of a run config file (`--run-config`): the file sets every flag of FLAG_DEFAULTS and none may also be given on the command line (SystemExit: two sources of one setting is
    a mistake, and a flag left at its default cannot be told from one that was typed). The file's `domain` must be the stage's."""
    cfg = RunConfig.load(path)
    if cfg.domain != domain:
        raise SystemExit(f"--run-config {path} is for domain {cfg.domain!r}, this stage is {domain!r}'s")
    clash = sorted(k for k, v in flags.items() if v != FLAG_DEFAULTS[k])
    if clash:
        raise SystemExit(f"--run-config {path} sets the run: do not also give " + ", ".join("--" + k.replace("_", "-") for k in clash))
    sampling = cfg.sampling or {}
    return {"model": cfg.teacher, "k": cfg.k, "reasoning": cfg.reasoning, "thinking": cfg.thinking, "max_tokens": cfg.max_tokens, "max_len": cfg.max_len, "quota": cfg.quota,
            "max_traces": cfg.max_traces, **{key: sampling.get(key, FLAG_DEFAULTS[key]) for key in SAMPLING_KEYS}, "max_tokens_by_family": family_tokens_text(cfg.max_tokens_by_family)}


def merge_train_flags(path, flags: dict) -> dict:
    """The training stage's flags (student, max_len, mix) of a run config file, with the same rule as `merge_flags`. `max_len` 0 in the file is the default window."""
    cfg = RunConfig.load(path)
    clash = sorted(k for k, v in flags.items() if v != TRAIN_FLAG_DEFAULTS[k])
    if clash:
        raise SystemExit(f"--run-config {path} sets the run: do not also give " + ", ".join("--" + k.replace("_", "-") for k in clash))
    return {"student": cfg.student, "max_len": cfg.max_len or MAX_LEN, "mix": cfg.mix}
