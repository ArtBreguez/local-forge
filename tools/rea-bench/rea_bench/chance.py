"""Chance level per variant: the estimated probability that a guess is right, written into every task's JSON as `chance`.

Estimate: draw 1000 tasks per variant of a family, group them by the wording of their question (instance numbers masked, so "called
with 17" and "called with 40" are one group, "16-bit" stays), and score a guess drawn from the generator's own answer distribution
for that wording: the chance on one task is how often its answer occurs among the OTHER tasks of its group (a joint answer such as
cipher name + direction + rounds counts as one). The chance of a variant is the mean over its tasks. A guesser cannot see the
variant label, which is why "which base, encode or decode" comes out near 1/6, not 1. Input strings of the fsm are different: a
guess is a string over the machine's alphabet, so the chance is |accepted| / alphabet^length. Values are floored at 1/1000.
Round 3 (m3): the estimate above is the rate of a guesser that SAMPLES the answer distribution. A model that reads the family off the code
and only guesses the number does better: it answers the variant's most common answer every time (XTEA decrypt: 32 rounds, right on 32% of
the tasks; the sampling estimate said 4.5%). So the chance of a variant is the larger of the two: the best constant answer within the
variant. It is a baseline for reading yields, not a guarantee: a task with a high chance keeps lucky guesses under rejection sampling,
which is what the agreement rule of sft.py (AGREE_CHANCE) is for.

    python3 tools/rea-bench chance            print the estimates
    python3 tools/rea-bench chance --write    rewrite the table below (after a generator change)
"""
from __future__ import annotations

import pathlib
import re
from collections import Counter

# <table> (generated, do not edit)
CHANCE: dict = {
    "antidebug/env": 0.015,
    "antidebug/ptrace": 0.011,
    "antidebug/ptrace_raw": 0.01,
    "antidebug/timing_ns": 0.003,
    "antidebug/timing_us": 0.003,
    "antidebug/tracerpid": 0.011,
    "bounds/char_class": 0.062,
    "bounds/clamp": 0.001,
    "bounds/index_limit": 0.037,
    "bounds/loop_iterations": 0.036,
    "bounds/range_chain": 0.001,
    "bounds/window": 0.002,
    "cipher/chacha": 0.198,
    "cipher/rc4": 0.101,
    "cipher/salsa": 0.203,
    "cipher/tea_dec": 0.277,
    "cipher/tea_enc": 0.305,
    "cipher/xtea_dec": 0.278,
    "cipher/xtea_enc": 0.274,
    "crc/bitwise_init": 0.571,
    "crc/bitwise_poly": 0.081,
    "crc/bitwise_xorout": 0.447,
    "crc/table_init": 0.5616,
    "crc/table_poly": 0.073,
    "crc/table_xorout": 0.451,
    "encoding/base32_decode": 0.705,
    "encoding/base32_encode": 0.676,
    "encoding/base58_decode": 0.686,
    "encoding/base58_encode": 0.717,
    "encoding/base64_decode": 0.682,
    "encoding/base64_encode": 0.705,
    "fsm/opcode_count": 0.264,
    "fsm/opcode_value": 0.06,
    "fsm/switch_input": 0.1763,
    "fsm/switch_states": 0.358,
    "fsm/table_input": 0.1681,
    "fsm/table_states": 0.369,
    "multi/chain": 0.006,
    "multi/decode_secret": 0.002,
    "multi/eval_arith": 0.012,
    "multi/key_helper": 0.01,
    "multi/limit_helper": 0.063,
    "pktsum/crc16": 0.3687,
    "pktsum/fletcher16": 0.3638,
    "pktsum/len_field": 0.2554,
    "pktsum/sum16_be": 0.499,
    "pktsum/sum16_le": 0.488,
    "pktsum/sum8": 0.306,
    "pktsum/xor8": 0.293,
    "strobf/alt_xor": 0.003,
    "strobf/reverse_xor": 0.003,
    "strobf/rot_n": 0.002,
    "strobf/stack_string": 0.003,
    "strobf/sub_const": 0.003,
    "strobf/xor_const": 0.003,
    "strobf/xor_rolling": 0.003,
    "structs/alloc_size": 0.219,
    "structs/array_stride": 0.211,
    "structs/field_read": 0.137,
    "structs/init_magic": 0.125,
    "structs/linked_next": 0.399,
    "structs/nested_offset": 0.187,
}
# </table>

PER_VARIANT = 1000


def _group(t) -> str:
    return re.sub(r"0x[0-9a-f]+|\b\d+\b(?!-)", "#", t.question)


def _answer(t) -> tuple:
    acc = t.truth["accept"]
    return tuple(g[0] for g in acc) if t.truth["score"] in ("tokens", "all_groups") else tuple(acc)


def estimate(seed: int = 424242, n: int = PER_VARIANT) -> dict:
    from . import families
    out = {}
    for fam, mod in families.modules().items():
        if fam == "legacy":
            continue
        tasks = families.generate(fam, n * len(mod.VARIANTS), seed)
        groups: dict = {}
        for t in tasks:
            groups.setdefault(_group(t), Counter())[_answer(t)] += 1
        for variant in mod.VARIANTS:
            ts = [t for t in tasks if t.variant == variant]
            if ts[0].truth.get("cand") and "d" in ts[0].params and "alphabet" in ts[0].params:      # fsm input strings
                p = sum(len(t.truth["accept"][0]) / len(t.params["alphabet"]) ** t.params["d"] for t in ts) / len(ts)
            else:
                p = sum((groups[_group(t)][_answer(t)] - 1) / max(sum(groups[_group(t)].values()) - 1, 1) for t in ts) / len(ts)
            best_constant = Counter(_answer(t) for t in ts).most_common(1)[0][1] / len(ts)
            out[f"{fam}/{variant}"] = round(max(p, best_constant, 1 / len(ts)), 4)
    return out


def write_table(table: dict, path: pathlib.Path = pathlib.Path(__file__)) -> None:
    body = "CHANCE: dict = {\n" + "".join(f'    "{k}": {v},\n' for k, v in sorted(table.items())) + "}\n"
    s = path.read_text()
    a, b = s.index("# <table>"), s.index("# </table>")
    path.write_text(s[:a] + "# <table> (generated, do not edit)\n" + body + s[b:])
