"""Chance level per variant: the estimated probability that a guess is right, written into every task's JSON as `chance`.

Estimate: draw 1000 tasks per variant of a family, group them by the wording of their question (instance numbers masked, so "called
with 17" and "called with 40" are one group, "16-bit" stays), and score a guess drawn from the generator's own answer distribution
for that wording: the chance on one task is how often its answer occurs among the OTHER tasks of its group (a joint answer such as
cipher name + direction + rounds counts as one). The chance of a variant is the mean over its tasks. A guesser cannot see the
variant label, which is why "which base, encode or decode" comes out near 1/6, not 1. Input strings of the fsm are different: a
guess is a string over the machine's alphabet, so the chance is |accepted| / alphabet^length. Values are floored at 1/1000.
It is a baseline for reading yields, not a guarantee: a task with a high chance keeps lucky guesses under rejection sampling.

    python3 tools/rea-bench chance            print the estimates
    python3 tools/rea-bench chance --write    rewrite the table below (after a generator change)
"""
from __future__ import annotations

import pathlib
import re
from collections import Counter

# <table> (generated, do not edit)
CHANCE: dict = {
    "antidebug/env": 0.0038,
    "antidebug/ptrace": 0.004,
    "antidebug/ptrace_raw": 0.004,
    "antidebug/timing_ns": 0.001,
    "antidebug/timing_us": 0.001,
    "antidebug/tracerpid": 0.0039,
    "bounds/char_class": 0.0403,
    "bounds/clamp": 0.001,
    "bounds/index_limit": 0.0107,
    "bounds/loop_iterations": 0.0188,
    "bounds/range_chain": 0.001,
    "bounds/window": 0.001,
    "cipher/chacha": 0.0314,
    "cipher/rc4": 0.0833,
    "cipher/salsa": 0.0304,
    "cipher/tea_dec": 0.0452,
    "cipher/tea_enc": 0.0473,
    "cipher/xtea_dec": 0.0452,
    "cipher/xtea_enc": 0.045,
    "crc/bitwise_init": 0.5496,
    "crc/bitwise_poly": 0.0519,
    "crc/bitwise_xorout": 0.4885,
    "crc/table_init": 0.5482,
    "crc/table_poly": 0.0467,
    "crc/table_xorout": 0.4913,
    "encoding/base32_decode": 0.119,
    "encoding/base32_encode": 0.1094,
    "encoding/base58_decode": 0.1127,
    "encoding/base58_encode": 0.1231,
    "encoding/base64_decode": 0.1114,
    "encoding/base64_encode": 0.119,
    "fsm/opcode_count": 0.2501,
    "fsm/opcode_value": 0.0425,
    "fsm/switch_input": 0.1763,
    "fsm/switch_states": 0.2677,
    "fsm/table_input": 0.1681,
    "fsm/table_states": 0.2683,
    "multi/chain": 0.0012,
    "multi/decode_secret": 0.001,
    "multi/eval_arith": 0.0029,
    "multi/key_helper": 0.0039,
    "multi/limit_helper": 0.0404,
    "pktsum/crc16": 0.3691,
    "pktsum/fletcher16": 0.3595,
    "pktsum/len_field": 0.2573,
    "pktsum/sum16_be": 0.3669,
    "pktsum/sum16_le": 0.353,
    "pktsum/sum8": 0.2271,
    "pktsum/xor8": 0.2245,
    "strobf/alt_xor": 0.001,
    "strobf/reverse_xor": 0.001,
    "strobf/rot_n": 0.001,
    "strobf/stack_string": 0.001,
    "strobf/sub_const": 0.001,
    "strobf/xor_const": 0.001,
    "strobf/xor_rolling": 0.001,
    "structs/alloc_size": 0.1392,
    "structs/array_stride": 0.1349,
    "structs/field_read": 0.0766,
    "structs/init_magic": 0.0708,
    "structs/linked_next": 0.2907,
    "structs/nested_offset": 0.1209,
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
            out[f"{fam}/{variant}"] = round(max(p, 1 / len(ts)), 4)
    return out


def write_table(table: dict, path: pathlib.Path = pathlib.Path(__file__)) -> None:
    body = "CHANCE: dict = {\n" + "".join(f'    "{k}": {v},\n' for k, v in sorted(table.items())) + "}\n"
    s = path.read_text()
    a, b = s.index("# <table>"), s.index("# </table>")
    path.write_text(s[:a] + "# <table> (generated, do not edit)\n" + body + s[b:])
