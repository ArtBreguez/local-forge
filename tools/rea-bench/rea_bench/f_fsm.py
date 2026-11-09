"""State machines and jump tables.

A random machine (S states, A input letters, the last state is an absorbing accept state) is written either as a transition table
or as a `switch (state)` that compilers turn into a jump table. The Python model supplies the truth: the shortest strings that
end in the accept state (BFS, all of them are accepted) and the number of states (all reachable). The harness enumerates every
input up to that length and prints which ones the C accepts, so "shortest" and "all" are both proven. A second form is an
opcode dispatcher (`switch (op)` over a sparse opcode set): which opcode computes a ^ b, how many opcodes exist.
"""
from __future__ import annotations

import itertools

from .core import Task, dec

FAMILY = "fsm"
LETTERS = "bcdfghjklmnpqrstvwxz"   # consonants only: a random accepted string never collides with an English word in an answer


def run(delta, alphabet, s):
    st = 0
    for ch in s:
        st = delta[st][alphabet.index(ch)]
    return st


def distances(delta):
    dist, frontier = {0: 0}, [0]
    while frontier:
        nxt = []
        for s in frontier:
            for t in delta[s]:
                if t not in dist:
                    dist[t] = dist[s] + 1
                    nxt.append(t)
        frontier = nxt
    return dist


def make_machine(rng):
    """(alphabet, delta, shortest accepting length d). Accept state = last; built around a planted path so d is 3..5."""
    for _ in range(2000):
        s = rng.randint(5, 8)
        a = rng.randint(2, 4)
        alphabet = rng.sample(LETTERS, a)
        acc = s - 1
        d = rng.randint(3, min(5, s - 1))
        mids = rng.sample(range(1, acc), d - 1)
        delta = [[rng.randrange(s) for _ in range(a)] for _ in range(s)]
        path = [0] + mids + [acc]
        for i in range(d):
            delta[path[i]][rng.randrange(a)] = path[i + 1]
        delta[acc] = [acc] * a
        dist = distances(delta)
        if len(dist) == s and dist[acc] == d:
            return alphabet, delta, d
    raise RuntimeError("no machine found")


def shortest_inputs(delta, alphabet, d):
    acc = len(delta) - 1
    return ["".join(p) for p in itertools.product(alphabet, repeat=d) if run(delta, alphabet, p) == acc]


def table_code(alphabet, delta):
    s, a = len(delta), len(alphabet)
    rows = ",\n".join("        {" + ", ".join(map(str, r)) + "}" for r in delta)
    return f"""int proc(const char *s) {{
    static const char alpha[] = "{''.join(alphabet)}";
    static const uint8_t T[{s}][{a}] = {{
{rows}
    }};
    unsigned st = 0;
    for (; *s; s++) {{
        const char *p = strchr(alpha, *s);
        if (!p) return -1;
        st = T[st][p - alpha];
    }}
    return (int)st;
}}"""


def switch_code(alphabet, delta):
    cases = []
    for st, row in enumerate(delta):
        chain = []
        for i, (ch, tgt) in enumerate(zip(alphabet, row)):
            chain.append(f"if (c == '{ch}') st = {tgt};" if i == 0 else f"else if (c == '{ch}') st = {tgt};")
        cases.append(f"        case {st}: " + " ".join(chain) + " else return -1; break;")
    return f"""int proc(const char *s) {{
    unsigned st = 0;
    for (; *s; s++) {{
        char c = *s;
        switch (st) {{
{chr(10).join(cases)}
        default: return -1;
        }}
    }}
    return (int)st;
}}"""


def input_harness(alphabet, d, acc):
    return f"""int main(void) {{
    const char *al = "{''.join(alphabet)}";
    unsigned A = {len(alphabet)};
    for (unsigned len = 1; len <= {d}; len++) {{
        unsigned total = 1;
        for (unsigned i = 0; i < len; i++) total *= A;
        for (unsigned x = 0; x < total; x++) {{
            char buf[16];
            unsigned y = x;
            for (int i = (int)len - 1; i >= 0; i--) {{ buf[i] = al[y % A]; y /= A; }}
            buf[len] = 0;
            if (proc(buf) == {acc}) puts(buf);
        }}
    }}
    return 0;
}}"""


def states_harness(alphabet, s):
    return f"""int main(void) {{
    const char *al = "{''.join(alphabet)}";
    unsigned A = {len(alphabet)};
    int seen[64] = {{0}};
    for (unsigned len = 0; len <= {s - 1}; len++) {{
        unsigned total = 1;
        for (unsigned i = 0; i < len; i++) total *= A;
        for (unsigned x = 0; x < total; x++) {{
            char buf[16];
            unsigned y = x;
            for (int i = (int)len - 1; i >= 0; i--) {{ buf[i] = al[y % A]; y /= A; }}
            buf[len] = 0;
            int r = proc(buf);
            if (r >= 0 && r < 64) seen[r] = 1;
        }}
    }}
    for (int i = 0; i < 64; i++) if (seen[i]) printf("%d\\n", i);
    return 0;
}}"""


def _machine_task(rng, variant, code_fn, ask):
    alphabet, delta, d = make_machine(rng)
    s, acc = len(delta), len(delta) - 1
    body = code_fn(alphabet, delta)
    if ask == "input":
        good = shortest_inputs(delta, alphabet, d)
        q = f"This state machine starts in state 0 and proc returns the state it ends in. Give a shortest input string (letters only) for which proc returns {acc}."
        bad = next("".join(p) for p in itertools.product(alphabet, repeat=d) if "".join(p) not in good)
        main = input_harness(alphabet, d, acc)
        return Task(kind=FAMILY, variant=variant, body=body, question=q, truth={"score": "tokens", "accept": [good]},
                    params={"delta": delta, "alphabet": alphabet, "d": d}, wrong=bad, check=(main, "".join(g + "\n" for g in good)))
    q = "This state machine starts in state 0 and proc returns the state it ends in. How many distinct states does it have (decimal)?"
    return Task(kind=FAMILY, variant=variant, body=body, question=q, truth={"score": "tokens", "accept": [dec(s)]},
               params={"delta": delta, "alphabet": alphabet}, wrong=str(s + 1),
               check=(states_harness(alphabet, s), "".join(f"{i}\n" for i in sorted(distances(delta)))))


def v_table_input(rng):
    return _machine_task(rng, "table_input", table_code, "input")


def v_switch_input(rng):
    return _machine_task(rng, "switch_input", switch_code, "input")


def v_table_states(rng):
    return _machine_task(rng, "table_states", table_code, "states")


def v_switch_states(rng):
    return _machine_task(rng, "switch_states", switch_code, "states")


MASK = 0xffffffff
OPS = {   # name: (C expression, python function, how the question writes it)
    "add": ("a + b", lambda a, b: (a + b) & MASK, "a + b"),
    "sub": ("a - b", lambda a, b: (a - b) & MASK, "a - b"),
    "mul": ("a * b", lambda a, b: (a * b) & MASK, "a * b"),
    "xor": ("a ^ b", lambda a, b: a ^ b, "a ^ b"),
    "and": ("a & b", lambda a, b: a & b, "a & b"),
    "or": ("a | b", lambda a, b: a | b, "a | b"),
    "shl": ("a << (b & 31)", lambda a, b: (a << (b & 31)) & MASK, "a << (b & 31)"),
    "shr": ("a >> (b & 31)", lambda a, b: a >> (b & 31), "a >> (b & 31)"),
}
HA, HB = 0x1234abcd, 7


def _dispatch(rng):
    names = rng.sample(list(OPS), rng.randint(5, 8))
    codes = rng.sample(range(0, 24), len(names))
    pairs = sorted(zip(codes, names))
    cases = "\n".join(f"    case {c}: return {OPS[n][0]};" for c, n in pairs)
    body = f"""uint32_t proc(uint32_t op, uint32_t a, uint32_t b) {{
    switch (op) {{
{cases}
    default: return 0;
    }}
}}"""
    main = f"""int main(void) {{
    for (uint32_t op = 0; op < 32; op++) printf("%u %08x\\n", op, proc(op, 0x{HA:x}u, {HB}u));
    return 0;
}}"""
    table = dict(pairs)
    expect = "".join(f"{op} {OPS[table[op]][1](HA, HB) if op in table else 0:08x}\n" for op in range(32))
    return pairs, body, main, expect


def v_opcode_value(rng):
    pairs, body, main, expect = _dispatch(rng)
    code, name = rng.choice(pairs)
    q = f"proc is an opcode dispatcher (a switch on its first argument). Which opcode value (decimal) makes it return {OPS[name][2]}?"
    return Task(kind=FAMILY, variant="opcode_value", body=body, question=q, truth={"score": "tokens", "accept": [dec(code)]},
                params={"pairs": pairs, "op": name}, wrong=str(code + 1), check=(main, expect))


def v_opcode_count(rng):
    pairs, body, main, expect = _dispatch(rng)
    q = "proc is an opcode dispatcher (a switch on its first argument). How many different opcodes does it handle, not counting the default (decimal)?"
    return Task(kind=FAMILY, variant="opcode_count", body=body, question=q, truth={"score": "tokens", "accept": [dec(len(pairs))]},
                params={"pairs": pairs}, wrong=str(len(pairs) + 1), check=(main, expect))


VARIANTS = {"table_input": v_table_input, "switch_input": v_switch_input, "table_states": v_table_states,
            "switch_states": v_switch_states, "opcode_value": v_opcode_value, "opcode_count": v_opcode_count}
