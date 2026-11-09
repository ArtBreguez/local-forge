#!/usr/bin/env python3
"""Phase 6: best-of-N with test selection, on the phase 3 benchmark.

Procedure under test (issue #23): sample N candidates for a problem (temperature ~0.7, distinct seeds), run the VISIBLE tests on each
candidate in order, keep the first that passes them, and fall back to the temperature-0 answer when none does. The visible/hidden split
of every problem's tests is fixed in `split.json` (see `split.py`) and the model never sees either part. Two success criteria are
reported for every arm, both computed from the stored code of the answers:

* **hidden** - the answer passes the hidden half of the tests (the owner's criterion: "hidden tests decide success");
* **full**   - the answer passes every test of the problem, visible and hidden, which is phase 3's definition (comparable to its pass@1).

    python bestofn.py score  A.jsonl      --out A.scored.jsonl         # run both halves on every stored answer (resumable)
                                          # add --tz America/Sao_Paulo for answers made on the UTC-3 host: some answers pass or fail by timezone
    python bestofn.py report --greedy A.jsonl --samples S10.jsonl --scored S10.scored.jsonl --greedy-scored A.scored.jsonl --ns 5,10
    python bestofn.py models --dir results/phase6/modal --ref heretic-9b --meta models.json

Time per problem is the sequential procedure's: generate a sample, run the visible tests, stop at the first pass; the temperature-0
answer is generated only when no sample passes. Test run time (well under a second) is not counted. Stdlib only.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import qbench
import split

HERE = Path(__file__).resolve().parent
KEY = tuple[str, int]


# ------------------------------------------------------------------------------------------------ scoring both halves

def score_row(p: qbench.Problem, row: dict[str, Any], parts: dict[str, Path], python: str | None = None, node: str | None = None) -> dict[str, Any]:
    rec: dict[str, Any] = {"type": "scored", "problem": row["problem"], "sample": row.get("sample", 0)}
    code = row.get("code")
    if not code:
        rec.update({"visible_pass": False, "hidden_pass": False, "kind": row.get("kind") or "no_code"})
        return rec
    v = qbench.run_tests(p, code, python=python, node=node, tests_file=parts["visible"])
    if v.kind in ("syntax", "timeout") and not v.passed:       # the same code cannot compile / terminates never: the other half says the same
        h = v
    else:
        h = qbench.run_tests(p, code, python=python, node=node, tests_file=parts["hidden"])
    rec.update({"visible_pass": v.passed, "hidden_pass": h.passed, "visible_total": v.tests_total, "hidden_total": h.tests_total,
                "kind": None if (v.passed and h.passed) else (v.kind if not v.passed else h.kind)})
    return rec


def load_scored(path: Path) -> dict[KEY, dict[str, Any]]:
    out: dict[KEY, dict[str, Any]] = {}
    if path.exists():
        for d in qbench.read_jsonl(path):
            out[(d["problem"], d["sample"])] = d
    return out


def score_file(src: Path, out: Path, workers: int = 2, python: str | None = None, node: str | None = None, pause: float = 0.0) -> dict[KEY, dict[str, Any]]:
    """Score every stored answer of `src` (a runner JSONL) on both halves of its problem's tests; appends to `out`, skips what is there."""
    problems = {p.id: p for p in qbench.load_problems()}
    _, rows = qbench.read_results(src)
    qbench.repair_tail(out)                  # a scoring run killed mid-write must not glue its next record onto a torn line
    done = load_scored(out)
    todo = [r for r in rows if r.get("kind") != "api_error" and (r["problem"], r.get("sample", 0)) not in done]
    with tempfile.TemporaryDirectory(prefix="bestofn-") as tmp:
        parts = split.write_parts(HERE, Path(tmp))
        out.parent.mkdir(parents=True, exist_ok=True)
        with ThreadPoolExecutor(max_workers=max(1, workers)) as ex, out.open("a", encoding="utf-8", newline="\n") as f:
            def one(r: dict[str, Any]) -> dict[str, Any]:
                rec = score_row(problems[r["problem"]], r, parts[r["problem"]], python, node)
                if pause:
                    time.sleep(pause)            # a duty cycle, so that scoring next to a timed benchmark on the same machine stays light
                return rec
            for rec in ex.map(one, todo):
                f.write(json.dumps(rec) + "\n")
                f.flush()
                done[(rec["problem"], rec["sample"])] = rec
    return done


# ------------------------------------------------------------------------------------------------ selection

def by_problem(rows: list[dict[str, Any]]) -> dict[str, dict[int, dict[str, Any]]]:
    out: dict[str, dict[int, dict[str, Any]]] = {}
    for r in rows:
        if r.get("kind") != "api_error":
            out.setdefault(r["problem"], {})[r.get("sample", 0)] = r
    return out


def complete_rounds(samples: dict[str, dict[int, Any]], problem_ids: list[str]) -> int:
    """How many sample rounds every problem has in full (samples 0..n-1 all present)."""
    n = 0
    while all(n in samples.get(pid, {}) for pid in problem_ids):
        n += 1
    return n


def select(greedy: dict[str, Any], cands: list[dict[str, Any]], vis: dict[KEY, dict[str, Any]], n: int) -> dict[str, Any]:
    """The first of the first `n` candidates (sample order) that passes the visible tests, else the greedy answer."""
    spent = 0.0
    tried = 0
    for c in sorted(cands, key=lambda r: r.get("sample", 0))[:n]:
        spent += c.get("wall_s") or 0.0
        tried += 1
        if vis[(c["problem"], c.get("sample", 0))]["visible_pass"]:
            return {"row": c, "source": "sample", "tried": tried, "time_s": spent}
    return {"row": greedy, "source": "greedy", "tried": tried, "time_s": spent + (greedy.get("wall_s") or 0.0)}


def evaluate(problem_ids: list[str], greedy: dict[str, dict[str, Any]], samples: dict[str, dict[int, dict[str, Any]]],
             vis: dict[KEY, dict[str, Any]], gvis: dict[KEY, dict[str, Any]], n: int) -> dict[str, Any]:
    """Best-of-n against the temperature-0 answer on the same problems. `vis` scores the samples, `gvis` the greedy answers."""
    per: dict[str, dict[str, Any]] = {}
    for pid in problem_ids:
        g = greedy[pid]
        cands = list(samples.get(pid, {}).values())
        sel = select(g, cands, vis, n)
        row = sel["row"]
        key = (row["problem"], row.get("sample", 0))
        sc = vis[key] if sel["source"] == "sample" else gvis.get((pid, g.get("sample", 0)))
        used = [c for c in sorted(cands, key=lambda r: r.get("sample", 0))[:n]]
        gsc = gvis.get((pid, g.get("sample", 0)))
        per[pid] = {
            "greedy_full": bool(g["pass"]), "greedy_hidden": bool(gsc["hidden_pass"]) if gsc else None,
            "bon_full": bool(row["pass"]), "bon_hidden": bool(sc["hidden_pass"]) if sc else None,
            "source": sel["source"], "tried": sel["tried"], "time_s": sel["time_s"],
            "time_all_s": sum(c.get("wall_s") or 0.0 for c in used) + (g.get("wall_s") or 0.0),
            "oracle_full": any(c["pass"] for c in used), "oracle_hidden": any(vis[(pid, c.get("sample", 0))]["hidden_pass"] for c in used),
            "sample_passes": sum(1 for c in used if c["pass"]), "n_used": len(used),
            "tokens": sum(c.get("completion_tokens") or 0 for c in used[:sel["tried"]]),
        }
    return per


def ci(k: int, n: int) -> str:
    lo, hi = qbench.wilson(k, n)
    return f"{k}/{n} = {100 * k / n:.0f}% ({100 * lo:.0f}-{100 * hi:.0f}%)"


def paired(per: dict[str, dict[str, Any]], a: str, b: str) -> tuple[int, int, float]:
    only_a = sum(1 for v in per.values() if v[a] and not v[b])
    only_b = sum(1 for v in per.values() if v[b] and not v[a])
    return only_a, only_b, qbench.mcnemar_exact(only_a, only_b)


def mins(xs: list[float]) -> str:
    return f"{statistics.mean(xs) / 60:.1f} / {statistics.median(xs) / 60:.1f}" if xs else "-"


def host_table(problem_ids: list[str], greedy, samples, vis, gvis, ns: list[int]) -> str:
    total = len(problem_ids)
    first = evaluate(problem_ids, greedy, samples, vis, gvis, ns[0])
    lines = ["| arm | hidden half passed (Wilson 95%) | only pass@1 / only best-of-N, McNemar p | full tests passed (phase 3's measure) | only pass@1 / only best-of-N, p "
             "| answer taken from a sample | selected answers that pass | oracle: any of N passes (full) | min per problem, mean / median (early stop) | min if all N are always generated |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    gh = sum(1 for v in first.values() if v["greedy_hidden"])
    gf = sum(1 for v in first.values() if v["greedy_full"])
    lines.append(f"| pass@1 (temperature 0, phase 3 arm A) | {ci(gh, total)} | | {ci(gf, total)} | | | | | {mins([greedy[p].get('wall_s') or 0 for p in problem_ids])} | |")
    for n in ns:
        per = evaluate(problem_ids, greedy, samples, vis, gvis, n)
        bh = sum(1 for v in per.values() if v["bon_hidden"])
        bf = sum(1 for v in per.values() if v["bon_full"])
        ah, bh2, ph = paired(per, "greedy_hidden", "bon_hidden")
        af, bf2, pf = paired(per, "greedy_full", "bon_full")
        picked = [v for v in per.values() if v["source"] == "sample"]
        good = sum(1 for v in picked if v["bon_full"])
        orc = sum(1 for v in per.values() if v["oracle_full"])
        lines.append(f"| best-of-{n} (visible tests choose) | {ci(bh, total)} | {ah} / {bh2}, p = {ph:.2f} | {ci(bf, total)} | {af} / {bf2}, p = {pf:.2f} "
                     f"| {len(picked)}/{total} | {good}/{len(picked)} | {orc}/{total} | {mins([v['time_s'] for v in per.values()])} | {mins([v['time_all_s'] for v in per.values()])} |")
    return "\n".join(lines)


def candidate_stats(problem_ids: list[str], samples, vis) -> str:
    rows = [(pid, s, r) for pid in problem_ids for s, r in samples.get(pid, {}).items()]
    n = len(rows)
    if not n:
        return "no candidates"
    full = sum(1 for _, _, r in rows if r["pass"])
    v = sum(1 for pid, s, _ in rows if vis[(pid, s)]["visible_pass"])
    h = sum(1 for pid, s, _ in rows if vis[(pid, s)]["hidden_pass"])
    both = sum(1 for pid, s, _ in rows if vis[(pid, s)]["visible_pass"] and not vis[(pid, s)]["hidden_pass"])
    toks = [r.get("completion_tokens") for _, _, r in rows if r.get("completion_tokens")]
    secs = [r["wall_s"] for _, _, r in rows if r.get("wall_s")]
    tps = [r["tok_s"] for _, _, r in rows if r.get("tok_s")]
    return (f"{n} candidates: {full} pass every test ({100 * full / n:.1f}%), {v} pass the visible half ({100 * v / n:.1f}%), {h} pass the hidden half "
            f"({100 * h / n:.1f}%); {both} pass the visible half and fail the hidden half (what the selection cannot see). "
            f"Median {statistics.median(toks):.0f} tokens, {statistics.median(secs):.0f} s, {statistics.median(tps):.1f} tok/s per candidate." if toks and secs and tps else f"{n} candidates")


def by_lang(problem_ids: list[str], per: dict[str, dict[str, Any]]) -> str:
    out = []
    for lang in ("py", "js"):
        ids = [p for p in problem_ids if p.startswith(lang)]
        out.append(f"{lang}: pass@1 {sum(per[p]['greedy_hidden'] or 0 for p in ids)}/{len(ids)} -> best-of-N {sum(per[p]['bon_hidden'] or 0 for p in ids)}/{len(ids)} (hidden half)")
    return "; ".join(out)


# ------------------------------------------------------------------------------------------------ model comparison

def model_table(dirpath: Path, ref: str, meta: dict[str, Any], ns: list[int]) -> str:
    problem_ids = [p.id for p in qbench.load_problems()]
    models: dict[str, dict[str, Any]] = {}
    for g in sorted(dirpath.glob("*.greedy.jsonl")):
        name = g.name[: -len(".greedy.jsonl")]
        s = dirpath / f"{name}.sample.jsonl"
        sc, gsc = dirpath / f"{name}.sample.scored.jsonl", dirpath / f"{name}.greedy.scored.jsonl"
        if not (s.exists() and sc.exists() and gsc.exists()):
            continue
        _, grows = qbench.read_results(g)
        _, srows = qbench.read_results(s)
        models[name] = {"greedy": {r["problem"]: r for r in grows if r.get("kind") != "api_error"}, "samples": by_problem(srows),
                        "vis": load_scored(sc), "gvis": load_scored(gsc)}
    head = ("| model | uncensored | pass@1: hidden half (Wilson 95%) | full tests | vs " + ref + " (hidden): only ref / only model, McNemar p | mean candidate pass rate at T=0.7 (full) | "
            + " | ".join(f"best-of-{n} hidden / full" for n in ns) + " | oracle pass@10 (full) |")
    lines = [head, "|" + "---|" * (head.count("|") - 1)]
    refm = models[ref]
    ref_hidden = {pid: bool(refm["gvis"][(pid, refm["greedy"][pid].get("sample", 0))]["hidden_pass"]) for pid in problem_ids}
    order = sorted(models, key=lambda m: (not meta.get(m, {}).get("uncensored", False), -sum(
        1 for pid in problem_ids if models[m]["gvis"][(pid, models[m]["greedy"][pid].get("sample", 0))]["hidden_pass"])))
    for name in order:
        m = models[name]
        per1 = evaluate(problem_ids, m["greedy"], m["samples"], m["vis"], m["gvis"], 1)
        gh = sum(1 for v in per1.values() if v["greedy_hidden"])
        gf = sum(1 for v in per1.values() if v["greedy_full"])
        only_ref = sum(1 for pid in problem_ids if ref_hidden[pid] and not per1[pid]["greedy_hidden"])
        only_m = sum(1 for pid in problem_ids if per1[pid]["greedy_hidden"] and not ref_hidden[pid])
        cells = []
        rounds = complete_rounds(m["samples"], problem_ids)
        for n in ns:
            if rounds < n:
                cells.append(f"(only {rounds} rounds)")
                continue
            per = evaluate(problem_ids, m["greedy"], m["samples"], m["vis"], m["gvis"], n)
            cells.append(f"{sum(1 for v in per.values() if v['bon_hidden'])} / {sum(1 for v in per.values() if v['bon_full'])}")
        tot = sum(len(v) for v in m["samples"].values())
        okc = sum(1 for v in m["samples"].values() for r in v.values() if r["pass"])
        orc = sum(1 for pid in problem_ids if any(r["pass"] for r in m["samples"].get(pid, {}).values()))
        info = meta.get(name, {})
        cmp_cell = "(reference)" if name == ref else f"{only_ref} / {only_m}, p = {qbench.mcnemar_exact(only_ref, only_m):.3f}"
        lines.append(f"| {name}{'' if info.get('uncensored') else ' (censored: reference row only)'} | {'yes' if info.get('uncensored') else 'no'} | {ci(gh, len(problem_ids))} | {gf}/40 "
                     f"| {cmp_cell} | {100 * okc / tot:.1f}% ({okc}/{tot}) | " + " | ".join(cells) + f" | {orc}/40 |")
    return "\n".join(lines)


# ------------------------------------------------------------------------------------------------ CLI

def parse_ns(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sc = sub.add_parser("score", help="run the visible and the hidden half on every stored answer of a runner JSONL")
    sc.add_argument("src", type=Path)
    sc.add_argument("--out", type=Path, required=True)
    sc.add_argument("--workers", type=int, default=2)
    sc.add_argument("--pause", type=float, default=0.0, help="seconds to sleep after every scored answer (per worker)")
    sc.add_argument("--python", default=None)
    sc.add_argument("--node", default=None)
    sc.add_argument("--tz", default=None, help="timezone the tests run in (e.g. America/Sao_Paulo for answers generated and scored on the UTC-3 host)")
    rp = sub.add_parser("report", help="pass rates and time of best-of-N against pass@1")
    rp.add_argument("--greedy", type=Path, required=True)
    rp.add_argument("--samples", type=Path, required=True)
    rp.add_argument("--scored", type=Path, required=True, help="output of `score` for --samples")
    rp.add_argument("--greedy-scored", type=Path, required=True, help="output of `score` for --greedy")
    rp.add_argument("--ns", default="5,10")
    md = sub.add_parser("models", help="the model comparison over a directory of <model>.greedy|sample[.scored].jsonl")
    md.add_argument("--dir", type=Path, required=True)
    md.add_argument("--ref", required=True, help="the model every other one is compared with (McNemar on the hidden half)")
    md.add_argument("--meta", type=Path, help="JSON {model: {uncensored: bool, ...}}")
    md.add_argument("--ns", default="5,10")
    a = ap.parse_args(argv)
    if a.cmd == "score":
        if a.tz:
            os.environ["QBENCH_TZ"] = a.tz      # read by qbench.run_tests
        done = score_file(a.src, a.out, a.workers, a.python, a.node, a.pause)
        print(f"{len(done)} scored answers in {a.out}")
        return 0
    if a.cmd == "report":
        problem_ids = [p.id for p in qbench.load_problems()]
        _, grows = qbench.read_results(a.greedy)
        _, srows = qbench.read_results(a.samples)
        greedy = {r["problem"]: r for r in grows if r.get("kind") != "api_error"}
        samples = by_problem(srows)
        vis, gvis = load_scored(a.scored), load_scored(a.greedy_scored)
        rounds = complete_rounds(samples, problem_ids)
        ns = [n for n in parse_ns(a.ns) if n <= rounds]
        print(f"complete sample rounds: {rounds} (reported N: {ns}); problems with a greedy answer: {len(greedy)}/{len(problem_ids)}\n")
        print(host_table(problem_ids, greedy, samples, vis, gvis, ns), "\n")
        print(candidate_stats(problem_ids, samples, vis), "\n")
        if ns:
            print(by_lang(problem_ids, evaluate(problem_ids, greedy, samples, vis, gvis, ns[-1])))
        return 0
    meta = json.loads(a.meta.read_text(encoding="utf-8")) if a.meta else {}
    print(model_table(a.dir, a.ref, meta, parse_ns(a.ns)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
