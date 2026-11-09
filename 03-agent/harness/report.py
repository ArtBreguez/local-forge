#!/usr/bin/env python3
"""Phase 4/5 report: success per arm with Wilson 95% intervals, the paired comparison of two arms (exact McNemar), median minutes per
task, the usable / not usable verdict of docs/PROTOCOL.md. Reads `<dir>/<arm>.jsonl` written by runner.py.

    python report.py results/phase4 > results/phase4/report.md
    python report.py results/phase5 --phase4 results/phase4 > results/phase5/report.md

Phase 5 adds what a 30-minute arm needs: each arm is judged under its own limit *and* re-scored under the protocol's 15 minutes (a pass that
took more than 900 s does not count there), the passes that arrive after minute 15 are listed, the tasks that timed out in phase 4 get
their own table, and the speed of an arm (the server's own generation tok/s, speculative draft acceptance) is reported next to its success.
`--phase4 DIR` brings the O rows of phase 4 in as the baseline `O`.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "quality"))
import qbench  # noqa: E402

USABLE_RATE = 0.60
USABLE_MEDIAN_MIN = 15.0
ORDER = ["O", "T", "T-norepair", "T-ngram", "O30", "O30-ngram", "O30-tool"]
LIMIT15_S = 900.0


def load_runs(directory: Path) -> tuple[dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    """({arm: meta}, {arm: [run records]}); a (task, run) written twice keeps the last record."""
    metas: dict[str, dict[str, Any]] = {}
    runs: dict[str, list[dict[str, Any]]] = {}
    for path in sorted(directory.glob("*.jsonl")):
        by_key: dict[tuple[str, int], dict[str, Any]] = {}
        arm = path.stem
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("type") == "meta":
                metas[r.get("arm", arm)] = r
            elif r.get("type") == "run":
                by_key[(r["task"], r["run"])] = r
                arm = r["arm"]
        if by_key:
            runs[arm] = sorted(by_key.values(), key=lambda r: (r["run"], r["task"]))
    return metas, runs


def first_runs(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for r in rows:
        if r["task"] not in out or r["run"] < out[r["task"]]["run"]:
            out[r["task"]] = r
    return out


def summarize(rows: list[dict[str, Any]], limit_s: float) -> dict[str, Any]:
    first = first_runs(rows)
    n = len(first)
    k = sum(1 for r in first.values() if r["success"])
    lo, hi = qbench.wilson(k, n) if n else (0.0, 0.0)
    mins_all = [min(r["wall_s"], limit_s) / 60 for r in rows]
    mins_ok = [r["wall_s"] / 60 for r in rows if r["success"]]
    med = qbench.median(mins_all)
    graded = [r["tests_passed"] / r["tests_total"] for r in first.values() if r.get("tests_total")]
    reasons: dict[str, int] = {}
    for r in first.values():
        reasons[r["reason"]] = reasons.get(r["reason"], 0) + 1
    rate = k / n if n else 0.0
    k15 = sum(1 for r in first.values() if r["success"] and r["wall_s"] <= LIMIT15_S)
    lo15, hi15 = qbench.wilson(k15, n) if n else (0.0, 0.0)
    med15 = qbench.median([min(r["wall_s"], LIMIT15_S) / 60 for r in rows])
    eff = [r["completion_tokens"] / r["model_seconds"] for r in rows if r.get("completion_tokens") and r.get("model_seconds")]
    drafted = sum(r.get("draft_n") or 0 for r in rows)
    return {"limit_s": limit_s, "first_ok_15": k15, "first_after_15": k - k15, "wilson_15": (lo15, hi15), "rate_15": k15 / n if n else 0.0,
            "median_min_15cap": med15, "median_min_first": qbench.median([min(r["wall_s"], limit_s) / 60 for r in first.values()]),
            "usable_15": bool(n) and k15 / n >= USABLE_RATE and med15 is not None and med15 <= USABLE_MEDIAN_MIN,
            "gen_tok_s_median": qbench.median([r["gen_tok_s"] for r in rows if r.get("gen_tok_s")]), "eff_tok_s_median": qbench.median(eff),
            "pp_tok_s_median": qbench.median([r["pp_tok_s"] for r in rows if r.get("pp_tok_s")]),
            "draft_rate": (sum(r.get("draft_accepted") or 0 for r in rows) / drafted) if drafted else None,
            "tasks": n, "successes": k, "rate": rate, "wilson": (lo, hi), "runs": len(rows), "median_min": med,
            "median_min_success": qbench.median(mins_ok), "graded": sum(graded) / len(graded) if graded else None,
            "steps_median": qbench.median([r["steps"] for r in rows if r.get("steps") is not None]),
            "tool_calls_median": qbench.median([r["tool_calls"] for r in rows if r.get("tool_calls") is not None]),
            "completion_tokens_median": qbench.median([r["completion_tokens"] for r in rows if r.get("completion_tokens") is not None]),
            "reasons": reasons, "timeouts": sum(1 for r in rows if r.get("timed_out")),
            "usable": bool(n) and rate >= USABLE_RATE and med is not None and med <= USABLE_MEDIAN_MIN,
            "usable_rate": rate >= USABLE_RATE, "usable_time": med is not None and med <= USABLE_MEDIAN_MIN}


def pooled(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Every run of the arm counted, not only the first run of each task: the success probability of one run (a second estimator for an
    agent that is not repeatable; the runs of one task are not independent, so the interval is optimistic)."""
    n, k = len(rows), sum(1 for r in rows if r["success"])
    lo, hi = qbench.wilson(k, n) if n else (0.0, 0.0)
    return {"runs": n, "successes": k, "rate": k / n if n else 0.0, "wilson": (lo, hi)}


def paired(a_rows: list[dict[str, Any]], b_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """First run of every task that both arms have: (only a, only b, both, neither), exact McNemar p."""
    fa, fb = first_runs(a_rows), first_runs(b_rows)
    common = sorted(set(fa) & set(fb))
    only_a = [t for t in common if fa[t]["success"] and not fb[t]["success"]]
    only_b = [t for t in common if fb[t]["success"] and not fa[t]["success"]]
    both = [t for t in common if fa[t]["success"] and fb[t]["success"]]
    p = qbench.mcnemar_exact(len(only_a), len(only_b)) if common else 1.0
    return {"n": len(common), "only_a": only_a, "only_b": only_b, "both": both, "neither": len(common) - len(only_a) - len(only_b) - len(both), "p": p}


def stability(rows: list[dict[str, Any]]) -> tuple[int, int]:
    """(tasks with two or more runs, of which every run agrees on success)."""
    by: dict[str, list[bool]] = {}
    for r in rows:
        by.setdefault(r["task"], []).append(bool(r["success"]))
    multi = {t: v for t, v in by.items() if len(v) > 1}
    return len(multi), sum(1 for v in multi.values() if len(set(v)) == 1)


def fmt_min(x: float | None) -> str:
    return "-" if x is None else f"{x:.1f}"


def arm_limit(rows: list[dict[str, Any]], meta: dict[str, Any], default: float = 900.0) -> float:
    """The limit an arm ran under: the rows say it (`limit_s`), else the meta line, else the default."""
    return float(max((r.get("limit_s") or 0 for r in rows), default=0) or meta.get("time_limit_s") or default)


def cell(r: dict[str, Any] | None) -> str:
    """One run in a table: `ok 7/7 3.1m` or `timeout 5/10 30.0m`."""
    if r is None:
        return "-"
    return f"{'ok' if r['success'] else r['reason']} {r['tests_passed']}/{r['tests_total']} {r['wall_s'] / 60:.1f}m"


def end_kind(r: dict[str, Any]) -> str:
    """How a failed run ended: `timeout` (the limit killed it), `error: <type>` (OpenCode's own error event ended it, e.g. a compaction that
    produced no summary) or `stopped on its own` (the agent ended the session with a test still failing: it gave up, which is not time).
    Older records have no `agent_errors`; for them the tail of the agent log (where an ending error sits) is read."""
    if r.get("success"):
        return "ok"
    if r.get("timed_out"):
        return "timeout"
    errs = r.get("agent_errors")
    if errs:
        return "error: " + errs[-1].split(":")[0]
    if errs is None:
        m = re.search(r'"type":"error".*?"error":\{"type":"([^"]+)"', r.get("agent_log_tail") or "")
        if m:
            return "error: " + m.group(1)
    return "stopped on its own"


def timeout_table(base_rows: list[dict[str, Any]], arms: dict[str, list[dict[str, Any]]]) -> list[str]:
    """Every task that timed out in some run of the baseline (phase 4's O): what the baseline runs did and what each other arm did, by run,
    so that "did it finish in 30 minutes, and at which minute" can be read off. A pass after minute 15 is marked `*`: time bought it."""
    slow = sorted({r["task"] for r in base_rows if r.get("timed_out")})
    if not slow:
        return []
    names = list(arms)
    out = ["| task | " + " | ".join(["O phase 4 (15 min)"] + [f"{a} run {k}" for a in names for k in (1, 2)]) + " |",
           "|---|" + "---|" * (1 + 2 * len(names))]
    for t in slow:
        base = "; ".join(f"run {r['run']}: {cell(r)}" for r in sorted(base_rows, key=lambda r: r["run"]) if r["task"] == t)
        cells = []
        for a in names:
            for k in (1, 2):
                r = next((x for x in arms[a] if x["task"] == t and x["run"] == k), None)
                c = cell(r)
                if r is not None and r["success"] and r["wall_s"] > LIMIT15_S:
                    c += " *"
                cells.append(c)
        out.append(f"| {t} | {base} | " + " | ".join(cells) + " |")
    out += ["", "`*` = passed, and only after minute 15."]
    return out


def build_report(directory: Path, baseline: str = "O", limit_s: float | None = None, phase4: Path | None = None) -> str:
    metas, runs = load_runs(directory)
    if phase4 is not None:                       # phase 4's OpenCode rows, as the baseline arm "O"
        m4, r4 = load_runs(phase4)
        if "O" in r4:
            runs["O"], metas["O"] = r4["O"], m4.get("O", {})
    if not runs:
        return "no runs\n"
    arms = [a for a in ORDER if a in runs] + [a for a in sorted(runs) if a not in ORDER]
    limits = {a: limit_s or arm_limit(runs[a], metas.get(a, {})) for a in arms}
    sums = {a: summarize(runs[a], limits[a]) for a in arms}
    title = "Phase 5" if any(a.startswith("O30") for a in arms) else "Phase 4"
    out = [f"# {title} report (generated by `03-agent/harness/report.py`)", "",
           "Time limit per run: " + ", ".join(f"{a} {limits[a] / 60:.0f} min" for a in arms) + ". Success = every test passes, the visible test files are "
           "byte-identical, finished inside the arm's limit.",
           "Primary unit: the first run of each task (every arm has one); a second run, where it exists, is used only for the stability line.", "",
           "| arm | tasks | success | rate (Wilson 95%) | median min (all runs) | median min (successes) | timeouts | tests passed (mean share) | median steps | median tool calls | median completion tokens | usable? |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for a in arms:
        s = sums[a]
        lo, hi = s["wilson"]
        graded = "-" if s["graded"] is None else f"{100 * s['graded']:.0f}%"
        verdict = "**yes**" if s["usable"] else "no"
        out.append(f"| {a} | {s['tasks']} | {s['successes']}/{s['tasks']} | {100 * s['rate']:.0f}% ({100 * lo:.0f}-{100 * hi:.0f}%) | "
                   f"{fmt_min(s['median_min'])} | {fmt_min(s['median_min_success'])} | {s['timeouts']} | {graded} | "
                   f"{fmt_min(s['steps_median'])} | {fmt_min(s['tool_calls_median'])} | {fmt_min(s['completion_tokens_median'])} | {verdict} |")
    out += ["", "Pooled over every run of the arm (second estimator; the runs of one task are not independent, so these intervals are optimistic): "
            + "; ".join(f"**{a}** {pooled(runs[a])['successes']}/{pooled(runs[a])['runs']} = {100 * pooled(runs[a])['rate']:.0f}% "
                        f"({100 * pooled(runs[a])['wilson'][0]:.0f}-{100 * pooled(runs[a])['wilson'][1]:.0f}%)" for a in arms) + ".",
            "", "Usable = success rate >= 60% **and** median time per task <= 15 min (docs/PROTOCOL.md). Times count a failed run at the time it ran "
            "(at most the limit).", "", "## Why runs failed (first run of each task)", ""]
    for a in arms:
        out.append(f"- **{a}**: " + ", ".join(f"{k} {v}" for k, v in sorted(sums[a]["reasons"].items(), key=lambda kv: -kv[1])))
    out += ["", "## How the failed runs ended (first run of each task)", "",
            "| arm | failed | timeout | stopped on its own with a test failing | ended by an OpenCode error | which |", "|---|---|---|---|---|---|"]
    for a in arms:
        fails = [end_kind(r) for r in first_runs(runs[a]).values() if not r["success"]]
        errs = sorted({k for k in fails if k.startswith("error")})
        out.append(f"| {a} | {len(fails)} | {fails.count('timeout')} | {fails.count('stopped on its own')} | {sum(1 for k in fails if k.startswith('error'))} | "
                   f"{', '.join(errs) or '-'} |")
    long_arms = [a for a in arms if limits[a] > LIMIT15_S]
    if long_arms:
        out += ["", "## Time: each arm at its own limit, and re-scored at the protocol's 15 minutes (first run of each task)", "",
                "| arm | limit | success at its limit | success within 15 min | passes after minute 15 | median min, first runs | median min, all runs | "
                "median min, all runs capped at 15 | usable at its limit? | usable at 15 min? |", "|---|---|---|---|---|---|---|---|---|---|"]
        for a in arms:
            x = sums[a]
            out.append(f"| {a} | {limits[a] / 60:.0f} min | {x['successes']}/{x['tasks']} = {100 * x['rate']:.0f}% ({100 * x['wilson'][0]:.0f}-{100 * x['wilson'][1]:.0f}%) | "
                       f"{x['first_ok_15']}/{x['tasks']} = {100 * x['rate_15']:.0f}% ({100 * x['wilson_15'][0]:.0f}-{100 * x['wilson_15'][1]:.0f}%) | "
                       f"{x['first_after_15']} | {fmt_min(x['median_min_first'])} | {fmt_min(x['median_min'])} | {fmt_min(x['median_min_15cap'])} | "
                       f"{'**yes**' if x['usable'] else 'no'} | {'**yes**' if x['usable_15'] else 'no'} |")
        out += ["", "A pass counts at 15 minutes only if its run took 900 s or less; a run that ran longer counts as 15 minutes in the capped median. "
                "The protocol's time criterion is the median per task <= 15 min, and a longer limit does not change it."]
    pairs = [(baseline, a) for a in arms if a != baseline] if baseline in runs else []
    if "O30" in runs:
        pairs += [("O30", a) for a in ("O30-ngram", "O30-tool") if a in runs]
    if pairs:
        out += ["", "## Paired comparisons (first run of each task, exact McNemar)", "",
                "| arm | against | tasks in common | only arm | only against | both | neither | p |", "|---|---|---|---|---|---|---|---|"]
        for base, a in pairs:
            pr = paired(runs[base], runs[a])
            out.append(f"| {a} | {base} | {pr['n']} | {len(pr['only_b'])} | {len(pr['only_a'])} | {len(pr['both'])} | {pr['neither']} | {pr['p']:.3f} |")
            if pr["only_b"] or pr["only_a"]:
                out.append(f"|  | only {a}: {', '.join(pr['only_b']) or '-'}; only {base}: {', '.join(pr['only_a']) or '-'} | | | | | | |")
    if baseline in runs and long_arms:
        tt = timeout_table(runs[baseline], {a: runs[a] for a in long_arms})
        if tt:
            out += ["", f"## Tasks that timed out at 15 minutes in {baseline} (any run): did they finish inside 30, and at which minute?", ""] + tt
    spoken = [a for a in arms if sums[a]["gen_tok_s_median"] or sums[a]["eff_tok_s_median"]]
    if spoken:
        out += ["", "## Speed (per run, from llama-server's own timings; `effective` = completion tokens over the model seconds the proxy measured)", "",
                "| arm | median generation tok/s | median prompt tok/s | median effective tok/s | draft acceptance | median model steps | median completion tokens | model minutes, all runs |",
                "|---|---|---|---|---|---|---|---|"]
        for a in spoken:
            x = sums[a]
            mins = sum(r.get("model_seconds") or 0 for r in runs[a]) / 60
            dr = "-" if x["draft_rate"] is None else f"{100 * x['draft_rate']:.0f}%"
            out.append(f"| {a} | {fmt_min(x['gen_tok_s_median'])} | {fmt_min(x['pp_tok_s_median'])} | {fmt_min(x['eff_tok_s_median'])} | {dr} | "
                       f"{fmt_min(x['steps_median'])} | {fmt_min(x['completion_tokens_median'])} | {mins:.0f} |")
    for a in arms:
        used = [r["tool_runs"] for r in runs[a] if r.get("tool_runs") is not None]
        if used:
            called = [u for u in used if u["calls"]]
            out += ["", f"## The test tool in {a}", "",
                    f"- runs: {len(used)}; runs that called it at least once: {len(called)}; calls in all runs: {sum(u['calls'] for u in used)} "
                    f"(median per run that used it: {fmt_min(qbench.median([u['calls'] for u in called]))}); "
                    f"calls that reported all tests passing: {sum(u['passing'] for u in used)}; runs whose last call said all pass: {sum(1 for u in used if u['last_ok'])}."]
    out += ["", "## Per task (first run: `ok`, or the reason; tests passed/total; minutes)", ""]
    tasks = sorted({r["task"] for a in arms for r in runs[a]})
    out.append("| task | kind | " + " | ".join(arms) + " |")
    out.append("|---|---|" + "---|" * len(arms))
    kinds = {r["task"]: r["kind"] for a in arms for r in runs[a]}
    firsts = {a: first_runs(runs[a]) for a in arms}
    for t in tasks:
        cells = []
        for a in arms:
            cells.append(cell(firsts[a].get(t)))
        out.append(f"| {t} | {kinds.get(t, '')} | " + " | ".join(cells) + " |")
    if len(arms) > 1:
        out += ["", "## Passes per task over every run of every arm (k of n runs; a task that passes in all of them or in none is not noise)", "",
                "| task | " + " | ".join(arms) + " | all arms |", "|---|" + "---|" * (len(arms) + 1)]
        for t in tasks:
            per = {a: [bool(r["success"]) for r in runs[a] if r["task"] == t] for a in arms}
            tot_k, tot_n = sum(sum(v) for v in per.values()), sum(len(v) for v in per.values())
            out.append(f"| {t} | " + " | ".join(f"{sum(v)}/{len(v)}" if v else "-" for v in per.values()) + f" | {tot_k}/{tot_n} |")
    out += ["", "## Stability (tasks with two runs)", ""]
    for a in arms:
        multi, same = stability(runs[a])
        if multi:
            out.append(f"- {a}: {same} of {multi} tasks gave the same success/fail in every run")
    orch = {a: [r["orchestrator"] for r in runs[a] if r.get("orchestrator")] for a in arms}
    for a in arms:
        if orch[a]:
            o = orch[a]
            out.append(f"- {a} orchestrator: repairs applied {sum(x.get('repairs') or 0 for x in o)}, syntax rejects {sum(x.get('syntax_rejects') or 0 for x in o)}, "
                       f"replies without an edit {sum(x.get('no_edit') or 0 for x in o)}, review rounds {sum(x.get('reviews') or 0 for x in o)}, "
                       f"stop reasons: " + ", ".join(sorted({str(x.get('stop_reason')) for x in o})))
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("directory", nargs="?", type=Path, default=Path("results/phase4"))
    ap.add_argument("--phase4", type=Path, default=None, help="directory with phase 4's O.jsonl: its rows become the baseline arm O")
    ap.add_argument("--baseline", default="O")
    a = ap.parse_args(sys.argv[1:] if argv is None else argv)
    sys.stdout.write(build_report(a.directory, baseline=a.baseline, phase4=a.phase4))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
