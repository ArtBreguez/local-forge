#!/usr/bin/env python3
"""Phase 4 report: success per arm with Wilson 95% intervals, the paired comparison of two arms (exact McNemar), median minutes per
task, the usable / not usable verdict of docs/PROTOCOL.md. Reads `<dir>/<arm>.jsonl` written by runner.py.

    python report.py results/phase4 > results/phase4/report.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "quality"))
import qbench  # noqa: E402

USABLE_RATE = 0.60
USABLE_MEDIAN_MIN = 15.0
ORDER = ["O", "T", "T-norepair", "T-ngram"]


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
    return {"tasks": n, "successes": k, "rate": rate, "wilson": (lo, hi), "runs": len(rows), "median_min": med,
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


def build_report(directory: Path, baseline: str = "O", limit_s: float | None = None) -> str:
    metas, runs = load_runs(directory)
    if not runs:
        return "no runs\n"
    arms = [a for a in ORDER if a in runs] + [a for a in sorted(runs) if a not in ORDER]
    limit = limit_s or next((m.get("time_limit_s") for m in metas.values() if m.get("time_limit_s")), 900.0)
    sums = {a: summarize(runs[a], limit) for a in arms}
    out = ["# Phase 4 report (generated by `03-agent/harness/report.py`)", "",
           f"Time limit per run: {limit / 60:.0f} min. Success = every test passes, the visible test files are byte-identical, finished inside the limit.",
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
    if baseline in runs:
        out += ["", f"## Paired comparison against {baseline} (first run of each task, exact McNemar)", "",
                "| arm | tasks in common | only arm | only " + baseline + " | both | neither | p |", "|---|---|---|---|---|---|---|"]
        for a in arms:
            if a == baseline:
                continue
            pr = paired(runs[baseline], runs[a])
            out.append(f"| {a} | {pr['n']} | {len(pr['only_b'])} | {len(pr['only_a'])} | {len(pr['both'])} | {pr['neither']} | {pr['p']:.3f} |")
            if pr["only_b"] or pr["only_a"]:
                out.append(f"|  | only {a}: {', '.join(pr['only_b']) or '-'}; only {baseline}: {', '.join(pr['only_a']) or '-'} | | | | | |")
    out += ["", "## Per task (first run: `ok`, or the reason; tests passed/total; minutes)", ""]
    tasks = sorted({r["task"] for a in arms for r in runs[a]})
    out.append("| task | kind | " + " | ".join(arms) + " |")
    out.append("|---|---|" + "---|" * len(arms))
    kinds = {r["task"]: r["kind"] for a in arms for r in runs[a]}
    firsts = {a: first_runs(runs[a]) for a in arms}
    for t in tasks:
        cells = []
        for a in arms:
            r = firsts[a].get(t)
            cells.append("-" if r is None else f"{'ok' if r['success'] else r['reason']} {r['tests_passed']}/{r['tests_total']} {r['wall_s'] / 60:.1f}m")
        out.append(f"| {t} | {kinds.get(t, '')} | " + " | ".join(cells) + " |")
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
    argv = sys.argv[1:] if argv is None else argv
    directory = Path(argv[0]) if argv else Path("results/phase4")
    sys.stdout.write(build_report(directory))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
