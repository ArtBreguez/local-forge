#!/usr/bin/env python3
"""Summarise `results/phase3/*.jsonl` as markdown: pass rate with a Wilson 95% interval per arm and an exact McNemar test
against the baseline arm on the same problems.

    python qreport.py ../../results/phase3 --baseline A

Files named `<ARM>.jsonl` hold the temperature-0, one-sample runs (pass@1). A file `<ARM>-t<temp>.jsonl` with several samples per
problem adds a pass@k section. Nothing here decides anything: a difference whose McNemar p is not below 0.05 is reported as
"within noise", which with 40 problems covers most differences smaller than about 10 percentage points.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import qbench


def pct(x: float) -> str:
    return f"{100 * x:.0f}%"


def load_arms(directory: Path) -> tuple[dict[str, tuple[dict, list[dict]]], dict[str, tuple[dict, list[dict]]]]:
    """(temperature-0 arms, sampled arms), keyed by arm label (file stem)."""
    t0: dict[str, tuple[dict, list[dict]]] = {}
    sampled: dict[str, tuple[dict, list[dict]]] = {}
    for f in sorted(directory.glob("*.jsonl")):
        meta, rows = qbench.read_results(f)
        if not rows:
            continue
        (sampled if re.search(r"-t\d", f.stem) else t0)[f.stem] = (meta, rows)
    return t0, sampled


def pass_map(rows: list[dict]) -> dict[str, bool]:
    return {r["problem"]: bool(r["pass"]) for r in rows if r.get("sample", 0) == 0}


def fraction_map(rows: list[dict]) -> dict[str, float]:
    return {r["problem"]: float(r.get("test_fraction", 1.0 if r["pass"] else 0.0)) for r in rows if r.get("sample", 0) == 0}


def sign_counts(base: dict[str, float], other: dict[str, float]) -> tuple[int, int]:
    """(problems where the baseline passed a larger share of the hidden tests, problems where the other arm did); ties are dropped."""
    common = set(base) & set(other)
    return sum(1 for p in common if base[p] > other[p] + 1e-9), sum(1 for p in common if other[p] > base[p] + 1e-9)


def paired(base: dict[str, bool], other: dict[str, bool]) -> tuple[int, int, int]:
    """(only baseline passes, only the other arm passes, problems compared)."""
    common = sorted(set(base) & set(other))
    b = sum(1 for p in common if base[p] and not other[p])
    c = sum(1 for p in common if not base[p] and other[p])
    return b, c, len(common)


def build_report(directory: Path, baseline: str = "A") -> str:
    t0, sampled = load_arms(directory)
    if not t0:
        return "no results found\n"
    out: list[str] = []
    base_pass = pass_map(t0[baseline][1]) if baseline in t0 else None
    base_text = {r["problem"]: r.get("response") for r in t0[baseline][1] if r.get("sample", 0) == 0} if baseline in t0 else {}
    out.append("| Arm | thinking | n | pass | rate | Wilson 95% CI | only A / only arm | McNemar p | vs A | same text as A | median tok/s | median tokens | cap hits | thinking budget used up | failures (kind: count) |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for arm, (meta, rows) in t0.items():
        rows0 = [r for r in rows if r.get("sample", 0) == 0]
        s = qbench.summarize(arm, rows0)
        lo, hi = qbench.wilson(s.passed, s.n)
        cmp_cols = ["—", "—", "baseline" if arm == baseline else "—"]
        if base_pass is not None and arm != baseline:
            b, c, n = paired(base_pass, pass_map(rows0))
            p = qbench.mcnemar_exact(b, c)
            verdict = "within noise" if p >= 0.05 else ("better" if c > b else "worse") + " (p<0.05)"
            if n != len(base_pass):
                verdict += f"; only {n} common problems"
            cmp_cols = [f"{b} / {c}", f"{p:.3f}", verdict]
        same = "—"
        if arm != baseline and base_text:
            pairs = [(base_text[r["problem"]], r.get("response")) for r in rows0 if r["problem"] in base_text]
            same = f"{sum(1 for x, y in pairs if x == y)}/{len(pairs)}"
        kinds = ", ".join(f"{k}: {v}" for k, v in sorted(s.kinds.items())) or "none"
        tok = f"{s.median_tok_s:.2f}" if s.median_tok_s else "n/a"
        toks = f"{s.median_tokens:.0f}" if s.median_tokens is not None else "n/a"
        out.append(f"| {arm} | {'on' if meta.get('thinking') else 'off'} | {s.n} | {s.passed} | {pct(s.passed / s.n)} | {pct(lo)}–{pct(hi)} | "
                   f"{cmp_cols[0]} | {cmp_cols[1]} | {cmp_cols[2]} | {same} | {tok} | {toks} | {s.cap_hits} | {sum(1 for r in rows0 if r.get('budget_exhausted'))} | {kinds} |")
    out.append("")

    out.append("Graded view: the share of each problem's hidden tests that passed (a syntax error, a timeout or no code counts as 0), averaged over the problems.")
    out.append("The sign test looks only at problems where the share differs from arm A.")
    out.append("")
    out.append("| Arm | mean share of tests passed | problems where A scored higher / arm scored higher | sign test p | vs A |")
    out.append("|---|---|---|---|---|")
    base_frac = fraction_map(t0[baseline][1]) if baseline in t0 else None
    for arm, (_, rows) in t0.items():
        fr = fraction_map(rows)
        mean = sum(fr.values()) / len(fr)
        if base_frac is not None and arm != baseline:
            hi_a, hi_arm = sign_counts(base_frac, fr)
            p = qbench.mcnemar_exact(hi_a, hi_arm)
            verdict = "within noise" if p >= 0.05 else ("better" if hi_arm > hi_a else "worse") + " (p<0.05)"
            out.append(f"| {arm} | {100 * mean:.1f}% | {hi_a} / {hi_arm} | {p:.3f} | {verdict} |")
        else:
            out.append(f"| {arm} | {100 * mean:.1f}% | — | — | {'baseline' if arm == baseline else '—'} |")
    out.append("")

    out.append("Pass rate by language and by difficulty (k/n):")
    out.append("")
    groups = [("py", lambda r: r["lang"] == "py"), ("js", lambda r: r["lang"] == "js"), ("easy", lambda r: r["difficulty"] == "easy"),
              ("medium", lambda r: r["difficulty"] == "medium"), ("hard", lambda r: r["difficulty"] == "hard"), ("subtle", lambda r: r["subtle"])]
    out.append("| Arm | " + " | ".join(g for g, _ in groups) + " |")
    out.append("|---|" + "---|" * len(groups))
    for arm, (_, rows) in t0.items():
        rows0 = [r for r in rows if r.get("sample", 0) == 0]
        cells = []
        for _, pred in groups:
            sel = [r for r in rows0 if pred(r)]
            cells.append(f"{sum(1 for r in sel if r['pass'])}/{len(sel)}")
        out.append(f"| {arm} | " + " | ".join(cells) + " |")
    out.append("")

    out.append("Per problem (P pass, . fail, C = fail with the token cap hit):")
    out.append("")
    arms = list(t0)
    out.append("| problem | " + " | ".join(arms) + " |")
    out.append("|---|" + "---|" * len(arms))
    problems = sorted({r["problem"] for _, rows in t0.values() for r in rows})
    by_arm = {arm: {r["problem"]: r for r in rows if r.get("sample", 0) == 0} for arm, (_, rows) in t0.items()}
    for pid in problems:
        cells = []
        for arm in arms:
            r = by_arm[arm].get(pid)
            cells.append("-" if r is None else "P" if r["pass"] else ("C" if r.get("cap_hit") else "."))
        out.append(f"| {pid} | " + " | ".join(cells) + " |")
    out.append("")

    for arm, (meta, rows) in sampled.items():
        per: dict[str, list[bool]] = {}
        for r in rows:
            per.setdefault(r["problem"], []).append(bool(r["pass"]))
        ns = {len(v) for v in per.values()}
        k = min(ns)
        p1 = sum(qbench.pass_at_k(len(v), sum(v), 1) for v in per.values()) / len(per)
        pk = sum(qbench.pass_at_k(len(v), sum(v), k) for v in per.values()) / len(per)
        out.append(f"{arm}: {len(per)} problems x {sorted(ns)} samples at temperature {meta.get('temperature')}: pass@1 = {pct(p1)}, pass@{k} = {pct(pk)}")
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("directory", type=Path)
    ap.add_argument("--baseline", default="A")
    a = ap.parse_args(argv)
    print(build_report(a.directory, a.baseline))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
