"""Score what `lora_eval.py` brought back: the base student and base + adapter on the same held-out tasks, with the truth that stayed on this machine.

    python3 -m distill.backends.kaggle.score_lora --domain rea --heldout results/phase6/rea-data/decompiled-seed1001-all.jsonl \\
        --base ~/work/kaggle-lora/out/replies-base.jsonl --lora ~/work/kaggle-lora/out/replies-lora.jsonl [--run-info ~/work/kaggle-lora/out/run.json] [--out report.json]

A reply is judged by the code `kaggle-score` runs: `score_replies.reply_of` (a cut-off reply ends in the truncation marker, a failed request is an error) and `traces.verdict`, whose last step is the subject's
own `domain.verify` (REA: `scoring.score_strict`, the whitelist grammar on the text after the last `Answer:` line). A reply that answered another prompt than the task's now (its `prompt_sha`) is refused, as
`kaggle-score` refuses it. For the generated families `scoring.score` (the evaluation scorer of rea-bench's held-out arm) and `score_strict` are the same function; they differ only for the legacy templates.

Per family and arm: replies, accepted (accuracy, with its 95% Wilson interval), wrong, no `Answer:` line, truncated, errors, the share cut off at the budget (`finish == "length"`) and the mean completion tokens,
over the replies BOTH arms have (a reply only one arm has, after a hard kill between the two arms of a batch, is counted apart as unpaired and enters no figure).
Per family and overall, the two arms PAIRED on the same replies: both right, base only, adapter only, both wrong, and the two-sided exact sign test over TASKS: a task goes to the arm that has more accepted replies
of it (with k=1 that is the discordant pairs; the k replies of a task are not independent draws, so they are never counted one by one). With ten tasks of a family an interval is wide (10 of 10 against 6 of
10 is not a significant difference by itself): read the overall paired line first, the families second. Stdlib only.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from distill import outputs, score_replies, traces  # noqa: E402

ARMS = ("base", "lora")


def wilson(k: int, n: int, z: float = 1.96) -> tuple:
    """The Wilson score interval of k successes in n trials (0, 0 for no trial)."""
    if n == 0:
        return 0.0, 0.0
    p, d = k / n, 1 + z * z / n
    centre, half = p + z * z / (2 * n), z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, (centre - half) / d), min(1.0, (centre + half) / d)


def sign_test_p(only_a: int, only_b: int) -> float:
    """Two-sided exact p-value of the sign test: `only_a` and `only_b` replies that exactly one arm got right, under 'either is as likely'. 1.0 for no discordant pair."""
    n = only_a + only_b
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(only_a, only_b) + 1)) / 2 ** n)


def read_arm(path, tasks: dict, *, domain) -> dict:
    """{(task id, sample index): {"kind", "verdict", "finish", "tokens"}} of one arm's replies file. A reply of a task that is not in `tasks` is counted apart (`info["unknown"]`), a missing or malformed entry
    is counted and skipped, a reply with no fingerprint is skipped as unchecked, and a reply that answered another prompt is SystemExit (the same refusal as `kaggle-score`)."""
    lines, info = score_replies.read_replies(path)
    info = collections.Counter(info)
    rows = {}
    for tid, rec in lines.items():
        task = tasks.get(tid)
        if task is None:
            info["unknown"] += 1
            continue
        sha = outputs.prompt_sha(domain.messages(task, "thinking"))
        for j, (out, own) in enumerate(zip(rec["outs"], score_replies.reply_fingerprints(rec))):
            if not isinstance(out, dict):
                info["missing"] += 1
                continue
            if own is None:
                info["unchecked"] += 1
                continue
            if own != sha:
                raise SystemExit(f"{path}: the reply {j} of {tid} answered another prompt than the task's now (fingerprint {own}, now {sha}): score the run that was built from these held-out tasks")
            verdict = traces.verdict(score_replies.reply_of(out, thinking=True), task, domain=domain)
            rows[(tid, j)] = {"kind": rec.get("kind") or task["kind"], "verdict": verdict, "finish": out.get("finish"), "tokens": out.get("tokens") if isinstance(out.get("tokens"), (int, float)) else 0}
    return {"rows": rows, "info": dict(info)}


def summarize(rows: dict) -> dict:
    """{"overall": stats, "by_kind": {kind: stats}} of one arm's rows. stats: replies, accepted, accuracy, ci95, wrong, no_answer_line, truncated, error, cut_off (share with finish length), mean_tokens."""
    groups = collections.defaultdict(list)
    for r in rows.values():
        groups["*"].append(r)
        groups[r["kind"]].append(r)

    def stats(rs: list) -> dict:
        n, c = len(rs), collections.Counter(r["verdict"] for r in rs)
        lo, hi = wilson(c["accepted"], n)
        return {"replies": n, "accepted": c["accepted"], "accuracy": round(c["accepted"] / n, 4) if n else None, "ci95": [round(lo, 3), round(hi, 3)], "wrong": c["wrong"], "no_answer_line": c["no_answer_line"],
                "truncated": c["truncated"], "error": c["error"], "cut_off": round(sum(r["finish"] == "length" for r in rs) / n, 4) if n else None,
                "mean_tokens": round(sum(r["tokens"] for r in rs) / n, 1) if n else None}
    return {"overall": stats(groups.pop("*", [])), "by_kind": {k: stats(v) for k, v in sorted(groups.items())}}


def pair(base: dict, lora: dict) -> dict:
    """The two arms on the replies both have: {"overall": counts, "by_kind": {kind: counts}} with counts = both, base_only, lora_only, neither (replies), n (replies), tasks, tasks_base_better, tasks_lora_better,
    per_task (the most paired replies one task has) and sign_p (the exact sign test over the TASKS: a task is the base's when the base has more accepted replies of it, the adapter's when it has, a tie is dropped),
    and `unpaired` = replies only one arm has (`unpaired_by_arm` per arm)."""
    common = sorted(set(base) & set(lora))
    groups, wins = collections.defaultdict(collections.Counter), collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0, 0]))
    for key in common:
        a, b = base[key]["verdict"] == "accepted", lora[key]["verdict"] == "accepted"
        name = "both" if a and b else "base_only" if a else "lora_only" if b else "neither"
        for g in ("*", base[key]["kind"]):
            groups[g][name] += 1
            tally = wins[g][key[0]]
            tally[0], tally[1], tally[2] = tally[0] + 1, tally[1] + a, tally[2] + b

    def counts(c: collections.Counter, per_task: dict) -> dict:
        d = {k: c[k] for k in ("both", "base_only", "lora_only", "neither")}
        won_base, won_lora = (sum(1 for n, x, y in per_task.values() if (x > y if arm == "base" else y > x)) for arm in ("base", "lora"))
        return {**d, "n": sum(d.values()), "tasks": len(per_task), "tasks_base_better": won_base, "tasks_lora_better": won_lora, "per_task": max((n for n, _, _ in per_task.values()), default=0),
                "sign_p": round(sign_test_p(won_base, won_lora), 4)}
    return {"overall": counts(groups.pop("*", collections.Counter()), wins.pop("*", {})), "by_kind": {k: counts(v, wins[k]) for k, v in sorted(groups.items())}, "unpaired": len(set(base) ^ set(lora)),
            "unpaired_by_arm": {"base": len(set(base) - set(lora)), "lora": len(set(lora) - set(base))}}


def score(base_path, lora_path, heldout_files, *, domain) -> dict:
    """The report: per arm `summarize` of the replies BOTH arms have and the read counts, the `pair`, and how many tasks the held-out files have against how many were answered."""
    tasks = {t["id"]: t for t in domain.load_heldout([pathlib.Path(f) for f in heldout_files])}
    arms = {"base": read_arm(base_path, tasks, domain=domain), "lora": read_arm(lora_path, tasks, domain=domain)}
    common = set(arms["base"]["rows"]) & set(arms["lora"]["rows"])
    return {"arms": {a: {**summarize({k: r for k, r in v["rows"].items() if k in common}), "read": v["info"], "unpaired": len(v["rows"]) - len(common)} for a, v in arms.items()},
            "paired": pair(arms["base"]["rows"], arms["lora"]["rows"]), "tasks_answered": len({tid for v in arms.values() for tid, _ in v["rows"]}), "tasks_in_files": len(tasks)}


def lines_of(report: dict, run_info: dict = None) -> list:
    """The table printed: a header about the adapter's training (from run.json), then one row per family and the overall row."""
    out = []
    if run_info:
        t = run_info.get("train") or {}
        if run_info.get("smoke"):
            out.append("WARNING: this is a SMOKE run (4 steps, one batch of replies): its numbers say nothing about the adapter")
        lr_last, lr_peak = t.get("lr_last"), t.get("lr_peak")
        stopped = " (stopped by the deadline" + (f", at a learning rate of {lr_last:.1e}, {100 * lr_last / lr_peak:.0f}% of the peak: the cosine had not finished, a weaker and noisier adapter than the planned one)" if lr_last is not None and lr_peak else ")") if t.get("deadline_hit") else ""
        out.append(f"adapter: {t.get('status', '?')}, {t.get('steps_done', '?')} of {t.get('steps_planned', '?')} steps, loss {t.get('loss_first')} -> {t.get('loss_last')}, {t.get('records')}" + stopped)
    pct = lambda s: "  -  " if s["accuracy"] is None else f"{100 * s['accuracy']:4.0f}%"
    out.append(f"{'family':<12}{'n':>4}  {'base acc [95% CI]':<22}{'lora acc [95% CI]':<22}{'only base':>10}{'only lora':>10}{'sign p':>8}   cut off b/l   mean tokens b/l")
    rows = [("OVERALL", "overall")] + [(k, k) for k in report["arms"]["base"]["by_kind"]]
    for label, key in rows:
        sel = (lambda arm: report["arms"][arm]["overall"]) if key == "overall" else (lambda arm, k=key: report["arms"][arm]["by_kind"].get(k))
        b, l = sel("base"), sel("lora")
        pr = report["paired"]["overall"] if key == "overall" else report["paired"]["by_kind"].get(key)
        if not (b and l and pr):
            continue
        ci = lambda s: f"{pct(s)} [{100 * s['ci95'][0]:.0f}-{100 * s['ci95'][1]:.0f}]"
        cut = lambda s: "-" if s["cut_off"] is None else f"{100 * s['cut_off']:.0f}%"
        out.append(f"{label:<12}{pr['n']:>4}  {ci(b):<22}{ci(l):<22}{pr['base_only']:>10}{pr['lora_only']:>10}{pr['sign_p']:>8}   {cut(b)}/{cut(l)}".ljust(104) + f"{b['mean_tokens']}/{l['mean_tokens']}")
    p = report["paired"]
    by_arm = p.get("unpaired_by_arm", {})
    out.append(f"paired replies {p['overall']['n']}, unpaired {p['unpaired']}" + (f" (base only {by_arm['base']}, adapter only {by_arm['lora']}: in no figure above)" if p["unpaired"] else "") + f"; tasks answered {report['tasks_answered']} of {report['tasks_in_files']} in the held-out files")
    if p["overall"].get("per_task", 0) > 1:
        out.append(f"k > 1 (up to {p['overall']['per_task']} replies per task): the sign p is over TASKS, the arm with more accepted replies of a task wins it, ties are dropped; the replies of a task are not independent draws")
    return out


def main(argv=None) -> None:
    from distill.domains import load
    ap = argparse.ArgumentParser(description="Score the base and adapter replies of a lora_eval.py run against the held-out truth")
    ap.add_argument("--domain", default="rea", help="the subject: a package of distill/domains (default %(default)s)")
    ap.add_argument("--heldout", type=pathlib.Path, nargs="+", required=True, help="the held-out task files the run was built from (the truth is read here)")
    ap.add_argument("--base", type=pathlib.Path, required=True, help="replies-base.jsonl")
    ap.add_argument("--lora", type=pathlib.Path, required=True, help="replies-lora.jsonl")
    ap.add_argument("--run-info", type=pathlib.Path, default=None, help="the run's run.json: its training summary heads the table")
    ap.add_argument("--out", type=pathlib.Path, default=None, help="also write the report as JSON here")
    ap.add_argument("--force", action="store_true", help="overwrite --out")
    a = ap.parse_args(argv)
    if a.out:
        outputs.refuse_overwrite([a.out], a.force)
    report = score(a.base, a.lora, a.heldout, domain=load(a.domain))
    info = score_replies.read_run_info(a.run_info) if a.run_info else None
    print("\n".join(lines_of(report, info)))
    if a.out:
        a.out.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
