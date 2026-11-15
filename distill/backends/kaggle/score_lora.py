"""Score what `lora_eval.py` brought back: the base student and base + adapter on the same held-out tasks, with the truth that stayed on this machine.

    python3 -m distill.backends.kaggle.score_lora --domain rea --config ~/work/kaggle-lora/data/lora-config.json --heldout results/phase6/rea-data/decompiled-seed1001-all.jsonl \\
        --base ~/work/kaggle-lora/out/replies-base.jsonl --lora ~/work/kaggle-lora/out/replies-lora.jsonl [--run-info ~/work/kaggle-lora/out/run.json] [--out report.json]

WHAT is scored is the build's, not the replies files': `--config` is the `lora-config.json` the kernel ran (`build_lora.py` wrote it), and only the tasks of its evaluation set (`built_from.eval.ids`: the held-out
tasks that share no content with a training task, after the floor and the quota) and only the replies written under its two keys enter a figure: `adapter_sha` (the adapter that answered: the training records and
the training settings) and `eval_sha` (the evaluation the reply belongs to: that adapter, the exclusion, the tasks asked, the reply budget and the sampling). A smoke build has keys of its own, so a smoke run's replies
are another adapter's for a full build, and with the smoke build's own config the report starts with the smoke warning. A replies file can hold more (an output attached with `--resume-from` that was
made by a build with other `--train-tasks` or another quota): a reply of a task outside the set is counted and printed ("N replies of tasks outside the evaluation set, not scored"), and so is a reply of
another adapter and a reply of this adapter written under another evaluation key. The report prints both keys. A `--heldout` file that lacks a task of the set, or a run.json of another build, is refused.
The config's word is not taken for its keys: both are computed again from its settings and the shas it records of the two files of the dataset (`lora_eval.check_keys`, the kernel's own check), and a config
that was edited after the build (a smoke build's with `smoke` set to false, another budget) is refused, as the kernel refuses it.

A reply is judged by the code `kaggle-score` runs: `score_replies.reply_of` (a cut-off reply ends in the truncation marker, a failed request is an error) and `traces.verdict`, whose last step is the subject's
own `domain.verify` (REA: `scoring.score_strict`, the whitelist grammar on the text after the last `Answer:` line). A reply that answered another prompt than the task's now (its `prompt_sha`) is refused
(SystemExit; `kaggle-score` skips and counts such a reply): the prompts are in the evaluation key (the bytes of the build's `eval_tasks.jsonl`), so a reply that carries the key and another prompt means the `--heldout` files, or the
subject's prompt code, are not the build's. The FAMILY of a row is the task's in the held-out file (where the truth is), never what the replies file says. For the generated families `scoring.score` (the evaluation scorer of rea-bench's held-out arm) and `score_strict` are the same function; they differ only for the legacy templates.

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

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for _p in (ROOT, HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
from distill import outputs, score_replies, traces  # noqa: E402
import lora_eval  # noqa: E402  (the kernel's own `resolve_config` and `check_keys`: the keys of a config are computed again here as the kernel computes them)

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


KEYS = ("adapter_sha", "eval_sha")          # the training identity and the evaluation identity of a build (`lora_eval.adapter_key`, `lora_eval.eval_key`)


def read_build(path) -> dict:
    """{"ids": the ids of the evaluation set, "adapter_sha", "eval_sha", "smoke"} of a build's `lora-config.json` (`build_lora.make_config`). SystemExit for a file that is not one: without the set and the two
    keys there is nothing that says which replies may be scored. `smoke`: the build is a smoke run's (its keys are its own: only smoke replies carry them, and the report says what they are worth).
    The config's word is not taken for its keys, nor for `smoke`, which is a setting in both: the two keys are computed again from its settings and the shas it records of `train.jsonl` and
    `eval_tasks.jsonl`, by the kernel's own check (`lora_eval.check_keys`, here without the dataset), and a config whose keys are not the keys of its settings is SystemExit, as it is in the kernel: a
    smoke build's config with `smoke` edited to false would otherwise have its smoke replies scored (they carry the keys the file still has) with no warning."""
    try:
        data = json.loads(pathlib.Path(path).read_text())
    except (OSError, json.JSONDecodeError):
        data = None
    built = data.get("built_from") if isinstance(data, dict) else None
    ev = built.get("eval") if isinstance(built, dict) else None
    ids, keys = (ev.get("ids") if isinstance(ev, dict) else None), {k: data.get(k) if isinstance(data, dict) else None for k in KEYS}
    if not (isinstance(ids, list) and ids and all(isinstance(i, str) for i in ids) and all(isinstance(v, str) and v for v in keys.values())):
        raise SystemExit(f"{path} is not the lora-config.json of a build_lora build (an `adapter_sha`, an `eval_sha` and the ids of its evaluation set, `built_from.eval.ids`): build again with this version, the set is what is scored")
    try:
        cfg = lora_eval.resolve_config(data)
        lora_eval.check_keys(cfg)
    except (ValueError, TypeError) as e:
        raise SystemExit(f"{path} is not the lora-config.json of a build_lora build (the kernel would not take its settings: {e})")
    except SystemExit as e:
        raise SystemExit(f"{path}: {e}")
    return {"ids": set(ids), **keys, "smoke": cfg["smoke"] is True}


def read_arm(path, tasks: dict, *, domain, build: dict) -> dict:
    """{"rows": {(task id, sample index): {"kind", "verdict", "finish", "tokens"}}, "info": counts, "outside": ids} of one arm's replies file. `tasks` is the evaluation set ({id: task}): `kind` is the
    task's own family (the held-out file's), whatever the `kind` of the replies line says, so a line that names another family never moves a reply into that family's row. The replies of a task
    that is not in it are counted (`info["outside"]`, the task ids in `outside`) and never scored. A missing or malformed entry is counted and skipped. A reply is scored only when it carries BOTH keys of
    `build` (`read_build`): one of another adapter, or that does not say (`info["other_adapter"]`), and one of this adapter written under another evaluation key, or that does not say (`info["other_eval"]`:
    another exclusion, another set of tasks or another prompt of a task; a resume asks it again, and it stays in the file when the deadline came first), are counted and skipped, before anything else of the reply is looked at.
    A reply with no fingerprint is skipped as unchecked, and a reply that answered another prompt is SystemExit (`kaggle-score` skips and counts one; here it carries the build's evaluation key, which holds the prompts)."""
    lines, info = score_replies.read_replies(path)
    info = collections.Counter(info)
    rows, outside = {}, []
    for tid, rec in lines.items():
        task = tasks.get(tid)
        if task is None:
            info["outside"] += sum(isinstance(out, dict) for out in rec["outs"])
            outside.append(tid)
            continue
        sha = outputs.prompt_sha(domain.messages(task, "thinking"))
        for j, (out, own) in enumerate(zip(rec["outs"], score_replies.reply_fingerprints(rec))):
            if not isinstance(out, dict):
                info["missing"] += 1
                continue
            if out.get("adapter_sha") != build["adapter_sha"]:
                info["other_adapter"] += 1
                continue
            if out.get("eval_sha") != build["eval_sha"]:
                info["other_eval"] += 1
                continue
            if own is None:
                info["unchecked"] += 1
                continue
            if own != sha:
                raise SystemExit(f"{path}: the reply {j} of {tid} answered another prompt than the task's now (fingerprint {own}, now {sha}): score the run that was built from these held-out tasks")
            verdict = traces.verdict(score_replies.reply_of(out, thinking=True), task, domain=domain)
            rows[(tid, j)] = {"kind": task["kind"], "verdict": verdict, "finish": out.get("finish"), "tokens": out.get("tokens") if isinstance(out.get("tokens"), (int, float)) else 0}
    return {"rows": rows, "info": dict(info), "outside": sorted(outside)}


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


def score(base_path, lora_path, heldout_files, *, domain, build: dict) -> dict:
    """The report over the evaluation set of `build` (`read_build`: its ids and its two keys) and nothing else: per arm `summarize` of the replies BOTH arms have and the read counts, the `pair`, how many tasks
    of the set were answered, the two keys, and what was NOT scored: `outside` (the replies, per arm, of tasks that are not in the set, and their ids: of either arm), `other_adapter` (the replies of another
    adapter) and `other_eval` (the replies of this adapter written under another evaluation key). The truth is read from the held-out files; SystemExit when one of the set's tasks is not in them (they are not
    the files the build was made from)."""
    held = {t["id"]: t for t in domain.load_heldout([pathlib.Path(f) for f in heldout_files])}
    absent = sorted(build["ids"] - set(held))
    if absent:
        raise SystemExit(f"{len(absent)} task(s) of the build's evaluation set are not in --heldout ({', '.join(absent)[:200]}): give the held-out files the build was made from")
    tasks = {tid: held[tid] for tid in sorted(build["ids"])}
    arms = {arm: read_arm(path, tasks, domain=domain, build=build) for arm, path in (("base", base_path), ("lora", lora_path))}
    common = set(arms["base"]["rows"]) & set(arms["lora"]["rows"])
    apart = lambda key: {"replies": sum(v["info"].get(key, 0) for v in arms.values()), "by_arm": {a: v["info"].get(key, 0) for a, v in arms.items()}}
    return {"arms": {a: {**summarize({k: r for k, r in v["rows"].items() if k in common}), "read": v["info"], "unpaired": len(v["rows"]) - len(common)} for a, v in arms.items()},
            "paired": pair(arms["base"]["rows"], arms["lora"]["rows"]), "tasks_answered": len({tid for v in arms.values() for tid, _ in v["rows"]}), "tasks_in_set": len(tasks),
            **{k: build[k] for k in KEYS}, "smoke": bool(build.get("smoke")), "outside": {**apart("outside"), "tasks": sorted({tid for v in arms.values() for tid in v["outside"]})}, "other_adapter": apart("other_adapter"),
            "other_eval": apart("other_eval")}


def check_run(run_info, build: dict, path="run.json") -> None:
    """SystemExit when the run.json given is of another build than `--config`: its `config.adapter_sha` is another one (the training summary that heads the table would be another adapter's) or its
    `config.eval_sha` is (the session asked another evaluation than the one that is scored). A run.json that does not say (none given, or one with no config or without that key) is not judged here; the
    replies say it themselves, one by one."""
    ran = run_info.get("config") if isinstance(run_info, dict) and isinstance(run_info.get("config"), dict) else {}
    for key, what in (("adapter_sha", "adapter key"), ("eval_sha", "evaluation key")):
        if ran.get(key) and ran[key] != build[key]:
            raise SystemExit(f"{path} is the run of {what} {ran[key]}, and --config has {what} {build[key]}: give the lora-config.json the kernel ran (the dataset's)")


def lines_of(report: dict, run_info: dict = None) -> list:
    """The table printed: the smoke warning when the build is a smoke run's (the config says so, with or without a run.json), a header about the adapter's training (from run.json), then one row per family
    and the overall row."""
    out = []
    if report.get("smoke") or (run_info or {}).get("smoke"):
        out.append(f"WARNING: this is a SMOKE run ({lora_eval.DEFAULTS['smoke_steps']} steps, one batch of replies): its numbers say nothing about the adapter")
    if run_info:
        t = run_info.get("train") or {}
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
    out.append(f"paired replies {p['overall']['n']}, unpaired {p['unpaired']}" + (f" (base only {by_arm['base']}, adapter only {by_arm['lora']}: in no figure above)" if p["unpaired"] else "") + f"; tasks answered {report['tasks_answered']} of {report['tasks_in_set']} in the evaluation set of the build")
    out.append(f"adapter key {report['adapter_sha']} (the training records and the training settings), evaluation key {report['eval_sha']} (that adapter, the exclusion, the tasks asked, the reply budget and the sampling): only the replies that carry both are scored")
    for key, what in (("outside", "of tasks outside the evaluation set"), ("other_adapter", f"of another adapter than {report['adapter_sha']} (another --sft or training setting, or a smoke run's)"),
                      ("other_eval", f"of this adapter written under another evaluation key than {report['eval_sha']} (another exclusion, set of tasks, prompt, reply budget, sampling or k: a resume asks them again)")):
        c = report[key]
        if key == "outside" or c["replies"]:
            out.append(f"{c['replies']} replies {what}, not scored" + (f" (base {c['by_arm']['base']}, adapter {c['by_arm']['lora']}" + (f"; tasks: {', '.join(c['tasks'])[:300]}" if c.get("tasks") else "") + ")" if c["replies"] else ""))
    if p["overall"].get("per_task", 0) > 1:
        out.append(f"k > 1 (up to {p['overall']['per_task']} replies per task): the sign p is over TASKS, the arm with more accepted replies of a task wins it, ties are dropped; the replies of a task are not independent draws")
    return out


def main(argv=None) -> None:
    from distill.domains import load
    ap = argparse.ArgumentParser(description="Score the base and adapter replies of a lora_eval.py run against the held-out truth")
    ap.add_argument("--domain", default="rea", help="the subject: a package of distill/domains (default %(default)s)")
    ap.add_argument("--config", type=pathlib.Path, required=True, help="the build's data/lora-config.json (the one the kernel ran): only the tasks of its evaluation set, and only the replies written under its adapter key and its evaluation key, are scored; a config whose keys are not the keys of its settings (edited after the build) is refused")
    ap.add_argument("--heldout", type=pathlib.Path, nargs="+", required=True, help="the held-out task files the run was built from (the truth is read here)")
    ap.add_argument("--base", type=pathlib.Path, required=True, help="replies-base.jsonl")
    ap.add_argument("--lora", type=pathlib.Path, required=True, help="replies-lora.jsonl")
    ap.add_argument("--run-info", type=pathlib.Path, default=None, help="the run's run.json: its training summary heads the table")
    ap.add_argument("--out", type=pathlib.Path, default=None, help="also write the report as JSON here")
    ap.add_argument("--force", action="store_true", help="overwrite --out")
    a = ap.parse_args(argv)
    if a.out:
        outputs.refuse_overwrite([a.out], a.force)
    build = read_build(a.config)
    info = score_replies.read_run_info(a.run_info) if a.run_info else None
    check_run(info, build, a.run_info)
    report = score(a.base, a.lora, a.heldout, domain=load(a.domain), build=build)
    print("\n".join(lines_of(report, info)))
    if a.out:
        a.out.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
