"""Round 14, fix 3 of the Kaggle thinking pilot: every cipher reply (6 of 6), 2 of 6 magic and 2 of 6 password replies hit the 4096-token cap (13% of all of them). A run config can give a family a budget of its own
(`max_tokens_by_family`); the Modal sample stage asks each task its own budget, sizes its slot for the longest of the run and counts what does not fit. The Kaggle kit is in tests/test_rea_thinking_kit.py.

  FamilyTokens   the run config: validation, flags, `server_plan`, the shipped thinking config
  ModalBudgets   the real `sample_answers` against a scripted server
  ModalWrapper   `modal_rea_data.py`: the flag, the run config, the call to the GPU function
  StageReports   the cut-off and the slot reports read a family against its own budget
"""
from __future__ import annotations

import json
import pathlib
import tempfile
import types
import unittest
from unittest import mock

from . import distill_fixtures as fx  # noqa: F401
from .test_rea_modal import ModalCase, StubbedSampleStage
from .thinking_pilot import THINKING_CONFIG

from distill import config, records  # noqa: E402
from distill.backends import modal_sample  # noqa: E402
from distill.domains.rea import DOMAIN  # noqa: E402
from .helpers import ROOT  # noqa: E402
from rea_bench import outputs as rea_outputs  # noqa: E402


class FamilyTokens(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def write(self, text):
        p = pathlib.Path(self.tmp.name) / "c.json"
        p.write_text(text)
        return p

    def test_the_default_is_no_override_and_a_run_without_one_is_described_as_it_was(self):
        c = config.RunConfig(mode="thinking", max_len=8192)
        self.assertEqual(c.max_tokens_by_family, {})
        self.assertNotIn("max_tokens_by_family", c.resolved())
        self.assertNotIn("max_tokens_by_family", c.sample_kwargs())
        self.assertEqual(config.family_tokens_text({}), "")

    def test_a_config_with_a_budget_per_family_is_read_dumped_and_described(self):
        c = config.RunConfig.load(self.write('{"mode": "thinking", "max_len": 8192, "max_tokens_by_family": {"cipher": 8192}}'))
        self.assertEqual(c.max_tokens_by_family, {"cipher": 8192})
        self.assertEqual(config.RunConfig.load(self.write(c.dump())).dump(), c.dump())
        self.assertEqual(c.sample_kwargs()["max_tokens_by_family"], {"cipher": 8192})
        r = c.resolved()
        self.assertEqual((r["max_tokens"], r["max_tokens_by_family"], r["slots"], r["slot_context"]), (4096, {"cipher": 8192}, 2, 12288), "no quota: every family is in the run")
        q = config.RunConfig(mode="thinking", max_len=8192, quota="crc=2,structs=2", max_tokens_by_family={"cipher": 8192}).resolved()
        self.assertEqual((q["slots"], q["slot_context"]), (4, 8192), "a family the quota leaves out does not size the slot")
        q = config.RunConfig(mode="thinking", max_len=8192, quota="crc=2,cipher=2", max_tokens_by_family={"cipher": 8192}).resolved()
        self.assertEqual((q["slots"], q["slot_context"]), (2, 12288))

    def test_a_budget_that_is_not_a_number_of_tokens_or_not_for_this_mode_is_refused_before_anything_is_rented(self):
        for bad in ({"cipher": 0}, {"cipher": -1}, {"cipher": "8192"}, {"cipher": True}, {"cipher": 1.5}, {"": 5}, [("cipher", 1)], "cipher=8192"):
            with self.assertRaises(SystemExit, msg=repr(bad)):
                config.RunConfig(mode="thinking", max_len=8192, max_tokens_by_family=bad)
        with self.assertRaisesRegex(SystemExit, "reasoning or thinking run"):
            config.RunConfig(mode="plain", max_tokens_by_family={"cipher": 8192})
        with self.assertRaisesRegex(SystemExit, "no task kind"):
            config.check_family_tokens({"cipher": 8192, "cypher": 8192}, DOMAIN.kinds())

    def test_the_flag_form(self):
        self.assertEqual(config.parse_family_tokens("cipher=8192,magic=8192"), {"cipher": 8192, "magic": 8192})
        self.assertEqual(config.parse_family_tokens(""), {})
        self.assertEqual(config.family_tokens_text(config.parse_family_tokens("cipher=8192, magic = 6000")), "cipher=8192,magic=6000")
        for bad, why in (("cipher", "kind=N"), ("cipher=x", "kind=N"), ("cipher=0", "at least 1"), ("cipher=1,cipher=2", "twice"), ("nope=1", "no task kind")):
            with self.assertRaisesRegex(SystemExit, why):
                config.parse_family_tokens(bad, DOMAIN.kinds())

    def test_a_number_that_is_not_ascii_digits_is_a_message_not_a_traceback(self):
        """`str.isdigit()` accepts '²', '①' and the digits of other scripts; `int()` refuses the first two with a ValueError (round 15, m3). Both flag parsers take ASCII digits only."""
        for digit in ("\u00b2", "\u00b3", "\u2460", "\u0661\u0662", "\uff11\uff12", "8\u00b2"):
            with self.assertRaisesRegex(SystemExit, "kind=N", msg=digit):
                config.parse_family_tokens(f"cipher={digit}", DOMAIN.kinds())
            with self.assertRaisesRegex(SystemExit, "bad --quota entry", msg=digit):
                config.parse_quota(f"crc={digit}", DOMAIN.kinds())
        self.assertEqual(config.parse_family_tokens("cipher= 8192 ", DOMAIN.kinds()), {"cipher": 8192})
        self.assertEqual(config.parse_quota("crc=250, structs = 100", DOMAIN.kinds()), {"crc": 250, "structs": 100})

    def test_a_task_gets_the_budget_of_its_family_and_the_slot_is_sized_for_the_longest_in_the_run(self):
        by = {"cipher": 8192, "magic": 8192}
        self.assertEqual([config.task_max_tokens(k, 4096, by) for k in ("cipher", "crc", "magic")], [8192, 4096, 8192])
        self.assertEqual(config.task_max_tokens("cipher", 4096, None), 4096)
        self.assertEqual(config.run_max_tokens(["crc", "structs"], 4096, by), 4096, "a family that is not in the run costs nothing")
        self.assertEqual(config.run_max_tokens(["crc", "cipher"], 4096, by), 8192)
        self.assertEqual(config.run_max_tokens([], 4096, by), 4096)
        self.assertEqual(config.run_max_tokens(["a"], 4096, {"a": 1024}), 1024)

    def test_the_server_plan_keeps_the_total_context_and_changes_nothing_without_a_longer_family(self):
        self.assertEqual(config.server_plan(4096), (4, 8192))
        self.assertEqual(config.server_plan(4096, 4096), (4, 8192))
        self.assertEqual(config.server_plan(4096, 2048), (4, 8192), "a shorter family needs no new slot")
        self.assertEqual(config.server_plan(900), (4, 8192))
        self.assertEqual(config.server_plan(4096, 8192), (2, 12288), "a prompt of up to 4096 tokens and a reply of 8192, two requests in flight")
        self.assertEqual(config.server_plan(4096, 6144), (3, 10240))
        for tokens, longest in ((4096, 8192), (4096, 6000), (768, 8192), (2048, 12000), (4096, 16384)):
            slots, ctx = config.server_plan(tokens, longest)
            self.assertLessEqual(slots * ctx, 4 * 8192, "never more memory than the 4 x 8192 every run has used")
            self.assertGreaterEqual(ctx, longest + (config.slot_context(tokens) - tokens), "the room the base left for a prompt is kept")
            self.assertEqual(ctx % 1024, 0)

    def test_the_shipped_thinking_config_gives_cipher_magic_and_password_8192_and_nothing_else(self):
        c = config.RunConfig.load(THINKING_CONFIG)
        self.assertEqual((c.max_tokens, c.max_tokens_by_family), (4096, {"cipher": 8192, "magic": 8192, "password": 8192}))
        for name in ("plain", "reasoning"):
            self.assertEqual(config.RunConfig.load(ROOT / "distill" / "configs" / f"rea-{name}.json").max_tokens_by_family, {})
        self.assertEqual(set(c.max_tokens_by_family) - set(DOMAIN.kinds()), set(), "every name is a family of the subject")
        full = config.RunConfig(**{**json.loads(THINKING_CONFIG.read_text()), "quota": "structs=10,cipher=10"}).resolved()
        self.assertEqual((full["slots"], full["slot_context"]), (2, 12288))

    def test_the_shipped_thinking_configs_window_holds_the_longest_record_its_slot_allows(self):
        """Round 15, m5: a record is trained only if prompt + reply + 32 fits `max_len`; the slot of a run with a cipher/magic/password quota allows 4096 + 8192 = 12288, so the config's window is that, not 8192
        (a reply the slot allowed would be generated, accepted, counted and then dropped by the length rule). modal_train takes any window: `batch_plan` is 1 x 16 from 8192 up."""
        c = config.RunConfig.load(THINKING_CONFIG)
        longest = max([c.max_tokens, *c.max_tokens_by_family.values()])
        self.assertEqual(c.max_len, 12288)
        self.assertGreaterEqual(c.max_len, config.slot_context(longest), "every record the slot allows fits the training window")
        self.assertEqual(config.server_plan(c.max_tokens, longest)[1], c.max_len)
        full = config.RunConfig(**{**json.loads(THINKING_CONFIG.read_text()), "quota": "structs=10,cipher=10"}).resolved()
        self.assertEqual(full["max_len"], full["slot_context"])
        self.assertEqual(config.merge_train_flags(THINKING_CONFIG, dict(config.TRAIN_FLAG_DEFAULTS))["max_len"], 12288, "modal_lora --run-config trains on the same window")

    def test_the_run_config_reaches_the_sample_stage_as_a_flag(self):
        flags = dict(config.FLAG_DEFAULTS)
        got = config.merge_flags(THINKING_CONFIG, flags, "rea")
        self.assertEqual(got["max_tokens_by_family"], "cipher=8192,magic=8192,password=8192")
        self.assertEqual(config.merge_flags(ROOT / "distill" / "configs" / "rea-plain.json", flags, "rea")["max_tokens_by_family"], "")
        with self.assertRaisesRegex(SystemExit, "do not also give --max-tokens-by-family"):
            config.merge_flags(THINKING_CONFIG, {**flags, "max_tokens_by_family": "cipher=1"}, "rea")


class ModalBudgets(StubbedSampleStage):
    """The real `sample_answers` against a scripted server: the 12 legacy tasks of the harness, two of them `password`."""

    def run_thinking(self, popen=None, **kw):
        args = dict(reasoning=True, thinking=True, max_tokens=4096, max_len=8192, max_tokens_by_family={"password": 8192})
        args.update(kw)
        started = []
        recorder = lambda cmd, *a, **k: (started.append(cmd), types.SimpleNamespace(kill=lambda: None))[1]
        with mock.patch.object(modal_sample.subprocess, "Popen", popen or recorder):
            res = self.run_stage(False, **args)
        self.started = started
        return res

    def test_each_request_asks_its_own_tasks_budget(self):
        self.run_thinking()
        by_kind = {}
        for body in self.bodies:
            tid = next(i for i in self.tasks if i in body["messages"][-1]["content"])
            by_kind.setdefault(self.tasks[tid]["kind"], set()).add(body["max_tokens"])
        self.assertEqual(by_kind["password"], {8192})
        self.assertEqual({k: v for k, v in by_kind.items() if k != "password"}, {k: {4096} for k in by_kind if k != "password"})
        self.assertGreater(len(by_kind), 5)

    def test_the_server_is_started_for_the_longest_budget_in_the_same_total_context(self):
        self.run_thinking()
        cmd = self.started[0]
        self.assertEqual((cmd[cmd.index("-c") + 1], cmd[cmd.index("--parallel") + 1]), (str(2 * 12288), "2"))
        self.run_thinking(max_tokens_by_family={})
        cmd = self.started[0]
        self.assertEqual((cmd[cmd.index("-c") + 1], cmd[cmd.index("--parallel") + 1]), (str(4 * 8192), "4"), "no family with a longer budget: the 4 x 8192 it always was")
        self.run_thinking(max_tokens_by_family={"password": 2048})
        cmd = self.started[0]
        self.assertEqual((cmd[cmd.index("-c") + 1], cmd[cmd.index("--parallel") + 1]), (str(4 * 8192), "4"), "a shorter budget needs no longer slot")

    def test_a_family_that_no_task_has_does_not_resize_the_server(self):
        self.run_thinking(max_tokens_by_family={"cipher": 8192})
        cmd = self.started[0]
        self.assertEqual((cmd[cmd.index("-c") + 1], cmd[cmd.index("--parallel") + 1]), (str(4 * 8192), "4"))

    def test_the_stats_record_the_budgets_the_slots_and_the_room(self):
        st = self.run_thinking()["stats"]
        self.assertEqual(st["sampling"]["max_tokens_by_family"], {"password": 8192})
        self.assertEqual((st["sampling"]["max_tokens"], st["sampling"]["slot_context"], st["sampling"]["parallel"]), (4096, 12288, 2))
        self.assertEqual(st["slot"]["max_prompt_tokens_by_kind"], {"password": 12288 - 8192})
        st = self.run_thinking(max_tokens_by_family={})["stats"]
        self.assertNotIn("max_tokens_by_family", st["sampling"])
        self.assertNotIn("max_prompt_tokens_by_kind", st["slot"])

    def test_a_prompt_that_leaves_less_than_its_own_budget_is_not_asked_and_is_counted_under_its_family(self):
        src = self.root / "data" / rea_outputs.decompiled_stem(self.tag)
        for tid in ("password-7-0001", "xor-key-7-0000"):                # the same 4,200-word prompt: it fits 4096 + 4,200 in a 12288 slot, not 8192 + 4,200
            p = src / f"{tid}.json"
            t = json.loads(p.read_text())
            p.write_text(json.dumps({**t, "pseudocode": f"/* {tid} */ " + "x " * 4200}))
        res = self.run_thinking()
        asked = {tid for tid, _i in self.requests}
        self.assertNotIn("password-7-0001", asked)
        self.assertIn("xor-key-7-0000", asked, "the same prompt with the family's default budget fits")
        self.assertIn("password-7-0011", asked)
        self.assertEqual((res["stats"]["slot"]["prompt_too_long"], res["stats"]["slot"]["by_kind"]), (1, {"password": 1}))
        self.assertIn("1 task(s) not asked", self.printed)
        self.assertIn("password=8192", self.printed)
        self.assertRegex(" ".join(records.reasoning_report(res["stats"])), r"password 1 \(over 4,096 tokens\)")

    def test_a_visible_reasoning_run_with_a_family_budget_checks_the_fit_too(self):
        src = self.root / "data" / rea_outputs.decompiled_stem(self.tag)
        for tid in ("password-7-0001", "xor-key-7-0000"):                # 15,000 words: with 768 tokens it fits the 16384-token slot, with the family's 8192 it does not
            p = src / f"{tid}.json"
            p.write_text(json.dumps({**json.loads(p.read_text()), "pseudocode": f"/* {tid} */ " + "x " * 15000}))
        res = self.run_thinking(thinking=False, max_tokens=0, max_len=4096)
        asked = {tid for tid, _i in self.requests}
        self.assertEqual(("password-7-0001" in asked, "xor-key-7-0000" in asked), (False, True))
        self.assertEqual(res["stats"]["slot"] if "slot" in res["stats"] else None, None, "the slot block is the thinking run's; a visible-reasoning run prints the count instead")
        self.assertIn("1 task(s) not asked", self.printed)
        cmd = self.started[0]
        self.assertEqual((cmd[cmd.index("-c") + 1], cmd[cmd.index("--parallel") + 1]), (str(2 * 16384), "2"), "768 -> 8192: the 8192-token slot grows by 7424, to 16384")

    def test_a_budget_for_a_family_that_does_not_exist_or_for_a_plain_run_is_refused(self):
        with self.assertRaisesRegex(SystemExit, "no task kind"):
            self.run_thinking(max_tokens_by_family={"pasword": 8192})
        with self.assertRaisesRegex(SystemExit, "reasoning or thinking run"):
            self.run_stage(False, max_tokens_by_family={"password": 8192})

    def test_a_longer_budget_gets_a_longer_request_timeout(self):
        seen = []
        import urllib.request
        real = urllib.request.urlopen

        def spy(req, timeout=None):
            if not isinstance(req, str) and req.full_url.endswith("/chat/completions"):
                seen.append((json.loads(req.data)["max_tokens"], timeout))
            return real(req, timeout=timeout)
        with mock.patch.object(urllib.request, "urlopen", spy):
            self.run_thinking()
        self.assertEqual(sorted(set(seen)), [(4096, 600), (8192, 1200)])


class ModalWrapper(ModalCase):
    """`modal_rea_data.py`: the flag, the run config, and what the call to the GPU function carries."""

    def call(self, **kw):
        with tempfile.TemporaryDirectory() as d:
            remote, _run = self.run_main(pathlib.Path(d), stage="sample", seed=7, family="legacy", model="codeforge-27b-heretic-q4", k=4, reasoning=True, thinking=True, max_len=8192, **kw)
        return remote.call_args

    def test_the_flag_and_the_run_config_make_the_same_call(self):
        from_flag = self.call(max_tokens_by_family="cipher=8192,magic=8192,password=8192")
        self.assertEqual(from_flag.kwargs["max_tokens_by_family"], {"cipher": 8192, "magic": 8192, "password": 8192})
        self.assertIn("2 slots", self.printed, "the line printed before the GPU is rented says what the server will be")
        self.assertIn("cipher=8192,magic=8192,password=8192", self.printed)
        self.assertIsNone(self.call().kwargs["max_tokens_by_family"], "no flag, no override")
        self.assertNotIn("slots;", self.printed)

    def test_the_cost_line_of_a_run_with_a_longer_family_says_how_many_requests_are_in_flight_and_each_prompt_room(self):
        self.call(max_tokens_by_family="cipher=8192")
        line = next(x for x in self.printed.splitlines() if x.startswith("projected cost (thinking pilot)"))
        self.assertIn("2 requests in flight", line)
        self.assertIn("a prompt over 4,096 tokens is not asked and is counted for a family with a budget of 8,192, over 8,192 for the others", line, "4,096 is the room of the 8,192-token families only")
        self.assertIn("not 2: what that does to the speed is not measured", line)
        self.call()
        line = next(x for x in self.printed.splitlines() if x.startswith("projected cost (thinking pilot)"))
        self.assertIn("(a prompt over 4,096 tokens is not asked and is counted)", line, "a run without a map: the line it always had")
        self.assertNotIn("in flight", line)

    def test_the_cost_line_function_takes_the_map_and_the_slots(self):
        p = modal_sample.project(10, 2, 8192)
        plain = modal_sample.thinking_cost_line(p)
        self.assertNotIn("in flight", plain)
        self.assertEqual(modal_sample.thinking_cost_line(p, "", {"cipher": 8192}, 8192, 4), plain, "no family is longer than max_tokens: nothing to add")
        line = modal_sample.thinking_cost_line(p, "10 tasks", {"cipher": 8192}, 4096, 2)
        self.assertIn("one 12,288-token slot of the server, 2 requests in flight", line)
        self.assertIn("over 4,096 tokens is not asked and is counted for a family with a budget of 8,192, over 8,192 for the others", line)
        self.assertTrue(line.endswith("Tasks: 10 tasks."))

    def test_the_training_script_takes_the_window_of_the_shipped_config(self):
        """Round 15, m5: `modal_train` has no cap on `max_len`: 12288 is a micro-batch of one sequence (`batch_plan` 1 x 16, the memory of it not measured), a pair over it is dropped."""
        import sys
        sys.modules.pop("distill.backends.modal_train", None)
        import distill.backends.modal_train as modal_train
        self.assertEqual((modal_train.batch_plan(8192), modal_train.batch_plan(12288)), ((1, 16), (1, 16)))
        self.assertIsNone(modal_train.encode_ids([1, 2], list(range(12289)), 12288), "a pair over the window is dropped, never cut")
        self.assertIsNotNone(modal_train.encode_ids([1, 2], list(range(12288)), 12288))
        f = config.merge_train_flags(THINKING_CONFIG, dict(config.TRAIN_FLAG_DEFAULTS))
        self.assertEqual(f["max_len"], 12288)

    def test_the_shipped_thinking_config_reaches_the_call_with_its_map(self):
        with tempfile.TemporaryDirectory() as d:
            remote, _run = self.run_main(pathlib.Path(d), stage="sample", seed=7, family="legacy", run_config=str(THINKING_CONFIG))
        self.assertEqual(remote.call_args.kwargs["max_tokens_by_family"], {"cipher": 8192, "magic": 8192, "password": 8192})

    def test_a_typo_in_a_family_stops_the_run_before_a_gpu_is_rented(self):
        with self.assertRaisesRegex(SystemExit, "no task kind"):
            self.call(max_tokens_by_family="cypher=8192")

    def test_the_flag_is_not_for_a_plain_run(self):
        with tempfile.TemporaryDirectory() as d, self.assertRaisesRegex(SystemExit, "--reasoning or --thinking"):
            self.run_main(pathlib.Path(d), stage="sample", seed=7, family="legacy", model="codeforge-27b-heretic-q4", max_tokens_by_family="cipher=8192")


class StageReports(unittest.TestCase):
    def stats(self, truncated: dict, by_family=None, slot=None):
        by_kind = {k: {"replies": 6, "truncated": t} for k, t in truncated.items()}
        return {"reasoning": {"mode": "thinking", "replies": 6 * len(truncated), "accepted": 10, "wrong": 0, "no_answer_line": 0, "truncated": sum(truncated.values()), "error": 0, "teacher_records": 5,
                              "max_traces": 3, "by_kind": by_kind},
                "sampling": {"thinking": True, "max_tokens": 4096, "slot_context": 12288, "parallel": 2, **({"max_tokens_by_family": by_family} if by_family else {})},
                **({"slot": slot} if slot else {})}


    def test_a_family_with_a_budget_of_its_own_is_read_against_it(self):
        lines = records.reasoning_report(self.stats({"cipher": 3, "crc": 0, "structs": 0}, {"cipher": 8192}))
        warning = next(x for x in lines if x.startswith("WARNING: families"))
        self.assertIn("cipher 3/6 at 8192", warning)
        self.assertNotIn("crc", warning)
        again = records.reasoning_report(self.stats({"cipher": 0, "crc": 4}, {"cipher": 8192}))
        self.assertIn("crc 4/6 at 4096", next(x for x in again if x.startswith("WARNING: families")), "a family without an entry has the run's budget")
        none = records.reasoning_report(self.stats({"cipher": 3, "crc": 0}))
        self.assertFalse(any(x.startswith("WARNING: families") for x in none), "no map: the report is what it was")

    def test_the_cut_off_warning_is_the_owners_5_percent_and_5_exactly_is_not_more(self):
        """Round 15, m5: TRUNCATED_WARN was 10% where the owner's rule (and ERROR_RATE_WARN) is 5%; the pilot cut off 13%."""
        self.assertEqual(records.TRUNCATED_WARN, 0.05)
        warns = lambda cut, of: [x for x in records.reasoning_report({**self.stats({"crc": 0}), "reasoning": {**self.stats({"crc": 0})["reasoning"], "replies": of, "truncated": cut, "by_kind": {}}}) if x.startswith("WARNING: ") and "cut off" in x]
        self.assertEqual(warns(5, 100), [])
        self.assertEqual(len(warns(6, 100)), 1)
        self.assertIn("6% of the replies were cut off", warns(6, 100)[0])
        self.assertEqual(len(warns(13, 100)), 1, "the pilot's 13%")
        by_kind = lambda cut: {**self.stats({"cipher": cut, "crc": 0}, {"cipher": 8192}), "reasoning": {**self.stats({"cipher": cut, "crc": 0}, {"cipher": 8192})["reasoning"], "truncated": 0}}
        self.assertEqual([x for x in records.reasoning_report(by_kind(0)) if x.startswith("WARNING: families")], [])
        got = [x for x in records.reasoning_report(by_kind(1)) if x.startswith("WARNING: families")]
        self.assertEqual(len(got), 1, "1 of 6 replies is 17%")
        self.assertIn("more than 5% of their replies", got[0])

    def test_the_throughput_line_names_the_slots_the_run_had(self):
        perf = {"requests": 40, "completion_tokens": 100000, "wall_s": 5000, "tok_per_s": 20.0, "s_per_request": 125.0}
        line = lambda st: next(x for x in records.reasoning_report(st) if x.startswith("this call"))
        two = {**self.stats({"crc": 0}, {"cipher": 8192}), "perf": perf}
        self.assertIn("tokens/s over the 2 slots", line(two))
        four = {**two, "sampling": {**two["sampling"], "parallel": 4}}
        self.assertIn("tokens/s over the 4 slots", line(four))
        self.assertNotIn("four slots", line(two) + line(four))
        bare = {**two, "sampling": {k: v for k, v in two["sampling"].items() if k != "parallel"}}
        self.assertIn("tokens/s over the slots (", line(bare), "an older stats file does not say: no number is claimed")

    def test_the_slot_warning_names_each_familys_room(self):
        slot = {"slot_context": 12288, "max_prompt_tokens": 8192, "prompt_too_long": 3, "by_kind": {"cipher": 1, "crc": 2}, "max_prompt_tokens_by_kind": {"cipher": 4096}}
        line = next(x for x in records.reasoning_report(self.stats({"cipher": 0, "crc": 0}, {"cipher": 8192}, slot)) if "were not asked" in x)
        self.assertIn("cipher 1 (over 4,096 tokens), crc 2)", line)
        old = {k: v for k, v in slot.items() if k != "max_prompt_tokens_by_kind"}
        line = next(x for x in records.reasoning_report(self.stats({"crc": 0}, None, old)) if "were not asked" in x)
        self.assertTrue(line.endswith("(cipher 1, crc 2)"), line)


if __name__ == "__main__":
    unittest.main()
