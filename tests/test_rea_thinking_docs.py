"""Round 14: the READMEs name what the three fixes and the llama-bin reuse added (the commands are checked against the files they name by tests/test_rea_kaggle_kit.py)."""
from __future__ import annotations

import unittest

from .helpers import ROOT
from .thinking_pilot import KIT


class Readme(unittest.TestCase):
    def test_the_kernel_readme_has_the_commands_to_publish_and_attach_llama_bin(self):
        text = (KIT / "README.md").read_text()
        for part in ("kaggle kernels output", "llama-bin", "dataset-metadata.json", "kaggle datasets create", "kaggle datasets version", "--llama-dataset", "LD_LIBRARY_PATH", "--version", "max_tokens_by_family",
                     "server_geometry", "2 x 12288", "second pass", "two builds of the kit", "Not run on Kaggle"):
            self.assertIn(part, text)
        self.assertNotIn("access_token", text)

    def test_the_kaggle_readme_names_the_new_flags_and_the_pilot(self):
        text = (ROOT / "tools" / "kaggle" / "README.md").read_text()
        for part in ("--max-tokens-by-family", "--run-config", "--llama-dataset", "max_tokens_by_family", "89 of 114", "antidebug", "distill/backends/kaggle/README.md"):
            self.assertIn(part, text)
        for name in ("tools/kaggle/README.md", "distill/README.md"):
            self.assertIn("max_tokens_by_family", (ROOT / name).read_text())


class Names(unittest.TestCase):
    def test_every_test_file_a_readme_names_exists(self):
        """The thinking-pilot docs named `tests/test_rea_thinking_pilot.py`, which was never a file (round 15, m2)."""
        import re
        for md in [*(ROOT / "tools").rglob("README.md"), *(ROOT / "distill").rglob("README.md"), *(ROOT / "tests").rglob("README.md"), ROOT / "docs" / "TEACHER_CANDIDATES.md"]:
            for name in re.findall(r"tests/(?:test_[a-z0-9_]+\.py|thinking_pilot\.py|distill_fixtures\.py)", md.read_text()):
                self.assertTrue((ROOT / name).exists(), f"{md.relative_to(ROOT)} names {name}")

    def test_the_prompt_quoting_limit_and_its_hole_are_in_the_docs_and_in_the_code(self):
        """Round 15, m4: the repetition filter counts what the prompt contains against 12, and says what still passes (traces.repetition, tools/modal and tools/kaggle READMEs, distill/README.md)."""
        from distill import traces
        self.assertEqual(traces.PROMPT_REPEAT_MAX, 12)
        kaggle = (ROOT / "tools" / "kaggle" / "README.md").read_text()
        for part in ("PROMPT_REPEAT_MAX", "more than 12 times", "Why 12 and not 8", "83 teacher records", "The hole that is left, stated", "check 3: <a line of the code>", "length rule",
                     "in a number or in a word", "`alpha: <a line of the code>`", "7 words or fewer when the label word itself changes"):
            self.assertIn(part, kaggle)
        modal = next(l for l in (ROOT / "tools" / "modal" / "README.md").read_text().splitlines() if l.startswith("| `repetition` |"))
        for part in ("counted against 12", "What still passes, stated", "6 words or fewer", "PROMPT_REPEAT_MAX", "in a number or in a word", "7 words or fewer when the label word itself changes"):
            self.assertIn(part, modal)
        generic = next(l for l in (ROOT / "distill" / "README.md").read_text().splitlines() if l.startswith("| 5 traces |"))
        for part in ("counted against **12**", "6 words or fewer", "in a number (6 words or fewer) or in a word (7 words or fewer)"):
            self.assertIn(part, generic)
        self.assertNotIn("not counting what is in the task's own prompt", generic)

    def test_the_modal_readme_says_5_percent_and_the_window_of_the_shipped_config(self):
        text = (ROOT / "tools" / "modal" / "README.md").read_text()
        self.assertNotIn("warns when more than 10% were cut off", text)
        self.assertNotIn("the run warns above 10%", text)
        for part in ("warns when more than 5% were cut off", "the run warns above 5%", "The window of the shipped thinking config is 12288", "`max_len: 12288`", "not** the same by construction", "start with `--smoke`"):
            self.assertIn(part, text)

    def test_modal_rea_s_new_mount_is_said(self):
        """PR #32's body said modal_rea.py was left as is; it gained `distill` in its image (round 15, m6)."""
        self.assertIn('add_local_python_source("modal_qbench", "distill")', (ROOT / "tools" / "modal" / "modal_rea.py").read_text())
        for readme in ("tools/modal/README.md", "distill/README.md"):
            text = (ROOT / readme).read_text()
            self.assertIn("mounts `distill`", text, readme)
            self.assertIn("has not been run on Modal", text, readme)

    def test_the_filter_table_and_the_cost_bullet_say_what_the_run_does(self):
        text = (ROOT / "tools" / "modal" / "README.md").read_text()
        row = next(l for l in text.splitlines() if l.startswith("| `repetition` |"))
        self.assertIn("task's own prompt", row)
        self.assertIn("2 x 12,288", text)
        self.assertNotIn("over the four slots", text)


if __name__ == "__main__":
    unittest.main()
