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


if __name__ == "__main__":
    unittest.main()
