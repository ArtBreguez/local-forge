import os
import tempfile
import unittest

from todo.cli import main


class CliTests(unittest.TestCase):
    def run_cli(self, *argv):
        lines = []
        code = main(["--file", self.path, *argv], out=lines.append)
        return code, lines

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.path = os.path.join(self.dir.name, "t.json")

    def test_add_list_done(self):
        self.assertEqual(self.run_cli("add", "Buy milk"), (0, ["[ ] 1 Buy milk"]))
        self.run_cli("add", "Call Bob")
        self.assertEqual(self.run_cli("done", "1"), (0, ["[x] 1 Buy milk"]))
        self.assertEqual(self.run_cli("list", "--pending"), (0, ["[ ] 2 Call Bob"]))
        self.assertEqual(self.run_cli("done", "7"), (1, ["no task 7"]))

    def test_tags_on_the_command_line(self):
        self.run_cli("add", "Buy milk", "--tag", "home", "--tag", "errand")
        self.run_cli("add", "Write report", "--tag", "work")
        code, lines = self.run_cli("list", "--tag", "home")
        self.assertEqual((code, lines), (0, ["[ ] 1 Buy milk #home #errand"]))
