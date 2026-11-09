import json
import os
import tempfile
import unittest

from todo import commands, storage
from todo.cli import main
from todo.models import Task


class TagsHiddenTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.path = os.path.join(self.dir.name, "t.json")

    def run_cli(self, *argv):
        lines = []
        code = main(["--file", self.path, *argv], out=lines.append)
        return code, lines

    def test_model_roundtrip_and_old_data(self):
        t = Task(1, "a")
        self.assertEqual(t.tags, [])
        t.tags.append("x")
        self.assertEqual(t.to_dict()["tags"], ["x"])
        self.assertEqual(Task.from_dict({"id": 2, "title": "b"}).tags, [])
        self.assertEqual(Task.from_dict(t.to_dict()).tags, ["x"])
        self.assertIsNot(Task(1, "a").tags, Task(2, "b").tags)

    def test_old_json_file_without_tags_still_loads_and_saves(self):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump([{"id": 1, "title": "old", "done": False}], f)
        self.assertEqual(self.run_cli("list"), (0, ["[ ] 1 old"]))
        self.run_cli("add", "new", "--tag", "x")
        saved = json.load(open(self.path, encoding="utf-8"))
        self.assertEqual([item["tags"] for item in saved], [[], ["x"]])

    def test_add_task_normalisation(self):
        tasks = []
        t = commands.add_task(tasks, "t", tags=("B", "a", " b ", "A"))
        self.assertEqual(t.tags, ["b", "a"])
        self.assertEqual(commands.add_task(tasks, "u").tags, [])
        for bad in ("", "   "):
            with self.assertRaises(ValueError):
                commands.add_task(tasks, "v", tags=["ok", bad])
        self.assertEqual(len(tasks), 2)

    def test_list_filter_is_case_insensitive_and_composes_with_pending(self):
        tasks = []
        commands.add_task(tasks, "a", tags=["Work"])
        commands.add_task(tasks, "b", tags=["work", "home"])
        commands.add_task(tasks, "c")
        commands.complete_task(tasks, 1)
        self.assertEqual([t.id for t in commands.list_tasks(tasks, tag="WORK")], [1, 2])
        self.assertEqual([t.id for t in commands.list_tasks(tasks, show_done=False, tag="work")], [2])
        self.assertEqual([t.id for t in commands.list_tasks(tasks, tag="none")], [])
        self.assertEqual([t.id for t in commands.list_tasks(tasks)], [1, 2, 3])

    def test_format_with_and_without_tags(self):
        t = Task(4, "x", True)
        self.assertEqual(commands.format_task(t), "[x] 4 x")
        t.tags = ["a", "b"]
        self.assertEqual(commands.format_task(t), "[x] 4 x #a #b")

    def test_cli_end_to_end(self):
        self.run_cli("add", "Buy milk", "--tag", "Home", "--tag", "errand")
        self.run_cli("add", "Report", "--tag", "work")
        self.run_cli("add", "Plain")
        self.run_cli("done", "2")
        self.assertEqual(self.run_cli("list", "--tag", "HOME"), (0, ["[ ] 1 Buy milk #home #errand"]))
        self.assertEqual(self.run_cli("list", "--tag", "work"), (0, ["[x] 2 Report #work"]))
        self.assertEqual(self.run_cli("list", "--pending", "--tag", "work"), (0, []))
        self.assertEqual(self.run_cli("list")[1][2], "[ ] 3 Plain")
