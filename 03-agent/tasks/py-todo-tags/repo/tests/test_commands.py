import unittest

from todo import commands
from todo.models import Task


class CommandTests(unittest.TestCase):
    def test_add_assigns_increasing_ids(self):
        tasks = []
        a = commands.add_task(tasks, " Buy milk ")
        b = commands.add_task(tasks, "Call Bob")
        self.assertEqual((a.id, a.title, b.id), (1, "Buy milk", 2))

    def test_empty_title_is_rejected(self):
        with self.assertRaises(ValueError):
            commands.add_task([], "   ")

    def test_complete_and_list(self):
        tasks = [Task(1, "a"), Task(2, "b")]
        commands.complete_task(tasks, 1)
        self.assertEqual([t.id for t in commands.list_tasks(tasks, show_done=False)], [2])
        with self.assertRaises(KeyError):
            commands.complete_task(tasks, 9)

    def test_format(self):
        self.assertEqual(commands.format_task(Task(3, "x", True)), "[x] 3 x")

    def test_tags_are_normalised(self):
        tasks = []
        t = commands.add_task(tasks, "Buy milk", tags=[" Home ", "errand", "HOME"])
        self.assertEqual(t.tags, ["home", "errand"])
        self.assertEqual(commands.format_task(t), "[ ] 1 Buy milk #home #errand")
