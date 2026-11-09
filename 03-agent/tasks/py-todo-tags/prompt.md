`todo/` is a small command line todo list (read `README.md` and the code first). Add tags to it:

- A task has a list of tags (`task.tags`, default empty). `Task.to_dict()` includes it under `"tags"`; `Task.from_dict()` must still load old data that has no `"tags"` key (empty list).
- `commands.add_task(tasks, title, tags=())`: tags are stripped and lower-cased, duplicates are dropped (the first occurrence keeps its place), and an empty tag raises `ValueError`.
- `commands.list_tasks(tasks, show_done=True, tag=None)`: with `tag` given, only tasks that have that tag (compared lower-case) are returned.
- `commands.format_task` appends the tags, each as ` #tag`: `[ ] 1 Buy milk #home #errand`.
- Command line: `add TITLE` accepts `--tag TAG` any number of times; `list` accepts `--tag TAG` (it can be combined with `--pending`).

The visible tests are in `tests/` (run them with `python -m unittest`); more tests that you cannot see check the same behaviour. Do not modify anything under `tests/`.
