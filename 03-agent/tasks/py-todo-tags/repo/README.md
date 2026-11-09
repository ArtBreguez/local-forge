# todo

A tiny todo list that keeps its tasks in a JSON file.

    python -m todo --file tasks.json add "Buy milk"
    python -m todo --file tasks.json list --pending
    python -m todo --file tasks.json done 1

Layout: `todo/models.py` (the `Task` class and its JSON form), `todo/storage.py` (load and save the file),
`todo/commands.py` (the operations on a list of tasks, and how a task is printed), `todo/cli.py` (argument parsing;
`main(argv, out=print)` returns the exit code).
