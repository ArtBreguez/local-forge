"""Command line front end."""
import argparse

from . import commands, storage


def build_parser():
    parser = argparse.ArgumentParser(prog="todo")
    parser.add_argument("--file", default="todo.json", help="JSON file that holds the tasks")
    sub = parser.add_subparsers(dest="cmd", required=True)
    add = sub.add_parser("add", help="add a task")
    add.add_argument("title")
    add.add_argument("--tag", action="append", default=[], help="tag the task (repeatable)")
    done = sub.add_parser("done", help="mark a task as done")
    done.add_argument("id", type=int)
    lst = sub.add_parser("list", help="list tasks")
    lst.add_argument("--pending", action="store_true", help="hide finished tasks")
    lst.add_argument("--tag", default=None, help="only tasks with this tag")
    return parser


def main(argv=None, out=print):
    args = build_parser().parse_args(argv)
    tasks = storage.load(args.file)
    if args.cmd == "add":
        task = commands.add_task(tasks, args.title, tags=args.tag)
        storage.save(args.file, tasks)
        out(commands.format_task(task))
    elif args.cmd == "done":
        try:
            task = commands.complete_task(tasks, args.id)
        except KeyError:
            out(f"no task {args.id}")
            return 1
        storage.save(args.file, tasks)
        out(commands.format_task(task))
    elif args.cmd == "list":
        for task in commands.list_tasks(tasks, show_done=not args.pending, tag=args.tag):
            out(commands.format_task(task))
    return 0
