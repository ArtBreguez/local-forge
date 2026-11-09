"""Operations on a list of tasks."""
from .models import Task


def normalise_tags(tags):
    result = []
    for tag in tags:
        tag = tag.strip().lower()
        if not tag:
            raise ValueError("a tag must not be empty")
        if tag not in result:
            result.append(tag)
    return result


def add_task(tasks, title, tags=()):
    title = title.strip()
    if not title:
        raise ValueError("title must not be empty")
    tags = normalise_tags(tags)
    new_id = max((t.id for t in tasks), default=0) + 1
    task = Task(new_id, title, tags=tags)
    tasks.append(task)
    return task


def complete_task(tasks, task_id):
    for task in tasks:
        if task.id == task_id:
            task.done = True
            return task
    raise KeyError(task_id)


def list_tasks(tasks, show_done=True, tag=None):
    wanted = tag.strip().lower() if tag is not None else None
    return [t for t in tasks if (show_done or not t.done) and (wanted is None or wanted in t.tags)]


def format_task(task):
    mark = "x" if task.done else " "
    tags = "".join(f" #{tag}" for tag in task.tags)
    return f"[{mark}] {task.id} {task.title}{tags}"
