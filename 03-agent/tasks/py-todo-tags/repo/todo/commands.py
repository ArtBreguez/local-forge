"""Operations on a list of tasks."""
from .models import Task


def add_task(tasks, title):
    title = title.strip()
    if not title:
        raise ValueError("title must not be empty")
    new_id = max((t.id for t in tasks), default=0) + 1
    task = Task(new_id, title)
    tasks.append(task)
    return task


def complete_task(tasks, task_id):
    for task in tasks:
        if task.id == task_id:
            task.done = True
            return task
    raise KeyError(task_id)


def list_tasks(tasks, show_done=True):
    return [t for t in tasks if show_done or not t.done]


def format_task(task):
    mark = "x" if task.done else " "
    return f"[{mark}] {task.id} {task.title}"
