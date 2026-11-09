"""Load and save the task list as a JSON file."""
import json
import os

from .models import Task


def load(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [Task.from_dict(item) for item in json.load(f)]


def save(path, tasks):
    with open(path, "w", encoding="utf-8") as f:
        json.dump([task.to_dict() for task in tasks], f, indent=2)
