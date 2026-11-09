"""The Task class."""


class Task:
    def __init__(self, id, title, done=False, tags=None):
        self.id = id
        self.title = title
        self.done = done
        self.tags = list(tags) if tags else []

    def to_dict(self):
        return {"id": self.id, "title": self.title, "done": self.done, "tags": list(self.tags)}

    @classmethod
    def from_dict(cls, data):
        return cls(data["id"], data["title"], data.get("done", False), data.get("tags", []))
