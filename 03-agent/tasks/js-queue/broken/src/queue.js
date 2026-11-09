// A queue of named tasks: highest priority first, first in first out among equal priorities.
class TaskQueue {
  constructor() {
    this.tasks = [];
  }

  push(name, priority = 0) {
    if (!Number.isInteger(priority)) throw new RangeError('priority must be an integer');
    let i = this.tasks.length;
    while (i > 0 && this.tasks[i - 1].priority < priority) i--;
    this.tasks.splice(i, 0, { name, priority });
    return this;
  }

  pop() {
    const task = this.tasks.shift();
    return task ? task.name : undefined;
  }

  get size() {
    return this.tasks.length;
  }

  items() {
    return this.tasks.map((task) => ({ name: task.name, priority: task.priority }));
  }
}

module.exports = { TaskQueue };
