// A first-in first-out queue of named tasks.
class TaskQueue {
  constructor() {
    this.tasks = [];
  }

  push(name) {
    this.tasks.push({ name });
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
    return this.tasks.map((task) => ({ name: task.name }));
  }
}

module.exports = { TaskQueue };
