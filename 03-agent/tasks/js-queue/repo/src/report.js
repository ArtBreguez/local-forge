// Text report of a queue: one line per task, in the order in which they will be popped.
function describe(queue) {
  return queue
    .items()
    .map((task, i) => `${i + 1}. ${task.name}`)
    .join('\n');
}

module.exports = { describe };
