function topoSort(graph) {
  const names = new Set();
  for (const [name, list] of Object.entries(graph)) {
    names.add(name);
    for (const dep of list) names.add(dep);
  }
  const state = new Map();
  const order = [];
  const visit = (name) => {
    if (state.get(name) === 'done') return;
    if (state.get(name) === 'active') throw new Error('dependency cycle detected');
    state.set(name, 'active');
    for (const dep of [...(graph[name] || [])].sort()) visit(dep);
    state.set(name, 'done');
    order.push(name);
  };
  for (const name of [...names].sort()) visit(name);
  return order;
}

module.exports = { topoSort };
