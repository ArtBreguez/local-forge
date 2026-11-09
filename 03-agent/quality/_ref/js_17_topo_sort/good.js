function topoSort(graph) {
  const deps = new Map();
  const touch = (name) => {
    if (!deps.has(name)) deps.set(name, new Set());
  };
  for (const [name, list] of Object.entries(graph)) {
    touch(name);
    for (const dep of list) {
      touch(dep);
      deps.get(name).add(dep);
    }
  }
  const waiting = new Map([...deps].map(([name, set]) => [name, new Set(set)]));
  const dependents = new Map([...deps.keys()].map((name) => [name, []]));
  for (const [name, set] of deps) {
    for (const dep of set) dependents.get(dep).push(name);
  }
  const ready = [...waiting].filter(([, set]) => set.size === 0).map(([name]) => name);
  const order = [];
  while (ready.length > 0) {
    ready.sort();
    const name = ready.shift();
    order.push(name);
    for (const next of dependents.get(name)) {
      const set = waiting.get(next);
      set.delete(name);
      if (set.size === 0) ready.push(next);
    }
  }
  if (order.length !== deps.size) throw new Error('dependency cycle detected');
  return order;
}

module.exports = { topoSort };
