function flatten(arr, depth = 1) {
  if (!Array.isArray(arr)) throw new TypeError('array expected');
  if (typeof depth !== 'number' || Number.isNaN(depth) || depth < 0) throw new RangeError('depth must be >= 0');
  const out = [];
  const walk = (items, level) => {
    for (const item of items) {
      if (Array.isArray(item) && level > 0) walk(item, level - 1);
      else out.push(item);
    }
  };
  walk(arr, depth + 1);
  return out;
}

module.exports = { flatten };
