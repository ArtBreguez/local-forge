function groupBy(items, keyFn) {
  if (!Array.isArray(items)) throw new TypeError('items must be an array');
  if (typeof keyFn !== 'function') throw new TypeError('keyFn must be a function');
  const out = new Map();
  items.forEach((item) => {
    const key = keyFn(item);
    if (!out.get(key)) out.set(key, []);
    out.get(key).push(item);
  });
  return out;
}

module.exports = { groupBy };
