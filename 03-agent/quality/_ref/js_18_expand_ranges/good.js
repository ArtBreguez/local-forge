function expandRanges(spec) {
  if (typeof spec !== 'string') throw new TypeError('string expected');
  if (spec.trim() === '') return [];
  const out = new Set();
  for (const raw of spec.split(',')) {
    const part = raw.trim();
    let m = /^([0-9]+)$/.exec(part);
    if (m) {
      out.add(Number(m[1]));
      continue;
    }
    m = /^([0-9]+)-([0-9]+)$/.exec(part);
    if (!m) throw new SyntaxError(`bad part: ${JSON.stringify(part)}`);
    const lo = Number(m[1]);
    const hi = Number(m[2]);
    if (lo > hi) throw new RangeError(`descending range: ${part}`);
    for (let i = lo; i <= hi; i += 1) out.add(i);
  }
  return [...out].sort((a, b) => a - b);
}

module.exports = { expandRanges };
