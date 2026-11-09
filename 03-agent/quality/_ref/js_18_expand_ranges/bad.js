function expandRanges(spec) {
  if (typeof spec !== 'string') throw new TypeError('string expected');
  if (spec.trim() === '') return [];
  const out = [];
  for (const raw of spec.split(',')) {
    const part = raw.trim();
    const bits = part.split('-');
    if (bits.some((b) => !/^[0-9]+$/.test(b))) throw new SyntaxError(`bad part: ${part}`);
    const lo = Number(bits[0]);
    const hi = Number(bits[bits.length - 1]);
    for (let i = Math.min(lo, hi); i <= Math.max(lo, hi); i += 1) out.push(i);
  }
  return out.sort();
}

module.exports = { expandRanges };
