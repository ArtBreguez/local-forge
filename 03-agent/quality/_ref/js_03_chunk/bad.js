function chunk(arr, size) {
  if (!Array.isArray(arr)) throw new TypeError('array expected');
  if (typeof size !== 'number') throw new TypeError('size must be a number');
  if (!Number.isInteger(size) || size < 1) throw new RangeError('size must be a positive integer');
  const out = [];
  for (let i = 0; i + size <= arr.length; i += size) out.push(arr.slice(i, i + size));
  return out;
}

module.exports = { chunk };
