function formatBytes(n) {
  if (typeof n !== 'number' || !Number.isFinite(n)) throw new TypeError('finite number expected');
  if (n < 0 || !Number.isInteger(n)) throw new RangeError('non-negative integer expected');
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  let value = n;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${unit === 0 ? value : value.toFixed(1)} ${units[unit]}`;
}

module.exports = { formatBytes };
