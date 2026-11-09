function formatBytes(n) {
  if (typeof n !== 'number' || !Number.isFinite(n)) throw new TypeError('finite number expected');
  if (n < 0 || !Number.isInteger(n)) throw new RangeError('non-negative integer expected');
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  if (n < 1024) return `${n} B`;
  let value = n;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  let rounded = Math.round(value * 10) / 10;
  if (rounded >= 1024 && unit < units.length - 1) {
    rounded = Math.round((rounded / 1024) * 10) / 10;
    unit += 1;
  }
  return `${rounded} ${units[unit]}`;
}

module.exports = { formatBytes };
