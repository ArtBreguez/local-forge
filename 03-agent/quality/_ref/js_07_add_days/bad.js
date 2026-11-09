function addDays(dateStr, days) {
  if (typeof dateStr !== 'string') throw new TypeError('dateStr must be a string');
  if (!Number.isInteger(days)) throw new TypeError('days must be an integer');
  const [y, m, d] = dateStr.split('-').map(Number);
  const dt = new Date(y, m - 1, d);
  dt.setDate(dt.getDate() + days);
  const pad = (n) => String(n).padStart(2, '0');
  return `${dt.getFullYear()}-${pad(dt.getMonth() + 1)}-${pad(dt.getDate())}`;
}

module.exports = { addDays };
