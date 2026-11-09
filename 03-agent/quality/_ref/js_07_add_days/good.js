function addDays(dateStr, days) {
  if (typeof dateStr !== 'string') throw new TypeError('dateStr must be a string');
  if (!Number.isInteger(days)) throw new TypeError('days must be an integer');
  const m = /^(\d{4})-(\d{2})-(\d{2})$/.exec(dateStr);
  if (!m) throw new RangeError('bad date shape');
  const y = Number(m[1]);
  const mo = Number(m[2]);
  const d = Number(m[3]);
  if (y < 1) throw new RangeError('year out of range');
  const dt = new Date(0);
  dt.setUTCFullYear(y, mo - 1, d);
  dt.setUTCHours(0, 0, 0, 0);
  if (dt.getUTCFullYear() !== y || dt.getUTCMonth() !== mo - 1 || dt.getUTCDate() !== d) {
    throw new RangeError('no such date');
  }
  dt.setUTCDate(dt.getUTCDate() + days);
  const year = dt.getUTCFullYear();
  if (year < 1 || year > 9999) throw new RangeError('result out of range');
  const pad = (n, w) => String(n).padStart(w, '0');
  return `${pad(year, 4)}-${pad(dt.getUTCMonth() + 1, 2)}-${pad(dt.getUTCDate(), 2)}`;
}

module.exports = { addDays };
