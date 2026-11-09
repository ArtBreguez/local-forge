function ageOn(birth, on) {
  const b = new Date(birth);
  const o = new Date(on);
  if (Number.isNaN(b.getTime()) || Number.isNaN(o.getTime())) throw new RangeError('bad date');
  if (o < b) throw new RangeError('on is earlier than birth');
  let age = o.getUTCFullYear() - b.getUTCFullYear();
  if (o.getUTCMonth() < b.getUTCMonth() || (o.getUTCMonth() === b.getUTCMonth() && o.getUTCDate() < b.getUTCDate())) age -= 1;
  return age;
}

module.exports = { ageOn };
