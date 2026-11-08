function parse(s) {
  const m = /^(\d{4})-(\d{2})-(\d{2})$/.exec(s);
  if (!m) throw new RangeError(`bad date: ${s}`);
  const [y, mo, d] = [Number(m[1]), Number(m[2]), Number(m[3])];
  const dt = new Date(Date.UTC(2000, 0, 1));
  dt.setUTCFullYear(y, mo - 1, d);
  if (dt.getUTCFullYear() !== y || dt.getUTCMonth() !== mo - 1 || dt.getUTCDate() !== d) {
    throw new RangeError(`no such date: ${s}`);
  }
  return { y, mo, d };
}

function isLeap(y) {
  return (y % 4 === 0 && y % 100 !== 0) || y % 400 === 0;
}

function ageOn(birth, on) {
  const b = parse(birth);
  const o = parse(on);
  if (o.y < b.y || (o.y === b.y && (o.mo < b.mo || (o.mo === b.mo && o.d < b.d)))) {
    throw new RangeError('on is earlier than birth');
  }
  let bm = b.mo;
  let bd = b.d;
  if (bm === 2 && bd === 29 && !isLeap(o.y)) {
    bm = 3;
    bd = 1;
  }
  let age = o.y - b.y;
  if (o.mo < bm || (o.mo === bm && o.d < bd)) age -= 1;
  return age;
}

module.exports = { ageOn };
