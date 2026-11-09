function decode(s) {
  return decodeURIComponent(s.replace(/\+/g, ' '));
}

function parseQuery(qs) {
  if (typeof qs !== 'string') throw new TypeError('string expected');
  if (qs.startsWith('?')) qs = qs.slice(1);
  const out = {};
  for (const pair of qs.split('&')) {
    if (pair === '') continue;
    const i = pair.indexOf('=');
    const key = decode(i === -1 ? pair : pair.slice(0, i));
    const value = decode(i === -1 ? '' : pair.slice(i + 1));
    if (key === '') continue;
    if (Object.prototype.hasOwnProperty.call(out, key)) {
      if (Array.isArray(out[key])) out[key].push(value);
      else out[key] = [out[key], value];
    } else {
      out[key] = value;
    }
  }
  return out;
}

module.exports = { parseQuery };
