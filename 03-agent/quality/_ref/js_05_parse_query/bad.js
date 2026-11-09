function parseQuery(qs) {
  if (typeof qs !== 'string') throw new TypeError('string expected');
  if (qs.startsWith('?')) qs = qs.slice(1);
  const out = {};
  for (const pair of qs.split('&')) {
    if (pair === '') continue;
    const i = pair.indexOf('=');
    const key = decodeURIComponent(i === -1 ? pair : pair.slice(0, i)).replace(/\+/g, ' ');
    const value = decodeURIComponent(i === -1 ? '' : pair.slice(i + 1)).replace(/\+/g, ' ');
    if (key === '') continue;
    if (key in out) {
      out[key] = [].concat(out[key], value);
    } else {
      out[key] = value;
    }
  }
  return out;
}

module.exports = { parseQuery };
