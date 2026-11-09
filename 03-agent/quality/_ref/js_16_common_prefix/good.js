function longestCommonPrefix(strs) {
  if (!Array.isArray(strs) || strs.some((s) => typeof s !== 'string')) {
    throw new TypeError('array of strings expected');
  }
  if (strs.length === 0) return '';
  const chars = strs.map((s) => Array.from(s));
  const first = chars[0];
  let n = 0;
  while (n < first.length && chars.every((c) => c[n] === first[n])) n += 1;
  return first.slice(0, n).join('');
}

module.exports = { longestCommonPrefix };
