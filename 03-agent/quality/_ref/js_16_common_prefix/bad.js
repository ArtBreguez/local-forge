function longestCommonPrefix(strs) {
  if (!Array.isArray(strs) || strs.some((s) => typeof s !== 'string')) {
    throw new TypeError('array of strings expected');
  }
  if (strs.length === 0) return '';
  let prefix = strs[0];
  for (const s of strs) {
    while (!s.startsWith(prefix)) prefix = prefix.slice(0, -1);
  }
  return prefix;
}

module.exports = { longestCommonPrefix };
