function lowerBound(arr, x) {
  if (!Array.isArray(arr)) throw new TypeError('array expected');
  if (typeof x !== 'number') throw new TypeError('number expected');
  let lo = 0;
  let hi = arr.length;
  while (lo < hi) {
    const mid = (lo + hi) >>> 1;
    if (arr[mid] < x) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

module.exports = { lowerBound };
