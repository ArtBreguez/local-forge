// Duration parsing for the agent test. Implement parseDuration.
//
// parseDuration(text) -> number of seconds (integer)
// Units: h (hours), m (minutes), s (seconds). Units may appear in any order once each,
// with optional whitespace between parts: "1h30m", "90s", "2h 5s", "45m".
// A bare number means seconds: "120" -> 120.
// Throw a TypeError when text is not a string.
// Throw a RangeError for anything else that is not a valid duration:
// empty string, unknown unit, a repeated unit, a negative or fractional number.

export function parseDuration(text) {
  if (typeof text !== 'string') throw new TypeError('text must be a string');
  const s = text.trim();
  if (s === '') throw new RangeError('empty duration');
  if (/^\d+$/.test(s)) return Number(s);
  const seconds = { h: 3600, m: 60, s: 1 };
  const seen = new Set();
  const part = /(\d+)([^\d\s]*)\s*/y;
  let total = 0;
  let pos = 0;
  while (pos < s.length) {
    part.lastIndex = pos;
    const m = part.exec(s);
    if (!m) throw new RangeError(`not a duration: ${text}`);
    const unit = m[2];
    if (!Object.hasOwn(seconds, unit) || seen.has(unit)) throw new RangeError(`bad unit in: ${text}`);
    seen.add(unit);
    total += Number(m[1]) * seconds[unit];
    pos = part.lastIndex;
  }
  return total;
}
