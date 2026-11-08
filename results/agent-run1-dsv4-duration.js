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
  if (typeof text !== 'string') {
    throw new TypeError('not a string');
  }
  
  const parts = text.trim().split(/\s+/).filter(Boolean);
  
  if (parts.length === 0) {
    throw new RangeError('invalid text');
  }
  
  let total = 0;
  const seenUnits = new Set();
  
  for (const part of parts) {
    // Check if part is a valid number with unit
    const match = part.match(/^(\d+)([hms])$/);
    if (!match || !['h', 'm', 's'].includes(match[2])) {
      throw new RangeError('invalid text');
    }
    
    const value = parseInt(match[1], 10);
    const unit = match[2];
    
    if (value < 0 || value !== value) {
      throw new RangeError('negative or fractional number');
    }
    
    if (seenUnits.has(unit)) {
      throw new RangeError('repeated unit: ' + unit);
    }
    
    seenUnits.add(unit);
    total += value * (unit === 'h' ? 3600 : unit === 'm' ? 60 : 1);
  }
  
  return total;
}
