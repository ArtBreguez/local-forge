function isValidCardNumber(input) {
  if (typeof input !== 'string') return false;
  const digits = input.replace(/[ -]/g, '');
  if (!/^[0-9]{12,19}$/.test(digits)) return false;
  let sum = 0;
  for (let i = 0; i < digits.length; i += 1) {
    let d = digits.charCodeAt(i) - 48;
    if (i % 2 === 1) {
      d *= 2;
      if (d > 9) d -= 9;
    }
    sum += d;
  }
  return sum % 10 === 0;
}

module.exports = { isValidCardNumber };
