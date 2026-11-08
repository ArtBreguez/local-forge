function isValidCardNumber(input) {
  if (typeof input !== 'string') return false;
  const digits = input.replace(/[ -]/g, '');
  if (!/^[0-9]{12,19}$/.test(digits)) return false;
  let sum = 0;
  let double = false;
  for (let i = digits.length - 1; i >= 0; i -= 1) {
    let d = digits.charCodeAt(i) - 48;
    if (double) {
      d *= 2;
      if (d > 9) d -= 9;
    }
    sum += d;
    double = !double;
  }
  return sum % 10 === 0;
}

module.exports = { isValidCardNumber };
