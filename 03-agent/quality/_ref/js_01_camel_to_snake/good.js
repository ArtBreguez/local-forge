function camelToSnake(str) {
  if (typeof str !== 'string') throw new TypeError('string expected');
  return str
    .replace(/([a-z0-9])([A-Z])/g, '$1_$2')
    .replace(/([A-Z]+)([A-Z][a-z])/g, '$1_$2')
    .toLowerCase();
}

module.exports = { camelToSnake };
