function render(template, data) {
  if (typeof template !== 'string') throw new TypeError('template must be a string');
  return template.replace(/\{\{\s*([^{}.\s]+(?:\.[^{}.\s]+)*)\s*\}\}/g, (match, path) => {
    let current = data;
    for (const segment of path.split('.')) {
      if (current === null || current === undefined) return '';
      if (!Object.prototype.hasOwnProperty.call(Object(current), segment)) return '';
      current = current[segment];
    }
    return current === null || current === undefined ? '' : String(current);
  });
}

module.exports = { render };
