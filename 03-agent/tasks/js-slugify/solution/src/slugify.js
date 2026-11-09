// slugify(text, options) -> string (see the task description)
function slugify(text, options = {}) {
  if (typeof text !== 'string') throw new TypeError('text must be a string');
  const { separator = '-', maxLength } = options;
  if (maxLength !== undefined && (!Number.isInteger(maxLength) || maxLength < 1)) {
    throw new RangeError('maxLength must be a positive integer');
  }
  const words = text
    .normalize('NFD')
    .replace(/[̀-ͯ]/g, '')
    .toLowerCase()
    .split(/[^a-z0-9]+/)
    .filter(Boolean);
  let slug = words.join(separator);
  if (maxLength !== undefined && slug.length > maxLength) {
    slug = slug.slice(0, maxLength);
    while (separator && slug.endsWith(separator)) slug = slug.slice(0, -separator.length);
  }
  return slug;
}

module.exports = { slugify };
