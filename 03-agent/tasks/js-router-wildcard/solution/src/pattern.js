// compile(pattern) -> match(path)
// A pattern is a path such as '/users/:id/posts'; a segment that starts with ':' captures that part of the path.
// match(path) returns an object with the captured params (values URI-decoded), or null if the path does not match.
// A last segment '*' matches one or more remaining segments; they are returned as params.splat, joined with '/'.
// Empty segments are ignored, so '/a/b/' and '/a//b' are the same as '/a/b'.
function split(path) {
  return path.split('/').filter((segment) => segment !== '');
}

function compile(pattern) {
  const wanted = split(pattern);
  const star = wanted.indexOf('*');
  if (star !== -1 && star !== wanted.length - 1) throw new Error(`'*' must be the last segment: ${pattern}`);
  const wild = star !== -1;
  const fixed = wild ? wanted.length - 1 : wanted.length;
  return function match(path) {
    const parts = split(path);
    if (wild ? parts.length <= fixed : parts.length !== fixed) return null;
    const params = {};
    for (let i = 0; i < fixed; i++) {
      if (wanted[i].startsWith(':')) {
        params[wanted[i].slice(1)] = decodeURIComponent(parts[i]);
      } else if (wanted[i] !== parts[i]) {
        return null;
      }
    }
    if (wild) params.splat = parts.slice(fixed).map(decodeURIComponent).join('/');
    return params;
  };
}

module.exports = { compile, split };
