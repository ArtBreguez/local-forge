// compile(pattern) -> match(path)
// A pattern is a path such as '/users/:id/posts'; a segment that starts with ':' captures that part of the path.
// match(path) returns an object with the captured params (values URI-decoded), or null if the path does not match.
// Empty segments are ignored, so '/a/b/' and '/a//b' are the same as '/a/b'.
function split(path) {
  return path.split('/').filter((segment) => segment !== '');
}

function compile(pattern) {
  const wanted = split(pattern);
  return function match(path) {
    const parts = split(path);
    if (parts.length !== wanted.length) return null;
    const params = {};
    for (let i = 0; i < wanted.length; i++) {
      if (wanted[i].startsWith(':')) {
        params[wanted[i].slice(1)] = decodeURIComponent(parts[i]);
      } else if (wanted[i] !== parts[i]) {
        return null;
      }
    }
    return params;
  };
}

module.exports = { compile, split };
