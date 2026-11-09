Implement `slugify(text, options)` in `src/slugify.js` (CommonJS, `module.exports = { slugify }`); the file holds a stub.

- Remove accents (decompose with Unicode NFD and drop the combining marks), then lower-case the text.
- Every run of characters that are not a-z or 0-9 becomes one separator (default `-`, option `separator`); the result has no separator at its start or end. Digits are kept.
- Option `maxLength` (a positive integer): cut the result to at most that many characters, then remove a separator left at the end by the cut.
- Throw `TypeError` if `text` is not a string; throw `RangeError` if `maxLength` is given and is not a positive integer.

Examples: `slugify("Crème Brûlée!")` -> `"creme-brulee"`; `slugify("Hello World", { separator: "_" })` -> `"hello_world"`; `slugify("hello world", { maxLength: 6 })` -> `"hello"`.

The visible tests are in `test/slugify.test.js` (run them with `node --test`); more tests that you cannot see check the same behaviour. Do not modify anything under `test/`.
