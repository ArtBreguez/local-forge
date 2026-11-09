Write a function `render(template, data)` that fills `{{ placeholders }}` in a template string.

- A placeholder is `{{`, optional whitespace, a path, optional whitespace, `}}`. A path is one or more segments separated by dots; a segment is any run of characters other than `{`, `}`, `.` and whitespace. `{{ user.name }}` and `{{user.name}}` are the same placeholder. Text that looks like a placeholder but has no valid path (`{{}}`, `{{ a b }}`) is left exactly as it is.
- The path is looked up in `data` one segment at a time using **own** properties only (inherited properties such as `toString` or `constructor` do not count). Array elements are reached by index (`items.1`).
- If any step of the lookup is missing, or the final value is `null` or `undefined`, the placeholder becomes an empty string. Any other value is converted with `String(value)`, so `0` renders as `0` and `false` as `false`. No HTML escaping is done, and the value is inserted literally (a value containing `$&` or `$1` stays as it is).

Throw a `TypeError` if `template` is not a string. `data` may be any value, including `null`.
