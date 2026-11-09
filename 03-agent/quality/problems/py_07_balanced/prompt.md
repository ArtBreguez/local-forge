Write a function `is_balanced(code)` that returns `True` if the round, square and curly brackets in `code` are correctly nested and balanced, and `False` otherwise.

Brackets are `()`, `[]` and `{}`. Every other character is ignored, except that brackets inside a double-quoted string literal do not count. A string literal starts at a `"` and ends at the next `"` that is not escaped; inside a string a backslash escapes the following character (so `\"` does not end the string, and `\\` is an escaped backslash). A string literal that is never closed makes the input unbalanced.

Examples: `'f(a[1], {"k": "]"})'` -> `True`; `'("a)'` -> `False`; `'(]'` -> `False`; `''` -> `True`.
