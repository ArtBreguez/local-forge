Write a function `wrap_text(text, width)` that word-wraps text into lines.

Split `text` into words on runs of whitespace (spaces, tabs and newlines all count; leading and trailing whitespace is ignored). Build lines greedily: put as many consecutive words as fit on a line, separated by single spaces, so that the line is at most `width` characters long. A word that is longer than `width` is never split: it goes on a line by itself. Return the list of lines (no trailing newline characters). Text with no words gives `[]`.

Example: `wrap_text("the quick brown fox", 9)` -> `["the quick", "brown fox"]`.

Raise `ValueError` if `width` is less than 1 and `TypeError` if `width` is not an `int`.
