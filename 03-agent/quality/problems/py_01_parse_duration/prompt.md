Write a function `parse_duration(text)` that converts a compact duration string into a whole number of seconds.

The string is a sequence of one or more `<integer><unit>` parts with no spaces between or around them. The unit is one of the lowercase letters `d` (86400 seconds), `h` (3600), `m` (60) or `s` (1). Units must appear in strictly descending order (d, then h, then m, then s) and each unit may appear at most once.

Examples: `"1h30m"` -> 5400, `"90s"` -> 90, `"2d"` -> 172800, `"0s"` -> 0.

Raise `ValueError` for anything else: an empty string, a number without a unit or a unit without a number, unknown units (including `ms`), repeated or out-of-order units, signs, decimals, upper-case units, or any whitespace. Raise `TypeError` if `text` is not a `str`.
