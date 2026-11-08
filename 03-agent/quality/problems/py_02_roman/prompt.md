Write a function `roman_to_int(s)` that converts a Roman numeral to an integer.

Only canonical numerals in the range 1..3999 are valid: upper-case letters `I V X L C D M`, written the standard way (additive notation with the six subtractive pairs IV, IX, XL, XC, CD, CM, and no more than three repeats of I, X, C or M in a row). Examples: `"IV"` -> 4, `"MCMXCIV"` -> 1994, `"MMMCMXCIX"` -> 3999.

Raise `ValueError` for the empty string, lower-case input, any other character, and any non-canonical numeral such as `"IIII"`, `"VV"`, `"IC"`, `"IL"`, `"XM"` or `"MMMM"`. Raise `TypeError` if `s` is not a `str`.
