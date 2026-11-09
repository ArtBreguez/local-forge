Write two functions, `rle_encode(s)` and `rle_decode(s)`, for run-length encoding.

`rle_encode(s)` replaces every maximal run of the same character with its length (in decimal) followed by the character: `"aaabcc"` -> `"3a1b2c"`, `"x"` -> `"1x"`, `""` -> `""`. The input must not contain any digit character; raise `ValueError` if it does.

`rle_decode(s)` reverses it: the string is a sequence of `<count><char>` groups, where the count is one or more digits (it may have several, e.g. `"12x"` is twelve x's) and the char is any single non-digit character. `"3a1b2c"` -> `"aaabcc"`. The count must be at least 1 in value; leading zeros are allowed (`"01a"` -> `"a"`), but a count of zero (`"0a"`, `"00a"`) is not. Adjacent groups with the same character are fine (`"2a2a"` -> `"aaaa"`). Raise `ValueError` for a string that is not of that form: a count of 0 (`"0a"`), a missing count (`"abc"`, `"a3"`), or trailing digits with no character (`"3"`, `"2a3"`). The empty string decodes to `""`.

Both functions raise `TypeError` for non-`str` input.
