Write a function `longestCommonPrefix(strs)` that returns the longest string that every string in the array starts with.

The comparison is case-sensitive and is done on whole Unicode characters (code points), not on UTF-16 code units: the result must never end in the middle of a surrogate pair, so for emoji such as `"\u{1F600}a"` and `"\u{1F601}a"` the common prefix is `""`, not half of a character. An empty array gives `""`, and a one-element array gives that element. If any string is empty the result is `""`.

Examples: `["flower","flow","flight"]` -> `"fl"`, `["dog","racecar","car"]` -> `""`.

Throw a `TypeError` if the argument is not an array of strings.
