Write a function `expandRanges(spec)` that expands a page-range style specification into a sorted list of numbers.

`spec` is a comma-separated list of parts. Each part is either a non-negative integer (`7`) or an inclusive range `lo-hi` of two non-negative integers (`3-6` means 3, 4, 5, 6). Whitespace around a part is ignored; there is no whitespace inside a part. Return the numbers as an array of numbers in ascending order with duplicates removed, so `"5,1-3,2"` -> `[1, 2, 3, 5]` and `"1-5,3-8"` -> `[1,2,3,4,5,6,7,8]`. A spec that is empty or only whitespace gives `[]`. Leading zeros are allowed (`"007"` is 7).

Throw a `RangeError` for a range whose start is greater than its end (`"5-3"`). Throw a `SyntaxError` for any other malformed part: an empty part (`"1,,2"`, a leading or trailing comma), letters, negative numbers, a plus sign, decimals, a missing end (`"1-"`), or more than one dash (`"1-2-3"`). Throw a `TypeError` if `spec` is not a string.
