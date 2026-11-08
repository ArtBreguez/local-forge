Write a function `formatBytes(n)` that formats a number of bytes for humans using binary units (1 KB = 1024 B).

The units are `B`, `KB`, `MB`, `GB`, `TB`. A value below 1024 is shown in bytes as `"<n> B"`. Otherwise divide by 1024 until the value is below 1024 or the unit is `TB`, round to at most one decimal place, and leave the decimal out when it is `.0`. If rounding turns the value into 1024 and a larger unit exists, move up one unit instead (so 1048575 bytes is `"1 MB"`, not `"1024 KB"`). Values beyond terabytes stay in `TB` (2^50 bytes is `"1024 TB"`).

Examples: `0` -> `"0 B"`, `1023` -> `"1023 B"`, `1024` -> `"1 KB"`, `1536` -> `"1.5 KB"`, `1075` -> `"1 KB"`, `1572864` -> `"1.5 MB"`.

Throw a `TypeError` if `n` is not a finite number (including `NaN`, `Infinity` and numeric strings). Throw a `RangeError` if `n` is negative or not an integer.
