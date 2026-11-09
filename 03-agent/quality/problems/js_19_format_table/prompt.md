Write a function `formatTable(rows)` that lays out rows of cells as aligned plain text.

`rows` is an array of arrays. Convert each cell with `String(cell)`, except that `null` and `undefined` become the empty string. Each column is as wide as its longest cell (measured with `.length`); rows with fewer cells than the widest row are treated as if the missing cells were empty strings. Pad every cell on the right with spaces to its column width, join the cells of a row with `" | "`, then remove trailing spaces from the end of the line. Join the lines with `"\n"` (no trailing newline). No rows gives `""`.

Example: `formatTable([["name","qty"],["apple",3],["kiwi",12]])` is
```
name  | qty
apple | 3
kiwi  | 12
```
(three lines; the first column is padded to 5 characters; the last column is not padded at the end of the line).

Throw a `TypeError` if `rows` is not an array of arrays.
