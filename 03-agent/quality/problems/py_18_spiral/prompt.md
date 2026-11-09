Write a function `spiral_order(matrix)` that returns the elements of a matrix in clockwise spiral order.

`matrix` is a list of rows (lists), all of the same length, not necessarily square. Start at the top-left element, go right along the top row, then down the right column, then left along the bottom row, then up the left column, then continue inward in the same way until every element has been visited exactly once. Example: `[[1,2,3],[4,5,6],[7,8,9]]` -> `[1,2,3,6,9,8,7,4,5]`.

An empty matrix (`[]`) or a matrix whose rows are all empty (`[[]]`, `[[], []]`) gives `[]`. Raise `ValueError` if the rows do not all have the same length. The input must not be modified.
