Write a function `allocate(total, weights)` that splits an integer amount of cents between recipients in proportion to integer weights, using the largest remainder method so that nothing is lost.

For each recipient compute the exact share `total * weight / sum(weights)`; give everyone the floor of their share; the cents still left over (the total minus the sum of the floors) are handed out one each to the recipients with the largest fractional remainders, and when remainders are equal the recipient with the lower index goes first. Return the list of integer amounts (same order as `weights`); it always sums to `total`. Use integer arithmetic.

Examples: `allocate(100, [1, 1, 1])` -> `[34, 33, 33]`; `allocate(10, [3, 2, 2])` -> `[4, 3, 3]`; `allocate(5, [0, 3, 0])` -> `[0, 5, 0]`.

Raise `ValueError` if `total` is negative, `weights` is empty, any weight is negative, or all weights are zero. `total` and the weights are `int`s.
