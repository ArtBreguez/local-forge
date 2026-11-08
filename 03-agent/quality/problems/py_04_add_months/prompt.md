Write a function `add_months(date_str, n)` that adds `n` calendar months to an ISO date and returns the result as an ISO date string.

`date_str` is `"YYYY-MM-DD"`; `n` is an integer and may be negative or zero. The year and month move by `n` months; the day of the month stays the same unless the target month is shorter, in which case it is clamped to the last day of the target month. Examples: `add_months("2024-01-31", 1)` -> `"2024-02-29"`, `add_months("2023-01-31", 1)` -> `"2023-02-28"`, `add_months("2024-03-31", -1)` -> `"2024-02-29"`, `add_months("2024-12-15", 1)` -> `"2025-01-15"`.

Raise `ValueError` if `date_str` is not a valid date or if the result would fall outside years 1..9999. Raise `TypeError` if `n` is not an `int` (a `bool` does not count as an int here).
