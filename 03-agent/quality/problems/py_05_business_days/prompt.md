Write a function `business_days(start, end, holidays=())` that counts working days in a half-open range of dates.

`start` and `end` are ISO date strings (`"YYYY-MM-DD"`); `holidays` is an iterable of ISO date strings. Count the dates `d` with `start <= d < end` that fall on Monday to Friday and are not in `holidays`. The end date itself is not counted. Holidays that fall on a weekend, fall outside the range, or are listed more than once must not change the result beyond what the rule above implies. If `start == end` the answer is 0.

Raise `ValueError` if `end` is earlier than `start` or if any date string is invalid.
