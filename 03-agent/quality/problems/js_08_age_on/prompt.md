Write a function `ageOn(birth, on)` that returns a person's age in whole years on a given date.

`birth` and `on` are `"YYYY-MM-DD"` strings. The age is the number of birthdays that have already happened on or before `on`, counting the day of the birthday itself as a completed year. Someone born on 29 February has their birthday on 1 March in years that are not leap years.

Examples: `ageOn("2000-01-15", "2024-01-14")` -> `23`, `ageOn("2000-01-15", "2024-01-15")` -> `24`, `ageOn("2000-02-29", "2023-02-28")` -> `22`, `ageOn("2000-02-29", "2023-03-01")` -> `23`.

Throw a `RangeError` if either string is not a real calendar date in `YYYY-MM-DD` form, or if `on` is earlier than `birth`.
