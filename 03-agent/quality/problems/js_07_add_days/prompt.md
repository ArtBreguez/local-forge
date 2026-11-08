Write a function `addDays(dateStr, days)` that adds a number of days to a calendar date and returns the new date as `"YYYY-MM-DD"`.

`dateStr` must be exactly four digits, `-`, two digits, `-`, two digits. `days` is an integer and may be negative or zero. Do the calendar arithmetic in UTC so that daylight-saving changes in the local time zone can never matter. Years are four digits in the result, zero-padded (year 99 is `0099`).

Examples: `addDays("2024-02-28", 2)` -> `"2024-03-01"`, `addDays("2023-12-31", 1)` -> `"2024-01-01"`, `addDays("2024-03-01", -1)` -> `"2024-02-29"`.

Throw a `RangeError` if `dateStr` has the wrong shape (`"2024-2-3"`, `"20240203"`, `"2024-02-03T00:00"`), if it is not a real calendar date (`"2023-02-30"`, `"2023-13-01"`), or if the year of the input or of the result is outside 0001..9999. Throw a `TypeError` if `days` is not an integer number or `dateStr` is not a string.
