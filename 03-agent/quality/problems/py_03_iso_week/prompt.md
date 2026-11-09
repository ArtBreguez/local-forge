Write a function `iso_week(date_str)` that returns the ISO 8601 week of a calendar date as a string `"YYYY-Www"`.

`date_str` must be exactly four digits, a hyphen, two digits, a hyphen, two digits (`"2021-01-03"`). The result uses ISO week numbering: weeks start on Monday and week 1 is the week that contains the year's first Thursday, so the year in the result is the ISO week-year, which can differ from the calendar year around 1 January and 31 December. The week number is zero-padded to two digits. Examples: `"2021-01-01"` -> `"2020-W53"`, `"2018-12-31"` -> `"2019-W01"`, `"2021-01-04"` -> `"2021-W01"`.

Raise `ValueError` if the string does not have exactly that shape (for example `"2021-1-1"`, `"20210101"` or `"2021-01-01T10:00"`) or if it is not a real calendar date (`"2023-02-29"`). Raise `TypeError` if `date_str` is not a `str`.
