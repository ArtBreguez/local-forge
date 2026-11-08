Write a function `to_24h(text)` that converts a 12-hour clock time to a 24-hour `"HH:MM"` string.

The input has the form `H:MM` or `HH:MM`, optionally followed by a single space (at most one), then `AM` or `PM` in any letter case (`"1:05pm"`, `"11:59 PM"`, `"9:00AM"`, `"07:30 am"`). Surrounding whitespace is ignored. The hour must be 1 to 12 and the minutes exactly two digits 00 to 59. `12:xx AM` is just after midnight (`"12:30 AM"` -> `"00:30"`), `12:xx PM` is just after noon (`"12:30 PM"` -> `"12:30"`); other `PM` hours get 12 added. The result is always two-digit hours and two-digit minutes.

Raise `ValueError` for anything else (hour 0 or above 12, minutes above 59 or with one digit, missing AM/PM, an unknown suffix, missing colon, empty text) and `TypeError` if `text` is not a `str`.
