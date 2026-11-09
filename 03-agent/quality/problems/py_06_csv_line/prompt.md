Write a function `parse_csv_line(line)` that splits one line of CSV text into a list of field strings. Do not use the `csv` module.

Rules:
- Fields are separated by commas. `"a,,b"` -> `["a", "", "b"]`; a trailing comma yields a trailing empty field (`"a,"` -> `["a", ""]`); the empty string yields one empty field (`[""]`).
- A field that starts with a double quote is a quoted field: it extends to the matching closing quote, may contain commas, and a doubled quote (`""`) inside it stands for one literal quote character. The surrounding quotes are not part of the value.
- Any other field is taken literally up to the next comma (spaces are preserved; a quote character in the middle of an unquoted field is just an ordinary character).
- After the closing quote of a quoted field there must be either a comma or the end of the line; anything else is an error.
- A quoted field that is never closed is an error.

Raise `ValueError` for the two error cases and `TypeError` if `line` is not a `str`. The line contains no newline characters.
