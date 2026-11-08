Write a function `validate_username(name)` that checks a username and returns its normalised form.

Normalise first: strip leading and trailing whitespace, then convert to lower case. Then the normalised name must satisfy all of these rules:
- length between 3 and 20 characters inclusive;
- only the ASCII characters `a-z`, `0-9`, `_` and `.`;
- starts with a letter `a-z`;
- does not end with `_` or `.`;
- never has two special characters (`_` or `.`) next to each other.

Return the normalised name when valid. Raise `ValueError` when a rule is violated and `TypeError` when `name` is not a `str`. Examples: `"  Alice_01 "` -> `"alice_01"`; `"a__b"` -> `ValueError`; `"Ünal"` -> `ValueError`.
