import re
from datetime import date

_SHAPE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}")


def iso_week(date_str):
    if not isinstance(date_str, str):
        raise TypeError("date_str must be a str")
    if _SHAPE.fullmatch(date_str) is None:
        raise ValueError(f"bad date: {date_str!r}")
    y, m, d = (int(p) for p in date_str.split("-"))
    year, week, _ = date(y, m, d).isocalendar()
    return f"{year:04d}-W{week:02d}"
