import calendar
from datetime import date


def add_months(date_str, n):
    if isinstance(n, bool) or not isinstance(n, int):
        raise TypeError("n must be an int")
    d = date.fromisoformat(date_str)
    total = d.year * 12 + (d.month - 1) + n
    year, month0 = divmod(total, 12)
    if not 1 <= year <= 9999:
        raise ValueError("result out of range")
    month = month0 + 1
    day = min(d.day, calendar.monthrange(year, month)[1])
    return date(year, month, day).isoformat()
