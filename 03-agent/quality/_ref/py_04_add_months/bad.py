from datetime import date


def add_months(date_str, n):
    if isinstance(n, bool) or not isinstance(n, int):
        raise TypeError("n must be an int")
    d = date.fromisoformat(date_str)
    month = d.month - 1 + n
    year = d.year + month // 12
    month = month % 12 + 1
    return d.replace(year=year, month=month).isoformat()
