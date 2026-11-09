from datetime import date, timedelta


def business_days(start, end, holidays=()):
    s = date.fromisoformat(start)
    e = date.fromisoformat(end)
    if e < s:
        raise ValueError("end is before start")
    holiday_set = {date.fromisoformat(h) for h in holidays}
    count = 0
    d = s
    while d < e:
        if d.weekday() < 5 and d not in holiday_set:
            count += 1
        d += timedelta(days=1)
    return count
