from datetime import datetime


def iso_week(date_str):
    if not isinstance(date_str, str):
        raise TypeError("date_str must be a str")
    d = datetime.strptime(date_str, "%Y-%m-%d")
    return d.strftime("%Y-W%W")
