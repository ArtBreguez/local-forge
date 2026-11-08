def to_24h(text):
    if not isinstance(text, str):
        raise TypeError("text must be a str")
    s = text.strip().upper()
    if s.endswith("AM") or s.endswith("PM"):
        body, suffix = s[:-2].strip(), s[-2:]
    else:
        raise ValueError(text)
    hour_s, _, minute_s = body.partition(":")
    hour, minute = int(hour_s), int(minute_s)
    if suffix == "PM":
        hour += 12
    return f"{hour:02d}:{minute:02d}"
