def validate_username(name):
    if not isinstance(name, str):
        raise TypeError("name must be a str")
    s = name.strip().lower()
    if not 3 <= len(s) <= 20:
        raise ValueError("length must be 3..20")
    if not s[0].isalpha():
        raise ValueError("must start with a letter")
    if not all(c.isalnum() or c in "_." for c in s):
        raise ValueError("bad character")
    return s
